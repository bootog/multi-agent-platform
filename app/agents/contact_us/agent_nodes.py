"""Chat-agent nodes for the Contact Us Agent.

The existing workflow nodes (nodes.py) do the Bootog work unchanged. The nodes here add
the conversational layer around them:

  understand_message      LLM -> typed TurnUnderstanding (intent, references, fields)
  resolve_provider        company named in chat -> provider id (existing provider tool)
  match_request           deterministic match of the user's reference to a request
  merge_customer_fields   user-supplied values over the request data (latest wins)
  validate_customer_data  required/format checks on the customer data
  ask_for_missing         records the pending question; the turn waits for the user
  respond                 LLM reply written only from structured facts

Every tracked node publishes Live Tracking events through `tracked_step`."""

import re
from datetime import date
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.contact_us.nodes import StepFailed, StepResult, tracked_step
from app.agents.contact_us.schemas import CUSTOMER_FIELDS, build_customer_draft, summarize_contact_request
from app.agents.contact_us.state import ContactUsChatState
from app.agents.contact_us.tools import get_provider_list
from app.agents.contact_us.understanding import INTENT_LABELS, compose_reply, understand
from app.agents.runtime import RunChannel, RunEmitter
from app.core.logging import get_logger
from app.integrations.bootog.customers import CUSTOMER_API_NOT_CONFIGURED
from app.llm import LlmNotConfiguredError

logger = get_logger(__name__)

FIELD_LABELS: dict[str, str] = {key: label for key, _, label, _ in CUSTOMER_FIELDS}
REQUIRED_FIELDS: list[str] = [key for key, _, _, required in CUSTOMER_FIELDS if required]
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_LIST_LIMIT = 25  # requests sent to the UI per listing
_FACTS_LIST_LIMIT = 10  # requests described to the LLM per listing

# ContactUsStatus in the host's libs/models/enums/contact-us-status.model.ts.
CONTACT_US_STATUS_LABELS: dict[int, str] = {
    1: "Proposal Submitted", 2: "Pending Approval", 3: "Needs Clarification", 4: "Under Review",
    5: "Approved", 6: "Rejected", 7: "In Implementation", 8: "Completed", 9: "On Hold",
    10: "Cancelled", 11: "CRM Completed", 12: "Closed", 13: "Re Open", 14: "Re Scheduled",
    15: "Assigned", 16: "Left Message", 17: "Email Sent", 18: "Document Requested",
}

ACTIVITY_TITLES: dict[str, str] = {
    "start_turn": "Agent Session",
    "understand_message": "Understanding Request",
    "resolve_provider": "Company Selection",
    "select_provider": "Provider Selection",
    "get_contact_us_data": "Contact Request Retrieval",
    "match_request": "Request Matching",
    "select_contact_request": "Contact Request Selected",
    "prepare_customer_data": "Customer Information Extracted",
    "merge_customer_fields": "Information Received",
    "validate_customer_data": "Customer Data Validation",
    "ask_for_missing": "Waiting for User",
    "awaiting_user": "Waiting for User",
    "create_customer": "Customer Creation",
    "verify_customer": "Customer Verification",
}

CAPABILITIES = [
    "List open Contact Us requests (recent, by date range, by company, or only incomplete ones)",
    "Find a request by person name, email or position in a list",
    "Show the details of a request",
    "Create a customer from a request, asking for any missing required details",
    "Remember corrections and extra notes during the conversation",
]


class ChatEmitter(RunEmitter):
    """Labels every step event for the session activity feed (title + a unique
    activityId, since a step can run more than once per turn) and remembers how
    each step ended."""

    def __init__(self, channel: RunChannel):
        super().__init__(channel)
        self.outcomes: list[tuple[str, str]] = []
        self._seq: dict[str, int] = {}
        self._open: set[str] = set()

    async def step(
        self,
        step: str,
        status: str,
        message: str,
        data: dict[str, Any] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        if status == "running" or step not in self._open:
            self._seq[step] = self._seq.get(step, 0) + 1
        if status == "running":
            self._open.add(step)
        else:
            self._open.discard(step)
            self.outcomes.append((step, status))
        event: dict[str, Any] = {
            "type": "step",
            "step": step,
            "activityId": f"{step}-{self._seq[step]}",
            "title": ACTIVITY_TITLES.get(step, step.replace("_", " ").title()),
            "status": status,
            "message": message,
        }
        if duration_ms is not None:
            event["durationMs"] = duration_ms
        if data:
            event["data"] = data
        await self._publish(event)


def _emitter(config: RunnableConfig) -> ChatEmitter:
    return config["configurable"]["emitter"]


# --- helpers -----------------------------------------------------------------


def full_name(record: dict[str, Any] | None) -> str:
    return " ".join(p for p in ((record or {}).get("firstName"), (record or {}).get("lastName")) if p) or "Unnamed request"


def _tokens(text: str | None) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").casefold()) if len(t) > 1]


def _name_matches(reference: str, record: dict[str, Any]) -> bool:
    wanted = _tokens(reference)
    have = _tokens(f"{record.get('firstName') or ''} {record.get('lastName') or ''}")
    return bool(wanted) and all(any(h == w or (len(w) >= 3 and h.startswith(w)) for h in have) for w in wanted)


def _iso_date(value: str | None) -> str | None:
    """Only well-formed dates reach the Bootog filter expression."""
    try:
        return date.fromisoformat((value or "").strip()).isoformat()
    except ValueError:
        return None


def _snake(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().casefold()).strip("_")[:60]


def request_summary(record: dict[str, Any]) -> dict[str, Any]:
    _, missing = build_customer_draft(record)
    status = record.get("status")
    return {
        **summarize_contact_request(record),
        "statusLabel": CONTACT_US_STATUS_LABELS.get(status) if isinstance(status, int) else None,
        "missingFields": missing,
    }


def customer_draft(state: ContactUsChatState) -> dict[str, Any] | None:
    """Request data overlaid with what the user supplied. Only CUSTOMER_FIELDS keys."""
    record = state.get("contact_request")
    if not record:
        return None
    draft, _ = build_customer_draft(record)
    for key, value in (state.get("user_customer_fields") or {}).items():
        if key in draft and value:
            draft[key] = value
    return draft


def listed_records(state: ContactUsChatState) -> list[dict[str, Any]]:
    records = state.get("contact_requests") or []
    if (state.get("request_filters") or {}).get("incomplete_only"):
        records = [r for r in records if build_customer_draft(r)[1]]
    return records


def _state_summary(state: ContactUsChatState) -> dict[str, Any]:
    """What the understanding LLM sees about the agent's state (no secrets)."""
    by_id = {r.get("id"): r for r in state.get("contact_requests") or []}
    listed = [by_id[i] for i in state.get("listed_request_ids") or [] if i in by_id]
    selected = state.get("contact_request")
    draft = customer_draft(state) or {}
    return {
        "agent": _agent(state),
        "company": state.get("provider_name") or "All companies available to the user",
        "ongoing_goal": state.get("operation") or "none",
        "selected_request": (
            {"id": selected.get("id"), "name": full_name(selected), "email": selected.get("emailId")} if selected else None
        ),
        "pending_question": state.get("pending_question"),
        "last_listed_requests": [
            {"position": i + 1, "id": r.get("id"), "name": full_name(r), "email": r.get("emailId"), "created": r.get("createdDate")}
            for i, r in enumerate(listed[:_LIST_LIMIT])
        ],
        "customer_fields": {FIELD_LABELS[k]: v for k, v in draft.items()},
        "extra_information": state.get("extra_information") or {},
    }


def _agent(state: ContactUsChatState) -> dict[str, str]:
    name = state.get("agent_name") or "Contact Us Agent"
    return {"name": name, "handles": f"{name.removesuffix(' Agent')} requests"}


# --- nodes -------------------------------------------------------------------


async def start_turn(state: ContactUsChatState, config: RunnableConfig) -> dict[str, Any]:
    """Resets per-turn fields; cross-turn memory (selection, fields, notes) is kept."""
    if not state.get("workflow_status"):
        await _emitter(config).step("start_turn", "completed", "Contact Us Agent session started")
    return {
        "execution_status": "running",
        "error": None,
        "current_step": "start_turn",
        "intent": None,
        "request_summary": None,
        "request_reference": None,
        "request_filters": None,
        "company_name": None,
        "match_result": None,
        "turn_field_updates": {},
        "turn_extra_information": {},
        "tool_errors": [],
        "last_tool_result": None,
        "agent_response": None,
        "completed_steps": [],
        "failed_steps": [],
        "workflow_status": "running",
        "is_complete": False,
        "turn_steps": ["start_turn"],
    }


@tracked_step("understand_message", "Understanding your message", "I couldn't interpret that message.")
async def understand_message(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    try:
        result = await understand(config, state.get("conversation_history") or [], _state_summary(state))
    except LlmNotConfiguredError as exc:
        raise StepFailed("The AI service is not configured, so I can't process messages yet.") from exc

    update: dict[str, Any] = {
        "intent": result.intent,
        "request_summary": result.request_summary.strip()[:160] or None,
        # The user has replied, so whatever was pending is consumed by this turn.
        "pending_question": None,
        "awaiting_user_input": False,
    }
    if result.intent == "create_customer" or result.goal == "create_customer":
        update["operation"] = "create_customer"
    elif result.intent == "cancel_operation" or result.goal == "none":
        update["operation"] = None

    if result.customer_field_updates is not None:
        update["turn_field_updates"] = {
            key: value.strip()[:200]
            for key, value in result.customer_field_updates.model_dump().items()
            if key in FIELD_LABELS and isinstance(value, str) and value.strip()
        }
    update["turn_extra_information"] = {
        _snake(item.key): item.value.strip()[:300]
        for item in result.extra_information
        if _snake(item.key) and item.value.strip()
    }

    ref = result.request_reference
    if ref and (ref.use_current or ref.name or ref.email or ref.request_id or ref.position):
        update["request_reference"] = ref.model_dump()

    if result.filters:
        f = result.filters
        filters = {
            "created_from": _iso_date(f.created_from),
            "created_to": _iso_date(f.created_to),
            "incomplete_only": f.incomplete_only,
        }
        update["request_filters"] = {k: v for k, v in filters.items() if v}
        if f.company_name and f.company_name.strip():
            update["company_name"] = f.company_name.strip()[:200]

    message = update["request_summary"] or INTENT_LABELS[result.intent]
    return StepResult(message, update, data={"intent": result.intent})


@tracked_step("resolve_provider", "Finding the company", "Unable to look up companies.")
async def resolve_provider(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    wanted = (state.get("company_name") or "").strip()
    if wanted.casefold() in ("all", "all companies", "any", "any company", "every company"):
        provider_id, provider_name = None, None
    else:
        providers = await get_provider_list.ainvoke({}, config=config)
        exact = [p for p in providers if p["providerName"].casefold() == wanted.casefold()]
        found = exact or [p for p in providers if wanted.casefold() in p["providerName"].casefold()]
        if len(found) != 1:
            status = "ambiguous" if found else "not_found"
            note = {"company": {"status": status, "requested": wanted, "candidates": [p["providerName"] for p in found[:5]]}}
            kept = state.get("provider_name") or "all companies"
            return StepResult(f"No unique company matched “{wanted}” — keeping {kept}", {"last_tool_result": note})
        provider_id, provider_name = found[0]["providerId"], found[0]["providerName"]

    update: dict[str, Any] = {"provider_id": provider_id, "provider_name": provider_name}
    if provider_id != state.get("provider_id"):
        # Loaded requests belong to the previous company.
        update.update(contact_requests=[], contact_requests_total=0, listed_request_ids=[])
    return StepResult(f"Using {provider_name or 'all companies'}", update)


@tracked_step("match_request", "Matching the request", "Unable to match the request.")
async def match_request(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    ref = state.get("request_reference") or {}
    records = state.get("contact_requests") or []
    current = state.get("contact_request")
    identifying = any(ref.get(k) for k in ("name", "email", "request_id", "position"))

    if not identifying:
        if current:
            return _matched(state, current, f"Using {full_name(current)}'s request")
        return _waiting_for_request("No request selected yet", {"status": "need_reference"})

    candidates = records
    if ref.get("position"):
        listed = state.get("listed_request_ids") or []
        pos = int(ref["position"])
        index = pos - 1 if pos > 0 else len(listed) + pos
        wanted_id = listed[index] if 0 <= index < len(listed) else None
        candidates = [r for r in records if r.get("id") == wanted_id]
    else:
        if ref.get("request_id"):
            candidates = [r for r in candidates if str(r.get("id")) == str(ref["request_id"]).strip()]
        if ref.get("email"):
            candidates = [r for r in candidates if (r.get("emailId") or "").casefold() == ref["email"].strip().casefold()]
        if ref.get("name"):
            candidates = [r for r in candidates if _name_matches(ref["name"], r)]

    if len(candidates) == 1:
        return _matched(state, candidates[0], f"Matched {full_name(candidates[0])}")
    if len(candidates) > 1:
        ids = [r.get("id") for r in candidates[:_LIST_LIMIT]]
        return StepResult(
            f"{len(candidates)} requests match — asking which one",
            {
                "match_result": {"status": "ambiguous", "candidate_ids": ids},
                "listed_request_ids": ids,
                "pending_question": {"type": "choose_request", "candidates": len(ids)},
                "awaiting_user_input": True,
            },
        )

    already_searched = (state.get("match_result") or {}).get("status") == "search"
    if ref.get("name") and not already_searched:
        filters = {**(state.get("request_filters") or {}), "contact_name": ref["name"].strip()[:100]}
        return StepResult(
            f"Not in the loaded requests — searching for “{ref['name'].strip()}”",
            {"match_result": {"status": "search"}, "request_filters": filters},
        )
    return _waiting_for_request("No matching request found", {"status": "not_found", "reference": ref})


def _matched(state: ContactUsChatState, record: dict[str, Any], message: str) -> StepResult:
    update: dict[str, Any] = {
        "contact_request_id": record.get("id"),
        "match_result": {"status": "matched", "request_id": record.get("id")},
    }
    current = state.get("contact_request")
    if current and current.get("id") != record.get("id"):
        # Switching person: details and notes collected for the previous one don't apply.
        update.update(
            user_customer_fields={},
            extra_information={},
            invalid_fields={},
            missing_required_fields=[],
            known_fields=[],
            customer_payload=None,
            customer_result=None,
        )
        if state.get("intent") != "create_customer":
            # Don't carry a creation in progress over to someone the user only looked up.
            update["operation"] = None
    return StepResult(message, update, data={"name": full_name(record)})


def _waiting_for_request(message: str, match: dict[str, Any]) -> StepResult:
    return StepResult(
        message,
        {"match_result": match, "pending_question": {"type": "which_request"}, "awaiting_user_input": True},
    )


@tracked_step("merge_customer_fields", "Recording the information you provided", "Unable to record that information.")
async def merge_customer_fields(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    previous = state.get("user_customer_fields") or {}
    updates = state.get("turn_field_updates") or {}
    extra_new = state.get("turn_extra_information") or {}

    corrected = [FIELD_LABELS[k] for k, v in updates.items() if previous.get(k) and previous[k] != v]
    received = [FIELD_LABELS[k] for k in updates if FIELD_LABELS[k] not in corrected]
    parts = []
    if received:
        parts.append(f"Received: {', '.join(received)}")
    if corrected:
        parts.append(f"Corrected: {', '.join(corrected)}")
    if extra_new:
        parts.append(f"Noted: {', '.join(k.replace('_', ' ') for k in extra_new)}")
    return StepResult(
        " · ".join(parts) or "Nothing new to record",
        {
            # Never overwrite with empty values: only non-empty updates reach here.
            "user_customer_fields": {**previous, **updates},
            "extra_information": {**(state.get("extra_information") or {}), **extra_new},
        },
        data={"fields": received + corrected},
    )


@tracked_step("validate_customer_data", "Validating customer data", "Unable to validate the customer data.")
async def validate_customer_data(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    draft = customer_draft(state)
    if draft is None:
        raise StepFailed("No Contact Us request is selected.")
    missing = [FIELD_LABELS[k] for k in REQUIRED_FIELDS if not draft.get(k)]
    invalid: dict[str, str] = {}
    if draft.get("email") and not _EMAIL.match(draft["email"]):
        invalid["Email"] = "not a valid email address"
    known = [FIELD_LABELS[k] for k, v in draft.items() if v]

    if missing or invalid:
        message = "Missing required: " + ", ".join(missing) if missing else ""
        if invalid:
            message += ("; " if message else "") + "Invalid: " + ", ".join(invalid)
    else:
        message = "All required customer fields are present"
    return StepResult(
        message,
        {"customer_payload": draft, "missing_required_fields": missing, "invalid_fields": invalid, "known_fields": known},
        data={"missingFields": missing},
    )


async def ask_for_missing(state: ContactUsChatState, config: RunnableConfig) -> dict[str, Any]:
    fields = (state.get("missing_required_fields") or []) + list(state.get("invalid_fields") or {})
    await _emitter(config).step("ask_for_missing", "waiting", f"Waiting for: {', '.join(fields)}")
    return {
        "pending_question": {"type": "missing_fields", "fields": fields},
        "awaiting_user_input": True,
        "current_step": "ask_for_missing",
    }


async def respond(state: ContactUsChatState, config: RunnableConfig) -> dict[str, Any]:
    emitter = _emitter(config)
    final = finalize_turn(state)
    merged: ContactUsChatState = {**state, **final}  # type: ignore[typeddict-item]
    facts = build_facts(merged)

    try:
        reply = await compose_reply(config, state.get("conversation_history") or [], facts)
    except Exception as exc:  # the turn's real results still stand; keep details in logs
        logger.warning("[CONTACT_US_AGENT] run=%s reply generation failed: %s", state.get("run_id"), type(exc).__name__)
        reply = ""
    if not reply:
        reply = _fallback_reply(merged)

    if merged.get("awaiting_user_input") and "ask_for_missing" not in (state.get("turn_steps") or []):
        await emitter.step("awaiting_user", "waiting", "Waiting for your reply")

    return {
        **final,
        "agent_response": reply,
        "conversation_history": [{"role": "assistant", "content": reply}],
        "completed_steps": [s for s, status in emitter.outcomes if status == "completed"],
        "failed_steps": [s for s, status in emitter.outcomes if status in ("failed", "blocked")],
    }


# --- turn results --------------------------------------------------------------


def listed_this_turn(state: ContactUsChatState) -> bool:
    """Requests were retrieved for a listing this turn (and the retrieval succeeded)."""
    retrieval_failed = state.get("current_step") == "get_contact_us_data" and state.get("execution_status") == "failed"
    return (
        "get_contact_us_data" in (state.get("turn_steps") or [])
        and not retrieval_failed
        and bool(state.get("kickoff") or state.get("intent") == "list_requests")
    )


def finalize_turn(state: ContactUsChatState) -> dict[str, Any]:
    """Deterministic end-of-turn bookkeeping: what was shown, what was created, status."""
    update: dict[str, Any] = {}
    if listed_this_turn(state):
        update["listed_request_ids"] = [r.get("id") for r in listed_records(state)[:_LIST_LIMIT]]

    result = state.get("customer_result") or {}
    request_id = (state.get("contact_request") or {}).get("id")
    verified = "verify_customer" in (state.get("turn_steps") or []) and state.get("execution_status") == "running"
    if verified and result.get("id") and request_id:
        update["created_customers"] = {**(state.get("created_customers") or {}), request_id: str(result["id"])}
        update["operation"] = None
        update["is_complete"] = True

    status = state.get("execution_status")
    if status == "failed":
        update["workflow_status"] = "failed"
    elif status == "blocked":
        update["workflow_status"] = "blocked"
    elif state.get("awaiting_user_input"):
        update["workflow_status"] = "waiting_for_user"
    else:
        update["workflow_status"] = "completed"
    return update


def creation_status(state: ContactUsChatState) -> dict[str, Any]:
    request_id = (state.get("contact_request") or {}).get("id")
    created = (state.get("created_customers") or {}).get(request_id or "")
    if created:
        return {"status": "created", "customer_id": created, "this_turn": bool(state.get("is_complete"))}
    step, status = state.get("current_step"), state.get("execution_status")
    if step == "create_customer" and status == "blocked":
        not_configured = state.get("error") == CUSTOMER_API_NOT_CONFIGURED
        return {"status": "api_not_configured" if not_configured else "blocked", "message": state.get("error")}
    if step in ("create_customer", "verify_customer") and status == "failed":
        return {"status": "failed", "message": state.get("error")}
    return {"status": "not_attempted"}


def customer_view(state: ContactUsChatState) -> dict[str, Any] | None:
    """Customer data for the UI: every field, where its value came from, what's missing."""
    draft = customer_draft(state)
    if draft is None:
        return None
    user_fields = state.get("user_customer_fields") or {}
    missing = [FIELD_LABELS[k] for k in REQUIRED_FIELDS if not draft.get(k)]
    return {
        "fields": [
            {
                "key": key,
                "label": label,
                "value": draft.get(key),
                "required": required,
                "source": "user" if key in user_fields else ("request" if draft.get(key) else None),
            }
            for key, _, label, required in CUSTOMER_FIELDS
        ],
        "missingRequiredFields": missing,
        "invalidFields": state.get("invalid_fields") or {},
        "creation": creation_status(state),
    }


def build_facts(state: ContactUsChatState) -> dict[str, Any]:
    """Everything the reply LLM may talk about, from real results only."""
    intent = state.get("intent")
    facts: dict[str, Any] = {
        "agent": _agent(state),
        "kickoff": bool(state.get("kickoff")),
        "intent": intent,
        "user_request": state.get("request_summary"),
        "company": state.get("provider_name") or "All companies available to the user",
    }
    if state.get("kickoff") or intent in ("greeting_or_help", "other", None):
        facts["capabilities"] = CAPABILITIES
    if state.get("execution_status") in ("failed", "blocked") and state.get("error"):
        facts["problem"] = {"step": ACTIVITY_TITLES.get(state.get("current_step") or "", "Agent"), "message": state["error"]}
    if state.get("last_tool_result"):
        facts["notes"] = state["last_tool_result"]

    if listed_this_turn(state):
        records = listed_records(state)
        facts["requests"] = {
            "kind": _agent(state)["handles"],
            "total_open_requests": state.get("contact_requests_total"),
            "returned": len(records),
            "filters": state.get("request_filters") or {},
            "shown": [request_summary(r) for r in records[:_FACTS_LIST_LIMIT]],
        }

    match = state.get("match_result")
    if match and match.get("status") != "search":
        by_id = {r.get("id"): r for r in state.get("contact_requests") or []}
        facts["request_match"] = {
            "status": match["status"],
            "candidates": [request_summary(by_id[i]) for i in match.get("candidate_ids", []) if i in by_id][:_FACTS_LIST_LIMIT],
        }

    selected = state.get("contact_request")
    if selected:
        facts["selected_request"] = request_summary(selected)
        if (
            state.get("operation") == "create_customer"
            or state.get("is_complete")
            or intent in ("show_request_details", "create_customer", "provide_information")
        ):
            view = customer_view(state) or {}
            facts["customer"] = {
                "creating_customer": state.get("operation") == "create_customer",
                "fields": {f["label"]: f["value"] for f in view.get("fields", [])},
                "missing_required": view.get("missingRequiredFields", []),
                "invalid": view.get("invalidFields", {}),
                "provided_this_turn": [FIELD_LABELS[k] for k in state.get("turn_field_updates") or {}],
                "creation": view.get("creation"),
            }
    elif state.get("turn_field_updates"):
        facts["customer_details_received_without_request"] = [FIELD_LABELS[k] for k in state["turn_field_updates"]]

    if state.get("extra_information"):
        facts["extra_information"] = {
            "all": state["extra_information"],
            "new_this_turn": state.get("turn_extra_information") or {},
        }
    if state.get("pending_question"):
        facts["pending_question"] = state["pending_question"]
    return facts


def _fallback_reply(state: ContactUsChatState) -> str:
    """Used only when the reply LLM is unavailable; states real results plainly."""
    if state.get("execution_status") in ("failed", "blocked") and state.get("error"):
        return f"{state['error']} Please try again."
    missing = state.get("missing_required_fields") or []
    if state.get("pending_question", {}) and missing:
        return "I still need: " + ", ".join(missing) + "."
    return "The step finished, but I couldn't write a full reply because the AI service is unavailable. Please try again."
