"""Chat-agent nodes for the Contact Us Agent.

The existing workflow nodes (nodes.py) do the Bootog work unchanged. The nodes here add
the conversational layer around them (conversion steps live in conversion_nodes.py):

  understand_message      LLM -> typed TurnUnderstanding (intent, references, fields)
  resolve_provider        company named in chat -> provider id (existing provider tool)
  match_request           deterministic match of the user's reference to a request
  merge_customer_fields   user-supplied values over the request data (latest wins)
  validate_customer_data  required/format checks on the conversion form
  ask_for_missing         records the pending question; the turn waits for the user
  respond                 LLM reply written only from structured facts

Every tracked node publishes Live Tracking events through `tracked_step`."""

import re
from datetime import date
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.contact_us.conversion_nodes import (
    active_type,
    conversion_draft,
    conversion_of,
    converted_key,
    draft_type,
)
from app.agents.contact_us.conversions import (
    CONVERSION_OPERATIONS,
    ALL_TYPES,
    FIELD_LABELS,
    PARTNER_TYPES,
    is_placeholder,
    STEP_LABELS,
    conversion_type,
    validate_draft,
)
from app.agents.contact_us.nodes import StepFailed, StepResult, tracked_step
from app.agents.contact_us.schemas import build_customer_draft, summarize_contact_request
from app.agents.contact_us.state import ContactUsChatState
from app.agents.contact_us.tools import get_provider_list
from app.agents.contact_us.understanding import INTENT_LABELS, compose_reply, understand
from app.agents.runtime import RunChannel, RunEmitter
from app.core.logging import get_logger
from app.llm import LlmNotConfiguredError

logger = get_logger(__name__)

_LIST_LIMIT = 25  # requests sent to the UI per listing
_FACTS_LIST_LIMIT = 10  # requests described to the LLM per listing

# ContactUsStatus in the host's libs/models/enums/contact-us-status.model.ts.
CONTACT_US_STATUS_LABELS: dict[int, str] = {
    1: "Proposal Submitted", 2: "Pending Approval", 3: "Needs Clarification", 4: "Under Review",
    5: "Approved", 6: "Rejected", 7: "In Implementation", 8: "Completed", 9: "On Hold",
    10: "Cancelled", 11: "CRM Completed", 12: "Closed", 13: "Re Open", 14: "Re Scheduled",
    15: "Assigned", 16: "Left Message", 17: "Email Sent", 18: "Document Requested",
}
_STATUS_BY_NAME = {re.sub(r"[^a-z]", "", label.casefold()): value for value, label in CONTACT_US_STATUS_LABELS.items()}

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
    "validate_customer_data": "Data Validation",
    "ask_for_missing": "Waiting for User",
    "awaiting_user": "Waiting for User",
    "create_customer": "Customer Creation",
    "verify_customer": "Customer Verification",
    "start_conversion": "Conversion Setup",
    "choose_conversion_type": "Waiting for User",
    "resolve_target_provider": "Service Provider Selection",
    "search_providers": "Provider Search",
    "resolve_assigned_user": "Assignee Lookup",
    "check_existing_account": "Existing Account Check",
    "resolve_role": "Role Lookup",
    "select_subcategories": "Service Categories",
    "confirm_conversion": "Waiting for Confirmation",
    "create_account": "Account Creation",
    "verify_account": "Account Verification",
    "link_provider_vendor": "Provider Link",
    "save_subcategories": "Service Categories Saved",
    "update_request_status": "Request Status Update",
    "refresh_requests": "Request List Refresh",
}

CAPABILITIES = [
    "List Contact Us requests (recent, by date, status, assignee or company, or only incomplete ones)",
    "Find a request by person name, email or position in a list — searching all requests, not just the latest page",
    "Show the details of a request",
    "Convert a request into a Customer, a B2B Client (insurance carrier) or a Partner "
    "(insurance agency, real estate agency or contractor), asking for any missing details and for confirmation",
    "Find BOOTOG service providers by name or location",
    "Retry a failed step of a conversion without creating a duplicate account",
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
    return [t for t in re.findall(r"[^\W_]+", (text or "").casefold()) if len(t) > 1]


def _name_matches(reference: str, record: dict[str, Any]) -> bool:
    """Every word of the reference matches a word of the name: exactly, as a prefix (3+
    letters), or — for a single letter — as an initial ("Gokulan A" matches "Gokulan Alice")."""
    wanted = re.findall(r"[^\W_]+", (reference or "").casefold())
    have = re.findall(r"[^\W_]+", f"{record.get('firstName') or ''} {record.get('lastName') or ''}".casefold())
    return bool(wanted) and all(
        any(h == w or ((len(w) >= 3 or len(w) == 1) and h.startswith(w)) for h in have) for w in wanted
    )


def _exact_name(reference: str, record: dict[str, Any]) -> bool:
    name = f"{record.get('firstName') or ''} {record.get('lastName') or ''}"
    return " ".join(re.findall(r"[^\W_]+", name.casefold())) == " ".join(re.findall(r"[^\W_]+", reference.casefold()))


def _iso_date(value: str | None) -> str | None:
    """Only well-formed dates reach the Bootog filter expression."""
    try:
        return date.fromisoformat((value or "").strip()).isoformat()
    except ValueError:
        return None


def _snake(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().casefold()).strip("_")[:60]


def status_values(names: list[str]) -> list[int]:
    """Status names -> ContactUsStatus values; unknown names are dropped (never guessed)."""
    out = []
    for name in names:
        value = _STATUS_BY_NAME.get(re.sub(r"[^a-z]", "", name.casefold()))
        if value and value not in out:
            out.append(value)
    return out


def request_summary(record: dict[str, Any]) -> dict[str, Any]:
    _, missing = build_customer_draft(record)
    status = record.get("status")
    return {
        **summarize_contact_request(record),
        "statusLabel": CONTACT_US_STATUS_LABELS.get(status) if isinstance(status, int) else None,
        "missingFields": missing,
    }


def customer_draft(state: ContactUsChatState) -> dict[str, Any] | None:
    """Request data overlaid with what the user supplied, in the active conversion's form."""
    return conversion_draft(state)


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
    conv = conversion_of(state)
    ctype = active_type(state)
    return {
        "agent": _agent(state),
        "company": state.get("provider_name") or "All companies available to the user",
        "ongoing_goal": state.get("operation") or "none",
        "conversion": (
            {
                "type": ctype.label if ctype else "partner (type not chosen yet)",
                "service_provider": (conv.get("target_provider") or {}).get("name"),
                "service_categories": [s["name"] for s in conv.get("subcategories") or []],
                "steps_done": conv.get("completed") or [],
                "failed_step": (conv.get("failed") or {}).get("step"),
            }
            if conv.get("request_id")
            else None
        ),
        "selected_request": (
            {"id": selected.get("id"), "name": full_name(selected), "email": selected.get("emailId")} if selected else None
        ),
        "pending_question": state.get("pending_question"),
        "options_offered": _options_offered(state),
        "last_listed_requests": [
            {"position": i + 1, "id": r.get("id"), "name": full_name(r), "email": r.get("emailId"), "created": r.get("createdDate")}
            for i, r in enumerate(listed[:_LIST_LIMIT])
        ],
        "form_fields": {FIELD_LABELS[k]: v for k, v in draft.items()},
        "extra_information": state.get("extra_information") or {},
    }


def _options_offered(state: ContactUsChatState) -> list[str] | None:
    question = (state.get("pending_question") or {}).get("type")
    conv = conversion_of(state)
    if question == "choose_partner_type":
        return [t.label for t in PARTNER_TYPES]
    if question == "choose_conversion_type":
        return [t.label for t in ALL_TYPES]
    if question == "choose_provider":
        return [p["name"] for p in conv.get("provider_candidates") or []]
    if question == "choose_subcategory":
        return [s["name"] for s in conv.get("subcategory_candidates") or []]
    return None


def _agent(state: ContactUsChatState) -> dict[str, str]:
    name = state.get("agent_name") or "Contact Us Agent"
    return {"name": name, "handles": f"{name.removesuffix(' Agent')} requests"}


# --- nodes -------------------------------------------------------------------


async def start_turn(state: ContactUsChatState, config: RunnableConfig) -> dict[str, Any]:
    """Resets per-turn fields; cross-turn memory (selection, fields, notes, conversion) is kept."""
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
        "turn_conversion": {},
        "answered_question": None,
        "provider_query": None,
        "provider_results": None,
        "refreshed": False,
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
        "answered_question": state.get("pending_question"),
        "pending_question": None,
        "awaiting_user_input": False,
    }

    conv = result.conversion
    if conv is not None:
        update["turn_conversion"] = {
            "partner_type": conv.partner_type,
            "target_provider_name": (conv.target_provider_name or "").strip()[:200] or None,
            "option_position": conv.option_position if conv.option_position and conv.option_position > 0 else None,
            "subcategory_names": [n.strip()[:100] for n in conv.subcategory_names if n.strip()][:10],
            "option_positions": [p for p in conv.option_positions if isinstance(p, int) and p > 0][:15],
        }

    operation = _operation_for(result.intent, result.goal, state.get("operation"))
    if operation:
        update["operation"] = operation
    elif result.intent == "cancel_operation" or result.goal == "none":
        update["operation"] = None
        current = conversion_of(state)
        if "create_account" not in (current.get("completed") or []) and not current.get("create_attempted"):
            update["conversion"] = None  # nothing was created: forget the draft conversion
    elif result.intent == "provide_information" and conv is not None and conv.partner_type and not state.get("operation"):
        update["operation"] = "create_partner"

    if result.customer_field_updates is not None:
        # Values identical to what the form already holds are not "provided by the user":
        # the model tends to echo the request's own data back.
        current = customer_draft(state) or {}
        update["turn_field_updates"] = {
            key: value.strip()[:200]
            for key, value in result.customer_field_updates.model_dump().items()
            if key in FIELD_LABELS
            and isinstance(value, str)
            and value.strip()
            and not is_placeholder(value)  # "Needs verification" is not a value: still missing
            and " ".join(value.split()).casefold() != " ".join(str(current.get(key) or "").split()).casefold()
        }
    update["turn_extra_information"] = {
        _snake(item.key): item.value.strip()[:300]
        for item in result.extra_information
        if _snake(item.key) and item.value.strip()
    }

    filters: dict[str, Any] = {}
    ignored: list[str] = []
    reference = _reference(result.request_reference, ignored)
    answered = (state.get("pending_question") or {}).get("type")
    if answered in ("choose_request", "which_request") and conv is not None and conv.option_position:
        # A bare number answering "which request?" picks from the request list shown,
        # however the model labelled it.
        if not reference or not any(reference.get(k) for k in REFERENCE_KEYS):
            reference = {"use_current": False, "position": int(conv.option_position)}
    if reference and reference.get("position") and not _mentions_position(state.get("current_user_message"), reference["position"]):
        # The model sometimes invents "position 1" for "her" / "this request". Picking a
        # list entry the user never named would silently switch requests (and drop the
        # details collected so far), so a position counts only when the message has it.
        reference.pop("position")
        if not reference.get("use_current") and not any(reference.get(k) for k in REFERENCE_KEYS):
            reference = {"use_current": True} if state.get("contact_request") else None
    if reference and reference.get("position"):
        # A number refers to the list the user is looking at. Resolve it to that entry's
        # id now; anything else the model attached (often copied from the list) is ignored.
        listed_ids = state.get("listed_request_ids") or []
        pos = reference["position"]
        index = pos - 1 if pos > 0 else len(listed_ids) + pos
        reference = {"use_current": False, "position": pos}
        if 0 <= index < len(listed_ids):
            reference["request_id"] = listed_ids[index]
    if reference and result.intent != "find_request" and _names_current_request(state, reference):
        # The user is still talking about the selected request: never reopen the choice.
        reference = {"use_current": True}
    if reference:
        update["request_reference"] = reference
        loaded = {r.get("id") for r in state.get("contact_requests") or []}
        if reference.get("request_id") and reference["request_id"] not in loaded:
            filters["request_id"] = reference["request_id"]  # exact lookup, any status
        elif not reference.get("position") and not reference.get("request_id"):
            # A name/email is searched across ALL requests, not just the page already loaded.
            if reference.get("name"):
                filters["contact_name"] = reference["name"][:100]
            if reference.get("email"):
                filters["email"] = reference["email"][:200]

    if result.filters:
        f = result.filters
        statuses = status_values(f.statuses)
        ignored += [f"status '{s}'" for s in f.statuses if not status_values([s])]
        for label, value in (("start date", f.created_from), ("end date", f.created_to)):
            if value and not _iso_date(value):
                ignored.append(f"{label} '{value}'")
        filters.update(
            created_from=_iso_date(f.created_from),
            created_to=_iso_date(f.created_to),
            incomplete_only=f.incomplete_only,
            statuses=statuses,
            assigned_user_name=(f.assigned_user_name or "").strip()[:100] or None,
        )
        # Asking for someone's requests or a period means all of them, whatever their
        # status; only a plain listing (or an explicit "open") keeps the open-status filter.
        filters["all_statuses"] = not statuses and not f.only_open and bool(
            filters["assigned_user_name"] or filters["created_from"] or filters["created_to"]
        )
        if f.company_name and f.company_name.strip():
            update["company_name"] = f.company_name.strip()[:200]
    if ignored:
        filters["ignored"] = ignored
    update["request_filters"] = {k: v for k, v in filters.items() if v} or None

    if result.intent == "list_providers" and result.provider_query:
        q = result.provider_query
        update["provider_query"] = {
            "name": (q.name or "").strip()[:100] or None,
            "location": (q.location or "").strip()[:100] or None,
        }

    message = update["request_summary"] or INTENT_LABELS[result.intent]
    return StepResult(message, update, data={"intent": result.intent})


REFERENCE_KEYS = ("name", "email", "request_id", "position", "created_date", "status", "assigned_user_name")
_DISCRIMINATORS = ("created_date", "status", "assigned_user_name")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def _reference(ref: Any, ignored: list[str]) -> dict[str, Any] | None:
    """The LLM's request reference, cleaned: only well-formed values survive."""
    if ref is None:
        return None
    out: dict[str, Any] = {"use_current": bool(ref.use_current)}
    for key in ("name", "email", "status", "assigned_user_name"):
        value = (getattr(ref, key) or "").strip()
        if value:
            out[key] = value[:200]
    request_id = (ref.request_id or "").strip()
    if request_id:
        if _REQUEST_ID.match(request_id):
            out["request_id"] = request_id
        else:
            ignored.append(f"request id '{request_id[:40]}'")
    if ref.position:
        out["position"] = int(ref.position)
    if ref.created_date:
        if _iso_date(ref.created_date):
            out["created_date"] = _iso_date(ref.created_date)
        else:
            ignored.append(f"date '{ref.created_date}'")
    if out.get("status") and not status_values([out["status"]]):
        ignored.append(f"status '{out.pop('status')}'")
    return out if out["use_current"] or any(out.get(k) for k in REFERENCE_KEYS) else None


_ORDINALS: dict[int, tuple[str, ...]] = {
    1: ("first", "1st"), 2: ("second", "2nd", "two"), 3: ("third", "3rd", "three"), 4: ("fourth", "4th", "four"),
    5: ("fifth", "5th", "five"), 6: ("sixth", "6th", "six"), 7: ("seventh", "7th", "seven"),
    8: ("eighth", "8th", "eight"), 9: ("ninth", "9th", "nine"), 10: ("tenth", "10th", "ten"),
}


def _mentions_position(message: str | None, position: int) -> bool:
    """The user's message really names this list position (a number or an ordinal)."""
    text = (message or "").casefold()
    if position < 0:
        return bool(re.search(r"\blast\b", text))
    if re.search(rf"(?<![\d.,-]){position}(?![\d.,]\d)(?!\d)", text):
        return True
    return any(re.search(rf"\b{word}\b", text) for word in _ORDINALS.get(position, ()))


def _names_current_request(state: ContactUsChatState, ref: dict[str, Any]) -> bool:
    """The reference only names the person/email of the request already selected."""
    current = state.get("contact_request")
    if not current or any(ref.get(k) for k in ("request_id", "position", *_DISCRIMINATORS)):
        return False
    if not (ref.get("name") or ref.get("email")):
        return False
    if ref.get("name") and not _name_matches(ref["name"], current):
        return False
    return not ref.get("email") or (current.get("emailId") or "").casefold() == ref["email"].casefold()


def _apply_discriminators(records: list[dict[str, Any]], ref: dict[str, Any]) -> list[dict[str, Any]]:
    out = records
    if ref.get("created_date"):
        out = [r for r in out if str(r.get("createdDate") or "")[:10] == ref["created_date"]]
    if ref.get("status"):
        wanted = set(status_values([ref["status"]]))
        out = [r for r in out if r.get("status") in wanted]
    if ref.get("assigned_user_name"):
        tokens = _tokens(ref["assigned_user_name"])
        out = [r for r in out if tokens and all(t in _tokens(r.get("assignedUserName")) for t in tokens)]
    return out


def _operation_for(intent: str, goal: str, current: str | None) -> str | None:
    operation = intent if intent in CONVERSION_OPERATIONS else goal if goal in CONVERSION_OPERATIONS else None
    if operation == "convert_request" and current in CONVERSION_OPERATIONS:
        return current  # "proceed" continues the conversion already chosen; it never guesses one
    return operation


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
    identifying = any(ref.get(k) for k in REFERENCE_KEYS)

    if not identifying:
        if current:
            return _matched(state, current, f"Using {full_name(current)}'s request")
        return _waiting_for_request("No request selected yet", {"status": "need_reference"})

    by_id = {r.get("id"): r for r in records}
    listed = [by_id[i] for i in state.get("listed_request_ids") or [] if i in by_id]
    if ref.get("position") and not ref.get("request_id"):
        # Numbers always refer to the list shown last, in the order it was shown (resolved
        # to an id when understood; this path only runs for an out-of-range number).
        pos = int(ref["position"])
        index = pos - 1 if pos > 0 else len(listed) + pos
        candidates = [listed[index]] if 0 <= index < len(listed) else []
    else:
        # Only a date/status/assignee: narrow the list the user is looking at.
        named = any(ref.get(k) for k in ("name", "email", "request_id"))
        candidates = records if named or not listed else listed
        if ref.get("request_id"):
            candidates = [r for r in candidates if str(r.get("id")).casefold() == ref["request_id"].casefold()]
        if ref.get("email"):
            candidates = [r for r in candidates if (r.get("emailId") or "").casefold() == ref["email"].strip().casefold()]
        if ref.get("name"):
            candidates = [r for r in candidates if _name_matches(ref["name"], r)]
            # "Gokulan A" means the people named exactly that when there are any; an
            # initial only widens the match when nobody has the exact name.
            exact = [r for r in candidates if _exact_name(ref["name"], r)]
            candidates = exact or candidates
        candidates = _apply_discriminators(candidates, ref)

    if len(candidates) == 1:
        return _matched(state, candidates[0], f"Matched {full_name(candidates[0])}")
    if len(candidates) > 1:
        # Several requests (often the same person) — never pick one arbitrarily. The ids
        # are stored in the order shown, so "number 2" always means the second card.
        ids = [r.get("id") for r in candidates[:_LIST_LIMIT]]
        emails = {(r.get("emailId") or "").casefold() for r in candidates}
        return StepResult(
            f"{len(candidates)} requests match — asking which one",
            {
                "match_result": {
                    "status": "ambiguous",
                    "candidate_ids": ids,
                    "total": len(candidates),
                    "shared_email": len(emails) == 1 and "" not in emails,
                },
                "listed_request_ids": ids,
                "pending_question": {"type": "choose_request", "candidates": len(ids), "total": len(candidates)},
                "awaiting_user_input": True,
            },
        )

    filters = state.get("request_filters") or {}
    searched = (state.get("match_result") or {}).get("status") == "search" or bool(
        filters.get("contact_name") or filters.get("email")
    )
    if (ref.get("name") or ref.get("email")) and not searched:
        search = {k: v for k, v in (("contact_name", ref.get("name")), ("email", ref.get("email"))) if v}
        return StepResult(
            "Not in the loaded requests — searching all requests",
            {"match_result": {"status": "search"}, "request_filters": {**filters, **search}},
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
        if state.get("intent") not in CONVERSION_OPERATIONS:
            # Don't carry a conversion in progress over to someone the user only looked up.
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


@tracked_step("validate_customer_data", "Validating the details", "Unable to validate the details.")
async def validate_customer_data(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    draft = customer_draft(state)
    if draft is None:
        raise StepFailed("No Contact Us request is selected.")
    ctype = draft_type(state)
    normalized, missing, invalid = validate_draft(ctype, draft)
    known = [FIELD_LABELS[k] for k, v in normalized.items() if v]

    if missing or invalid:
        message = "Missing required: " + ", ".join(missing) if missing else ""
        if invalid:
            message += ("; " if message else "") + "Invalid: " + ", ".join(invalid)
    else:
        message = f"All required {ctype.label} details are present"
    return StepResult(
        message,
        {"customer_payload": normalized, "missing_required_fields": missing, "invalid_fields": invalid, "known_fields": known},
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

    waiting_steps = {"ask_for_missing", "confirm_conversion", "choose_conversion_type"}
    if merged.get("awaiting_user_input") and not waiting_steps & set(state.get("turn_steps") or []):
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
    return bool(state.get("refreshed")) or (
        "get_contact_us_data" in (state.get("turn_steps") or [])
        and not retrieval_failed
        and bool(state.get("kickoff") or state.get("intent") == "list_requests")
    )


def conversion_finished(state: ContactUsChatState) -> bool:
    ctype = active_type(state)
    done = set(conversion_of(state).get("completed") or [])
    return bool(ctype) and set(ctype.steps()) <= done


def finalize_turn(state: ContactUsChatState) -> dict[str, Any]:
    """Deterministic end-of-turn bookkeeping: what was shown, what was created, status."""
    update: dict[str, Any] = {}
    if listed_this_turn(state):
        update["listed_request_ids"] = [r.get("id") for r in listed_records(state)[:_LIST_LIMIT]]

    conv = conversion_of(state)
    request_id = conv.get("request_id")
    if "update_request_status" in (state.get("turn_steps") or []) and conversion_finished(state) and request_id:
        created_id = conv.get("user_id") or conv.get("created_vendor_id") or "created"
        key = converted_key(request_id, conv.get("type"))
        update["created_customers"] = {**(state.get("created_customers") or {}), key: str(created_id)}
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
    """Shape kept from the customer card: not_attempted / created / blocked / failed."""
    conv = conversion_of(state)
    request_id = (state.get("contact_request") or {}).get("id")
    if conv.get("request_id") != request_id:
        conv = {}
    done = conv.get("completed") or []
    ctype = active_type(state)
    pending = [s for s in ctype.steps() if s not in done] if ctype else []
    if "create_account" in done:
        return {
            "status": "created",
            "customer_id": conv.get("user_id") or conv.get("created_vendor_id"),
            "this_turn": bool(state.get("is_complete")) or "create_account" in (state.get("turn_steps") or []),
            "pending_steps": [STEP_LABELS[s] for s in pending],
            "message": (conv.get("failed") or {}).get("message"),
        }
    created = (state.get("created_customers") or {}).get(converted_key(request_id, ctype.key if ctype else None))
    if created:
        return {"status": "created", "customer_id": created, "this_turn": bool(state.get("is_complete")), "pending_steps": []}
    if conv.get("blocked") in ("existing_role", "already_converted", "type_locked"):
        return {"status": "blocked", "message": conv["blocked"]}
    if conv.get("failed"):
        return {"status": "failed", "message": conv["failed"].get("message")}
    return {"status": "not_attempted"}


def customer_view(state: ContactUsChatState) -> dict[str, Any] | None:
    """The conversion form for the UI: every field, where its value came from, what's missing."""
    draft = customer_draft(state)
    if draft is None:
        return None
    ctype = draft_type(state)
    draft, missing, _ = validate_draft(ctype, draft)  # show what would be sent (FL, US, phone mask)
    user_fields = state.get("user_customer_fields") or {}
    conv = conversion_of(state)
    return {
        "conversionType": ctype.key,
        "conversionLabel": ctype.label,
        "serviceProvider": (conv.get("target_provider") or {}).get("name"),
        "fields": [
            {
                "key": key,
                "label": FIELD_LABELS[key],
                "value": draft.get(key),
                "required": key in ctype.required,
                "source": "user" if key in user_fields else ("request" if draft.get(key) else None),
            }
            for key in ctype.fields
        ],
        "missingRequiredFields": missing,
        "invalidFields": state.get("invalid_fields") or {},
        "creation": creation_status(state),
    }


def _conversion_facts(state: ContactUsChatState) -> dict[str, Any] | None:
    conv = conversion_of(state)
    if not conv.get("request_id") or conv.get("request_id") != (state.get("contact_request") or {}).get("id"):
        return None
    ctype = active_type(state)
    done = conv.get("completed") or []
    facts: dict[str, Any] = {
        "type": ctype.label if ctype else None,
        "role": conv.get("role_name"),
        "service_provider": (conv.get("target_provider") or {}).get("name"),
        "service_categories": [s["name"] for s in conv.get("subcategories") or []],
        "steps": {
            STEP_LABELS[s]: ("done" if s in done else "failed" if (conv.get("failed") or {}).get("step") == s else "not yet")
            for s in (ctype.steps() if ctype else [])
        },
    }
    if conv.get("account"):
        account = conv["account"]
        facts["existing_account"] = {
            "email": account.get("email"),
            "registered": account.get("exists"),
            "already_has_required_role": account.get("has_role"),
            "roles": account.get("roles"),
        }
    if conv.get("blocked") == "existing_role":
        facts["blocked"] = (
            f"The email is already registered with the {ctype.role_name} role. Nothing was created. "
            "Alternatives: convert to a different type, use another email, or cancel."
        )
    elif conv.get("blocked") == "already_converted":
        label = conversion_type(conv.get("blocked_type")).label if conv.get("blocked_type") else "this type"
        facts["blocked"] = f"This request was already converted to {label}; nothing new was created."
    elif conv.get("blocked") == "type_locked":
        facts["blocked"] = f"An account was already created as {ctype.label}; the type can no longer be changed."
    if conv.get("failed"):
        failed = conv["failed"]
        facts["failure"] = {
            "step": STEP_LABELS.get(failed["step"], failed["step"]),
            "message": failed.get("message"),
            "outcome_unknown": bool(failed.get("outcome_unknown")),
            "next": _failure_advice(failed),
        }
    question = (state.get("pending_question") or {}).get("type")
    if question == "confirm_conversion":
        draft = state.get("customer_payload") or {}
        record = state.get("contact_request") or {}
        facts["awaiting_confirmation"] = {
            "will_create": ctype.label,
            "from_request": {"id": record.get("id"), "name": full_name(record), "email": record.get("emailId")},
            "service_provider": (conv.get("target_provider") or {}).get("name"),
            "service_categories": [s["name"] for s in conv.get("subcategories") or []],
            "carried_over_from_previous_conversion": conv.get("carried_over") or {},
            "role": conv.get("role_name"),
            "details": {FIELD_LABELS[k]: v for k, v in draft.items() if v},
            "coordinates": "from the request" if draft.get("latitude") else "not available — left empty",
            "then": [STEP_LABELS[s] for s in ctype.steps()[1:]],
        }
    if question == "choose_partner_type":
        facts["partner_types"] = [t.label.removeprefix("Partner — ") for t in PARTNER_TYPES]
    if question == "choose_conversion_type":
        facts["conversion_types"] = [{"number": i + 1, "type": t.label} for i, t in enumerate(ALL_TYPES)]
    if question == "choose_provider":
        facts["provider_choice"] = {
            **(state.get("pending_question") or {}),
            "options": [
                {"position": i + 1, "name": p["name"], "address": p.get("address")}
                for i, p in enumerate(conv.get("provider_candidates") or [])
            ],
        }
    if question == "choose_subcategory":
        facts["service_category_choice"] = {
            **(state.get("pending_question") or {}),
            "options": [{"position": i + 1, "name": s["name"]} for i, s in enumerate(conv.get("subcategory_candidates") or [])],
        }
    if conversion_finished(state):
        facts["result"] = {
            "completed": True,
            "request_status": "CRM Completed",
            "user_id": conv.get("user_id"),
            "company_id": conv.get("created_vendor_id") if ctype and ctype.skip_tenant else None,
        }
    return facts


def _failure_advice(failed: dict[str, Any]) -> str:
    if failed.get("code") == "CSRF_INVALID":
        return (
            "Bootog's gateway refused the write (anti-forgery check). Nothing was created. An administrator "
            "must fix the agent's Bootog connection; then 'retry' sends it again."
        )
    if failed.get("step") == "create_account":
        if failed.get("kind") == "rejected":
            return "Bootog rejected these details. Change the detail it complains about (or cancel); the same details are not resent."
        if failed.get("kind") == "access":
            return "Bootog refused access. Nothing was created. 'retry' sends it again once access is fixed."
        return "It is unknown whether the account was created. 'retry' first checks BOOTOG, so no duplicate is created."
    return "The account exists. 'retry' resumes from this step without creating another account."


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
        assignee = state["last_tool_result"].get("assignee") if isinstance(state["last_tool_result"], dict) else None
        if assignee:
            facts["assignee_lookup"] = {
                "requested": assignee.get("requested"),
                "found": False,
                "meaning": (
                    "No Contact Us request is assigned to anyone by that name, so no request search was "
                    "run. This is NOT 'no requests in that period'."
                    if assignee.get("status") == "not_found"
                    else "Several assignees match that name; ask which one."
                ),
                "assignees_with_requests": assignee.get("candidates"),
            }

    filters = state.get("request_filters") or {}
    if filters.get("ignored"):
        facts["ignored_filters"] = filters["ignored"]  # could not be applied; say so
    if listed_this_turn(state):
        records = listed_records(state)
        if filters.get("statuses"):
            scope = "statuses: " + ", ".join(CONTACT_US_STATUS_LABELS.get(s, str(s)) for s in filters["statuses"])
        elif filters.get("all_statuses"):
            scope = "all statuses, including completed and closed requests"
        else:
            scope = "open requests only (Completed, CRM Completed and Closed are excluded)"
        facts["requests"] = {
            "kind": _agent(state)["handles"],
            "status_scope": scope,
            "total_matching": state.get("contact_requests_total"),
            "returned": len(records),
            "filters": {k: v for k, v in filters.items() if k not in ("assigned_user_ids", "ignored", "all_statuses", "statuses")},
            "refreshed_after_conversion": bool(state.get("refreshed")),
            "shown": [{"number": i + 1, **request_summary(r)} for i, r in enumerate(records[:_FACTS_LIST_LIMIT])],
        }
    if state.get("provider_results"):
        facts["providers"] = state["provider_results"]

    match = state.get("match_result")
    if match and match.get("status") != "search":
        by_id = {r.get("id"): r for r in state.get("contact_requests") or []}
        candidates = [by_id[i] for i in match.get("candidate_ids", []) if i in by_id]
        facts["request_match"] = {
            "status": match["status"],
            "searched_all_requests": bool(filters.get("contact_name") or filters.get("email")),
            "search_truncated": bool(state.get("contact_requests_truncated")),
            "total_matches": match.get("total", len(candidates)),
            "shared_email": (
                (by_id.get((match.get("candidate_ids") or [None])[0]) or {}).get("emailId") if match.get("shared_email") else None
            ),
            "how_to_choose": "Reply with the number, the request id, or its date / status / assignee.",
            # Numbered exactly as stored: "number 2" selects the second entry.
            "candidates": [{"number": i + 1, **request_summary(r)} for i, r in enumerate(candidates[:_LIST_LIMIT])],
        }

    selected = state.get("contact_request")
    converting = state.get("operation") in CONVERSION_OPERATIONS
    if selected:
        facts["selected_request"] = request_summary(selected)
        if converting or state.get("is_complete") or intent in ("show_request_details", "provide_information", *CONVERSION_OPERATIONS):
            view = customer_view(state) or {}
            facts["form"] = {
                "converting": converting,
                "form": view.get("conversionLabel"),
                "fields": {f["label"]: f["value"] for f in view.get("fields", [])},
                "missing_required": view.get("missingRequiredFields", []),
                "invalid": view.get("invalidFields", {}),
                "provided_this_turn": [FIELD_LABELS[k] for k in state.get("turn_field_updates") or {}],
            }
        conversion = _conversion_facts(state)
        if conversion:
            facts["conversion"] = conversion
    elif state.get("turn_field_updates"):
        facts["details_received_without_request"] = [FIELD_LABELS[k] for k in state["turn_field_updates"]]

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
    question = (state.get("pending_question") or {}).get("type")
    if question == "missing_fields" and missing:
        return "I still need: " + ", ".join(missing) + "."
    if question == "confirm_conversion":
        ctype = active_type(state)
        return f"Everything needed to create the {ctype.label if ctype else 'account'} is ready. Reply 'confirm' to proceed."
    return "The step finished, but I couldn't write a full reply because the AI service is unavailable. Please try again."

