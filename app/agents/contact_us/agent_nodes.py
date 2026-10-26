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

import math
import re
import time
from datetime import date
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.contact_us.conversion_nodes import (
    STANDALONE_FORM_PREFIX,
    active_type,
    assigned_users,
    conversion_draft,
    conversion_of,
    conversion_record,
    conversion_ref,
    conversion_steps,
    conversion_uploads,
    converted_key,
    draft_type,
    find_providers,
    standalone,
)
from app.agents.contact_us.conversions import (
    ADDRESS_FORM_FIELDS,
    ALL_TYPES,
    CONVERSION_OPERATIONS,
    FIELD_LABELS,
    PARTNER_FORM,
    PARTNER_TYPES,
    STEP_LABELS,
    UPLOAD_EXTENSIONS,
    UPLOAD_MAX_BYTES,
    UPLOAD_SPECS,
    ConversionType,
    build_draft,
    conversion_type,
    is_placeholder,
    partner_type_name,
    read_upload,
    validate_draft,
)
from app.agents.contact_us.nodes import StepFailed, StepResult, tracked_step
from app.agents.contact_us.request_actions import (
    CONTACT_US_STATUS_LABELS,
    COMMENT_LIMIT,
    SETTABLE_STATUSES,
    status_change_facts,
    status_change_view,
)
from app.agents.contact_us.schemas import build_customer_draft, summarize_contact_request
from app.agents.contact_us.state import ContactUsChatState
from app.agents.contact_us.tools import get_provider_list, list_page_size, search_service_providers
from app.agents.contact_us.understanding import INTENT_LABELS, compose_reply, understand
from app.agents.runtime import RunChannel, RunEmitter
from app.core.logging import get_logger
from app.integrations.bootog import BootogApiError
from app.llm import LlmNotConfiguredError

logger = get_logger(__name__)

_LIST_LIMIT = 25  # requests sent to the UI per listing
_FACTS_LIST_LIMIT = 10  # requests described to the LLM per listing

# ContactUsStatus in the host's libs/models/enums/contact-us-status.model.ts.
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
    "load_filter_options": "Filter Options",
    "get_tracking_history": "Request History",
    "prepare_status_change": "Status Change",
    "change_request_status": "Status Change",
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
    letters), or — for a single letter — as an initial ("Sam A" matches "Sam Avery")."""
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


def _understand_status_change(state: ContactUsChatState, result: Any, intent: str, update: dict[str, Any]) -> None:
    """A status change asked for, confirmed or cancelled in the chat. Confirming only marks
    the pending change: change_request_status checks it again before anything is sent."""
    pending = state.get("status_change") or {}
    awaiting = (state.get("pending_question") or {}).get("type") == "confirm_status_change" and pending.get(
        "state") in ("confirming", "choosing")
    if awaiting and intent == "confirm_action":
        if pending.get("state") == "confirming":
            update["status_change_confirmed"] = True
        else:  # no status chosen yet: nothing to confirm, ask again
            update.update(status_change={**pending, "run_id": state.get("run_id")},
                          pending_question={"type": "confirm_status_change"}, awaiting_user_input=True)
        return
    if awaiting and intent == "cancel_operation":
        update["status_change"] = {**pending, "state": "cancelled", "run_id": state.get("run_id")}
        return
    if intent != "change_status":
        if awaiting:
            update["status_change"] = None  # the user moved on: that change is no longer pending
        return
    ask = result.status_change
    update["status_request"] = {"status": ask.status if ask else None, "comment": ask.comment if ask else None}
    positions = [p for p in (ask.positions if ask else []) if isinstance(p, int) and p != 0]
    listed = state.get("listed_request_ids") or []
    if positions:
        indexes = {p: (p - 1 if p > 0 else len(listed) + p) for p in positions}
        missing = [p for p, i in indexes.items() if not 0 <= i < len(listed)]
        if missing:
            # Never fall back to another request: nothing is prepared, the user is told.
            update.update(status_targets=[], request_reference=None, request_filters=None,
                          last_tool_result={"status_change": {"not_in_the_list_shown": missing, "requests_shown": len(listed)}})
        else:
            # Several requests of the list: their ids, nothing to match.
            ids = list(dict.fromkeys(listed[i] for i in indexes.values()))
            update.update(status_targets=ids, request_reference=None, request_filters=None)
    elif awaiting and not any((update.get("request_reference") or {}).get(k) for k in REFERENCE_KEYS):
        # "Make it Approved" answering the pending change: the same requests.
        update.update(status_targets=pending.get("ids"), request_reference=None, request_filters=None)


def merge_list_filters(state: ContactUsChatState, f: Any) -> tuple[dict[str, Any], list[str], bool]:
    """The request-list filters after this turn: the ones in force, minus what the user
    removed, plus what they set (a status or period replaces the previous one; everything
    else is kept). Any change starts again at page 1. -> (filters, problems, company
    removed). With problems the caller lists nothing and keeps the previous filters."""
    active = dict(state.get("active_filters") or {})
    clear = set(f.clear) if f is not None else set()
    problems: list[str] = []
    previous_page = int(active.pop("page", 1) or 1)
    if "all" in clear:
        active = {}
    if "status" in clear:
        for key in ("statuses", "all_statuses", "only_open"):
            active.pop(key, None)
    if "date" in clear:
        active.pop("created_from", None)
        active.pop("created_to", None)
    if "assignee" in clear:
        for key in ("assigned_user_name", "assigned_user_id", "assigned_to_me"):
            active.pop(key, None)
    page = 1
    if f is not None:
        statuses = status_values(f.statuses)
        problems += [f"status '{s}'" for s in f.statuses if not status_values([s])]
        if statuses:
            active.update(statuses=statuses, all_statuses=False, only_open=False)
        elif f.all_statuses:
            active.update(statuses=[], all_statuses=True, only_open=False)
        elif f.only_open:
            active.update(statuses=[], all_statuses=False, only_open=True)
        if f.created_from or f.created_to:
            start, end = _iso_date(f.created_from), _iso_date(f.created_to)
            problems += [f"{label} '{raw}'" for label, raw, ok in (("start date", f.created_from, start), ("end date", f.created_to, end)) if raw and not ok]
            if start and end and start > end:
                problems.append(f"date range {start} to {end} (the start is after the end)")
            active.update(created_from=start, created_to=end)  # a new period replaces the old one
        if f.assigned_to_me:
            # Resolved from the signed-in user's own id (resolve_assigned_user), never a name.
            active.pop("assigned_user_name", None)
            active.pop("assigned_user_id", None)
            active["assigned_to_me"] = True
        elif f.assigned_user_name and f.assigned_user_name.strip():
            active.pop("assigned_user_id", None)  # a new name is resolved again
            active.pop("assigned_to_me", None)
            active["assigned_user_name"] = f.assigned_user_name.strip()[:100]
        if f.page or f.page_number:
            page = f.page_number if f.page_number else previous_page + (1 if f.page == "next" else -1)
            pages = list_pages(state)
            if page < 1:
                problems.append("previous page (this is already the first page)")
            elif pages and page > pages:
                problems.append(f"page {page} (the list has {pages} page{'s' if pages != 1 else ''})")
    active["page"] = page
    return {k: v for k, v in active.items() if v not in (None, [], False) or k == "page"}, problems, (
        "all" in clear or "company" in clear
    )


def query_filters(active: dict[str, Any]) -> dict[str, Any]:
    """GET ContactUs parameters for the filters in force. Status scope: the chosen
    statuses; else every status when asked for, or for a period / an assignee (unless
    "open" was asked for); else the screen's default open-status clause."""
    statuses = active.get("statuses") or []
    assigned = active.get("assigned_user_id") or active.get("assigned_user_name") or active.get("assigned_to_me")
    widened = bool(active.get("all_statuses") or active.get("created_from") or active.get("created_to") or assigned)
    out = {
        "statuses": statuses,
        "all_statuses": not statuses and not active.get("only_open") and widened,
        "created_from": active.get("created_from"),
        "created_to": active.get("created_to"),
        # A resolved assignee is sent as is; a name (or "me") is resolved first.
        "assigned_user_ids": [active["assigned_user_id"]] if active.get("assigned_user_id") else None,
        "assigned_user_name": None if active.get("assigned_user_id") else active.get("assigned_user_name"),
        "assigned_to_me": bool(active.get("assigned_to_me")) and not active.get("assigned_user_id"),
        "page": active.get("page") or 1,
    }
    return {k: v for k, v in out.items() if v}


def request_filters_view(state: ContactUsChatState) -> dict[str, Any]:
    """The chat's filter card: the filters in force and what can be chosen. Status ids are
    ContactUsStatus values; companies are SearchProviders ids (the `targetProviderId`)."""
    active = state.get("active_filters") or {}
    query = query_filters(active)
    catalog = state.get("provider_catalog") or {}
    provider_id = state.get("provider_id")
    return {
        "statusOptions": [{"id": v, "name": n} for v, n in CONTACT_US_STATUS_LABELS.items()],
        "statuses": active.get("statuses") or [],
        # selected: those statuses · all: every status · open: the screen's default (open ones)
        "statusScope": "selected" if query.get("statuses") else "all" if query.get("all_statuses") else "open",
        "allStatuses": bool(active.get("all_statuses")),
        "provider": {"id": provider_id, "name": state.get("provider_name") or provider_id} if provider_id else None,
        "providerOptions": catalog.get("providers") or [],
        "providerOptionsTruncated": bool(catalog.get("truncated")),
        "createdFrom": active.get("created_from"),
        "createdTo": active.get("created_to"),
        "assignee": active.get("assigned_user_name"),
        "assigneeId": active.get("assigned_user_id"),
        # GetContactUsAssignedUser for this module (the screen's Assigned User filter).
        "assigneeOptions": [{"id": u["id"], "name": u["name"]} for u in (state.get("assigned_users_cache") or {}).get("users") or []],
        "page": int(active.get("page") or 1),
        "pages": list_pages(state),
        "pageSize": list_page_size(),
        "total": state.get("contact_requests_total") or 0,
    }


def _describe_filters(state: ContactUsChatState, view: dict[str, Any]) -> dict[str, Any]:
    names = {o["id"]: o["name"] for o in view["statusOptions"]}
    return {
        "status": [names[s] for s in view["statuses"] if s in names] or {
            "all": "every status", "open": "open requests only (Completed, CRM Completed and Closed excluded)",
        }[view["statusScope"] if view["statusScope"] != "selected" else "open"],
        "company": (view["provider"] or {}).get("name") or "all companies",
        "created": {"from": view["createdFrom"], "to": view["createdTo"]} if view["createdFrom"] or view["createdTo"] else "any date",
        "assigned_to": view["assignee"],
    }


def list_pages(state: ContactUsChatState) -> int:
    """Pages of the last list, at one card list per page (0 when unknown)."""
    total = state.get("contact_requests_total")
    return math.ceil(total / list_page_size()) if isinstance(total, int) and total > 0 else 0


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
                "linked_contact_us_request": conv.get("request_id"),  # None: a partner created on its own
            }
            if conv.get("request_id") or standalone(state)
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
        # Kept for follow-ups ("only approved ones"): set only what changes.
        "active_list_filters": _describe_filters(state, request_filters_view(state)),
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
        "form_submitted": False,
        "explicit_create": False,
        "completed_this_turn": False,
        "rejected_field_updates": {},
        "rejected_uploads": {},
        "filter_problems": [],
        "company_pick": None,
        "filters_before": None,
        "status_targets": None,
        "status_request": None,
        "status_change_confirmed": False,
        "history": None,
        "post_creation_edit": [],
        "edit_request": None,
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
    if state.get("form_submission"):
        return _understand_form(state, state["form_submission"])
    try:
        result = await understand(config, state.get("conversation_history") or [], _state_summary(state))
    except LlmNotConfiguredError as exc:
        raise StepFailed("The AI service is not configured, so I can't process messages yet.") from exc

    intent = result.intent
    if (state.get("pending_question") or {}).get("type") == "next_action" and intent == "confirm_action":
        # "Yes" to "convert another request?" names no request: show the open ones to pick
        # from. A bare yes never starts a conversion.
        intent = "list_requests"
    company_pick = assignee_pick = None
    pending_type = (state.get("pending_question") or {}).get("type")
    if pending_type in ("choose_company", "choose_assignee") and result.conversion is not None:
        # A number answering "which company / assignee?" picks from those offered, and lists.
        candidates = state.get("company_candidates" if pending_type == "choose_company" else "assignee_candidates") or []
        pos = result.conversion.option_position or 0
        if 1 <= pos <= len(candidates):
            intent = "list_requests"
            if pending_type == "choose_company":
                company_pick = candidates[pos - 1]
            else:
                assignee_pick = candidates[pos - 1]
    update: dict[str, Any] = {
        "intent": intent,
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
    # "No" to a status-change confirmation cancels only that change (below), never a conversion.
    answers_status_change = pending_type == "confirm_status_change" and result.intent in ("confirm_action", "cancel_operation")
    if operation:
        update["operation"] = operation
    elif answers_status_change:
        pass
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

    previous_question = (state.get("pending_question") or {}).get("type")
    if conv is not None and conv.review_first:
        update["turn_conversion"] = {**update.get("turn_conversion", {}), "review_first": True}
    if result.intent == "review_details":
        update["turn_conversion"] = {**update.get("turn_conversion", {}), "review_first": True}
    # "Create it" counts only as the answer to the ready-to-create summary the user saw.
    update["explicit_create"] = previous_question == "ready_to_create" and result.intent in (
        "confirm_action", "create_customer", "convert_request"
    )
    if result.intent == "edit_details" and not update.get("turn_field_updates"):
        wanted = _edit_fields(conv.edit_fields if conv is not None else [])
        update["edit_request"] = {"fields": wanted}

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
    _understand_status_change(state, result, intent, update)

    if intent == "list_requests":
        # A request list: this turn's filters are applied ON TOP of those already in force.
        active, problems, clear_company = merge_list_filters(state, result.filters)
        problems = list(dict.fromkeys([*ignored, *problems]))
        update["filters_before"] = dict(state.get("active_filters") or {})
        if assignee_pick:
            for key in ("assigned_user_name", "assigned_to_me"):
                active.pop(key, None)
            active.update(assigned_user_id=assignee_pick["id"], assigned_user_name=assignee_pick["name"])
        if company_pick:
            update["company_pick"] = company_pick
        elif clear_company and state.get("provider_id"):
            update["company_name"] = "all"
        if problems:
            # Never list with a filter silently dropped: ask instead, the filters stay as they were.
            update.update(filter_problems=problems, request_filters=None)
        else:
            update.update(active_filters=active, request_filters={
                **{k: v for k, v in filters.items() if k in ("contact_name", "email", "request_id")},
                **query_filters(active),
                "incomplete_only": bool(result.filters and result.filters.incomplete_only),
            })
        message = update["request_summary"] or INTENT_LABELS[intent]
        return StepResult(message, update, data={"intent": intent})

    if ignored:
        filters["ignored"] = ignored
    update["request_filters"] = {k: v for k, v in filters.items() if v} or None

    if result.intent == "list_providers" and result.provider_query:
        q = result.provider_query
        update["provider_query"] = {
            "name": (q.name or "").strip()[:100] or None,
            "location": (q.location or "").strip()[:100] or None,
        }

    message = update["request_summary"] or INTENT_LABELS[intent]
    return StepResult(message, update, data={"intent": intent})


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


def _edit_fields(requested: list[str]) -> list[str]:
    """Field keys the user wants to change ("address" = the address fields); all editable
    fields when nothing specific was named."""
    keys: list[str] = []
    for item in requested:
        name = (item or "").strip()
        if name.casefold() == "address":
            keys += list(ADDRESS_FORM_FIELDS)
        elif name in FIELD_LABELS and name not in ("latitude", "longitude"):
            keys.append(name)
        else:
            label_match = [k for k, label in FIELD_LABELS.items() if label.casefold() == name.casefold()]
            keys += label_match
    return list(dict.fromkeys(keys)) or [k for k in FIELD_LABELS if k not in ("latitude", "longitude")]


def _understand_form(state: ContactUsChatState, form: dict[str, Any]) -> StepResult:
    """A chat card's structured form: no interpretation needed, so no LLM call. Only known
    fields of the current conversion form are taken, for the request the card belongs to."""
    if form.get("action") == "apply_filters":
        return _understand_filter_form(state, form)
    if form.get("action") == "status_change":
        return _understand_status_form(state, form)
    record = state.get("contact_request") or {}
    update: dict[str, Any] = {
        "answered_question": state.get("pending_question"),
        "pending_question": None,
        "awaiting_user_input": False,
        "request_reference": {"use_current": True},
    }
    request_id = str(form.get("request_id") or "").strip()
    # A partner card without a Contact Us request names its form id instead (never a
    # request id: the prefix can't match one).
    unlinked_card = request_id.startswith(STANDALONE_FORM_PREFIX)
    conv = conversion_of(state)
    # No request selected at all: the conversation's saved state was lost (the agent
    # restarted and the chat was restored from its transcript only). The card still names
    # the request it was shown for, so that exact request is selected again — BOOTOG
    # returns it only to a caller allowed to see it — instead of refusing the form.
    restored = not record and not unlinked_card and bool(_REQUEST_ID.match(request_id))
    # The same for a partner card without a request: its partner form is started again.
    unlinked_lost = unlinked_card and not record and not conv.get("standalone")
    stale = (
        (record or conv.get("form_id") != request_id) and not unlinked_lost
        if unlinked_card
        else not restored and request_id != str(record.get("id"))
    )
    if stale:
        # A request IS selected (or another partner form is open) and the card belongs to
        # something else: apply nothing.
        update.update(intent="other", request_summary="Submitted a form for another request",
                      last_tool_result={"form": "This form belongs to a request that is no longer selected; nothing was changed."})
        return StepResult("Form ignored — it belongs to another request", update)

    form_type = conversion_type(form.get("conversion_type")) if restored or unlinked_lost else None
    if unlinked_lost and form_type not in PARTNER_TYPES:
        form_type = None
    # A restored Partner card whose company type was still open keeps its Partner form.
    pending_partner = (restored and form.get("conversion_type") == PARTNER_FORM.key) or (unlinked_lost and not form_type)
    ctype = form_type or (PARTNER_FORM if pending_partner else draft_type(state))
    current = {} if restored or unlinked_lost else customer_draft(state) or {}
    fields: dict[str, str] = {}
    for key, value in (form.get("fields") or {}).items():
        if key not in ctype.fields or key in ("latitude", "longitude") or not isinstance(value, str):
            continue
        text = " ".join(value.split())[:200]
        if not text or is_placeholder(text):
            continue  # an empty input never clears a value that is already there
        if text.casefold() != " ".join(str(current.get(key) or "").split()).casefold():
            fields[key] = text
    if "latitude" in ctype.fields:
        for key, value in _form_coordinates(form.get("fields") or {}).items():
            if value != str(current.get(key) or ""):
                fields[key] = value
    provider_id = (form.get("fields") or {}).get(PROVIDER_FORM_KEY)
    current_provider = (conversion_of(state).get("target_provider") or {}).get("id")
    if ctype.needs_target_provider and isinstance(provider_id, str) and provider_id.strip() and provider_id.strip() != current_provider:
        # Checked against BOOTOG's provider list in resolve_target_provider; never trusted as is.
        update["turn_conversion"] = {"target_provider_id": provider_id.strip()[:64]}
    # Vendor Company Type (Partner forms): one of the partner types, never anything else.
    partner = conversion_type((form.get("fields") or {}).get(COMPANY_TYPE_FORM_KEY))
    type_changed = (
        ctype.operation == "create_partner" and partner in PARTNER_TYPES and partner is not active_type(state)
    )
    if type_changed:
        update["turn_conversion"] = {**(update.get("turn_conversion") or {}), "partner_type": partner.key}
        if (restored or unlinked_lost) and form_type is None:
            form_type = partner
    files_update, uploads_changed, uploads_rejected = _form_uploads(state, form, form_type or ctype, request_id)
    if files_update is not None:
        update["conversion_files"] = files_update
    create = form.get("action") == "create"
    operation = state.get("operation") if state.get("operation") in CONVERSION_OPERATIONS else ctype.operation
    if type_changed:
        operation = "create_partner"
    update.update(
        intent="confirm_action" if create else "provide_information",
        request_summary="Create the customer" if create else "Submitted the details form",
        operation=operation,
        turn_field_updates=fields,
        form_submitted=True,
        explicit_create=create,
        rejected_uploads=uploads_rejected,
    )
    changed = (
        [FIELD_LABELS[k] for k in fields]
        + ([COMPANY_TYPE_LABEL] if type_changed else [])
        + ([PROVIDER_LABEL] if (update.get("turn_conversion") or {}).get("target_provider_id") else [])
        + uploads_changed
    )
    if unlinked_lost:
        # Started again under the card's own form id, so its images still apply. As after
        # any restore, nothing is created from this submission: the summary is confirmed first.
        turn = {**(update.get("turn_conversion") or {}), "review_first": True, "form_id": request_id}
        if form_type:
            turn["partner_type"] = form_type.key
        update.update(
            operation="create_partner", turn_conversion=turn, request_reference=None, explicit_create=False,
            last_tool_result={"form": "The conversation had been restored after the agent restarted, so the partner "
                              "form was started again from the card with the submitted details. Earlier changes made "
                              "in the chat may be missing: ask the user to check the details. Nothing was created."},
        )
        return StepResult("Starting the partner form again from the card", update, data={"form": form.get("action"), "fields": list(fields)})
    if restored:
        _restore_from_form(update, request_id, form_type)
        return StepResult("Selecting the card's request again", update, data={"form": form.get("action"), "fields": list(fields)})
    label = "Create requested" if create else (f"Details submitted: {', '.join(changed)}" if changed else "Details submitted (no changes)")
    return StepResult(label, update, data={"form": form.get("action"), "fields": list(fields)})


def _restore_from_form(update: dict[str, Any], request_id: str, ctype: ConversionType | None) -> None:
    """The card's request is selected again by its id and the submitted fields apply to
    it. Details the user gave before the state was lost are gone, so nothing is created
    from this submission: the restored summary is shown and waits for an explicit Create
    (Customer) or confirmation (B2B / Partners). Without a known conversion type the
    request and details are only shown — a conversion type is never guessed."""
    turn = dict(update.get("turn_conversion") or {})
    if ctype is not None:
        turn["review_first"] = True
        if ctype in PARTNER_TYPES:
            turn["partner_type"] = ctype.key
    update.update(
        intent="provide_information",
        request_summary="Submitted the details form (conversation restored)",
        operation=ctype.operation if ctype else None,
        request_reference={"use_current": False, "request_id": request_id},
        request_filters={"request_id": request_id},
        turn_conversion=turn,
        form_submitted=False,
        explicit_create=False,
        last_tool_result={
            "form": "The conversation had been restored after the agent restarted, so the request was "
            "selected again from the card and the submitted details were applied to it. Earlier changes "
            "made in the chat before the restart may be missing: ask the user to check the details. "
            "Nothing was created from this submission."
        },
    )


def _understand_status_form(state: ContactUsChatState, form: dict[str, Any]) -> StepResult:
    """The status-change card: "Change status" confirms exactly the status and comment the
    card shows, for the requests it shows (its token); "Cancel" drops the change. A card
    that is no longer the pending change (already done, replaced, cancelled) changes nothing."""
    fields = {k: v.strip() for k, v in (form.get("fields") or {}).items() if isinstance(v, str)}
    pending = state.get("status_change") or {}
    update: dict[str, Any] = {
        "intent": "change_status",
        "answered_question": state.get("pending_question"),
        "pending_question": None,
        "awaiting_user_input": False,
        "form_submitted": True,
    }
    live = pending.get("state") in ("confirming", "choosing") and fields.get("token") and fields["token"] == pending.get("token")
    if not live:
        update.update(intent="other", request_summary="Submitted an old status change",
                      last_tool_result={"status_change": "That status change is no longer pending; nothing was changed."})
        return StepResult("Status change ignored — no longer pending", update)
    if fields.get("decision") == "cancel":
        update.update(request_summary="Cancelled the status change",
                      status_change={**pending, "state": "cancelled", "run_id": state.get("run_id")})
        return StepResult("Status change cancelled", update)
    status = int(fields["status"]) if fields.get("status", "").isdigit() else None
    comment = fields.get("comment", "")[:COMMENT_LIMIT] or None
    if status not in SETTABLE_STATUSES:
        update.update(request_summary="Status change without a valid status",
                      status_change={**pending, "run_id": state.get("run_id"), "problem": "Choose one of the statuses offered"},
                      pending_question={"type": "confirm_status_change"}, awaiting_user_input=True)
        return StepResult("No valid status chosen — asking again", update)
    update.update(
        intent="confirm_action",
        request_summary=f"Change the status to {SETTABLE_STATUSES[status]}",
        status_change={**pending, "state": "confirming", "status": {"id": status, "name": SETTABLE_STATUSES[status]},
                       "comment": comment, "problem": None},
        status_change_confirmed=True,
    )
    return StepResult(f"Status change confirmed from the card: {SETTABLE_STATUSES[status]}", update)


def _understand_filter_form(state: ContactUsChatState, form: dict[str, Any]) -> StepResult:
    """The request list's filter card: the complete set of filters the user wants (what it
    leaves empty is removed). Each value is checked: status ids against ContactUsStatus,
    dates as YYYY-MM-DD with start <= end, the company against the list the card offered.
    A value that fails lists nothing and keeps the previous filters."""
    fields = {k: v.strip() for k, v in (form.get("fields") or {}).items() if isinstance(v, str)}
    problems: list[str] = []
    statuses: list[int] = []
    for raw in filter(None, fields.get("statuses", "").split(",")):
        value = int(raw) if raw.strip().isdigit() else None
        if value in CONTACT_US_STATUS_LABELS:
            if value not in statuses:
                statuses.append(value)
        else:
            problems.append(f"status '{raw.strip()[:20]}'")
    start, end = _iso_date(fields.get("createdFrom")), _iso_date(fields.get("createdTo"))
    for label, raw, ok in (("start date", fields.get("createdFrom"), start), ("end date", fields.get("createdTo"), end)):
        if raw and not ok:
            problems.append(f"{label} '{raw[:20]}'")
    if start and end and start > end:
        problems.append(f"date range {start} to {end} (the start is after the end)")
    page = int(fields["page"]) if fields.get("page", "").isdigit() and int(fields["page"]) > 0 else 1
    pages = list_pages(state)
    if pages and page > pages:
        problems.append(f"page {page} (the list has {pages} page{'s' if pages != 1 else ''})")

    update: dict[str, Any] = {
        "intent": "list_requests",
        "request_summary": "Applied the list filters",
        "answered_question": state.get("pending_question"),
        "pending_question": None,
        "awaiting_user_input": False,
        "form_submitted": True,
    }
    provider_id = fields.get("providerId", "")
    if provider_id and provider_id != state.get("provider_id"):
        offered = {p["id"]: p for p in (state.get("provider_catalog") or {}).get("providers") or []}
        if provider_id in offered:
            update["company_pick"] = {"id": provider_id, "name": offered[provider_id]["name"]}
        else:
            problems.append("company (not in the company list)")  # never an id the card wasn't given
    elif not provider_id and state.get("provider_id"):
        update["company_name"] = "all"
    before = state.get("active_filters") or {}
    assignee = {k: before.get(k) for k in ("assigned_user_id", "assigned_user_name", "assigned_to_me")}
    if "assignedUserId" in fields:  # the card's Assigned User; without it, the one in force stays
        wanted = fields["assignedUserId"]
        offered = {u["id"]: u for u in (state.get("assigned_users_cache") or {}).get("users") or []}
        if not wanted:
            assignee = {}
        elif wanted == before.get("assigned_user_id"):
            pass
        elif wanted in offered:
            assignee = {"assigned_user_id": wanted, "assigned_user_name": offered[wanted]["name"]}
        else:
            problems.append("assigned user (not in the assigned-user list)")
    if problems:
        update.update(filter_problems=problems, request_filters=None, company_pick=None, company_name=None)
        return StepResult(f"Filters not applied: {', '.join(problems)}", update)

    active = {
        "statuses": statuses,
        "all_statuses": not statuses and fields.get("allStatuses") == "true",
        "created_from": start,
        "created_to": end,
        **assignee,
        "page": page,
    }
    active = {k: v for k, v in active.items() if v or k == "page"}
    update.update(active_filters=active, request_filters=query_filters(active))
    return StepResult("Filters applied from the card", update, data={"form": "apply_filters"})


def _form_uploads(
    state: ContactUsChatState, form: dict[str, Any], ctype: ConversionType, request_id: str
) -> tuple[dict[str, Any] | None, list[str], dict[str, str]]:
    """Images changed on the card -> (new `conversion_files` or None, labels changed,
    {label: why not accepted}). A file that can't be accepted keeps the attached one, and
    nothing changes once the account exists (the images were sent with it)."""
    submitted = form.get("files") or {}
    if not ctype.uploads or not submitted or account_exists(state):
        return None, [], {}
    held = state.get("conversion_files") or {}
    files = dict(held.get("files") or {}) if held.get("request_id") == request_id else {}
    changed: list[str] = []
    rejected: dict[str, str] = {}
    for key, value in submitted.items():
        if key not in ctype.uploads:
            continue
        label = UPLOAD_SPECS[key].label
        if value is None:
            if files.pop(key, None):
                changed.append(label)
            continue
        upload, error = read_upload(value)
        if error:
            rejected[label] = error
        elif (files.get(key) or {}).get("sha") != upload["sha"]:
            files[key] = upload
            changed.append(label)
    if not changed:
        return None, [], rejected
    return {"request_id": request_id, "files": files}, changed, rejected


def _form_coordinates(values: dict[str, Any]) -> dict[str, str]:
    """Coordinates on a form submission: the card sends them only for an address picked
    from Google Places and not edited since (they are never typed in). Taken only as a
    pair of in-range numbers; anything else is dropped, never guessed."""
    lat, lng = values.get("latitude"), values.get("longitude")
    if not isinstance(lat, str) or not isinstance(lng, str):
        return {}
    try:
        lat_n, lng_n = float(lat), float(lng)
    except ValueError:
        return {}
    if not (-90 <= lat_n <= 90 and -180 <= lng_n <= 180) or (lat_n == 0 and lng_n == 0):
        return {}  # also NaN/inf; 0,0 is "not geocoded"
    return {"latitude": lat.strip(), "longitude": lng.strip()}


def _operation_for(intent: str, goal: str, current: str | None) -> str | None:
    operation = intent if intent in CONVERSION_OPERATIONS else goal if goal in CONVERSION_OPERATIONS else None
    if operation == "convert_request" and current in CONVERSION_OPERATIONS:
        return current  # "proceed" continues the conversion already chosen; it never guesses one
    return operation


_COMPANY_OPTIONS = 10  # companies offered when a name matches several
_CATALOG_SECONDS = 600


def provider_catalog_stale(state: ContactUsChatState) -> bool:
    """The filter card's company or assigned-user list is due for a reload."""
    assignees = state.get("assigned_users_cache") or {}
    return (
        time.time() - (state.get("provider_catalog") or {}).get("at", 0) > _CATALOG_SECONDS
        or assignees.get("module") != state.get("module_filter")
        or time.time() - assignees.get("at", 0) > _CATALOG_SECONDS
    )


@tracked_step("load_filter_options", "Loading the filter options", "Unable to load the filter options.")
async def load_filter_options(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    """The filter card's dropdowns, each read at most every 10 minutes: the Contact Us
    screen's company list (SearchProviders, every page, bounded) and its assigned users
    (GetContactUsAssignedUser). A list BOOTOG refuses is simply not offered on the card;
    the request list itself is unaffected."""
    update: dict[str, Any] = {}
    parts: list[str] = []
    if time.time() - (state.get("provider_catalog") or {}).get("at", 0) > _CATALOG_SECONDS:
        try:
            result = await search_service_providers.ainvoke({"all_pages": True}, config=config)
            providers = [{"id": p["id"], "name": p["name"]} for p in result["providers"]]
            update["provider_catalog"] = {"at": time.time(), "providers": providers, "truncated": result["truncated"]}
            parts.append(f"{len(providers)} companies")
        except BootogApiError:
            update["provider_catalog"] = {"at": time.time(), "providers": [], "unavailable": True}
            parts.append("company list unavailable")
    try:
        users, cached = await assigned_users(state, config)
        update.update(cached)
        parts.append(f"{len(users)} assigned users")
    except BootogApiError:
        update["assigned_users_cache"] = {"at": time.time(), "module": state.get("module_filter"), "users": []}
        parts.append("assigned-user list unavailable")
    return StepResult(" · ".join(parts) or "Filter options are up to date", update)


async def _find_companies(config: RunnableConfig, wanted: str) -> list[dict[str, str]]:
    """[{id, name}] matching `wanted`: an exact name alone when there is one. SearchProviders
    first (the Contact Us screen's company filter); the session's own companies when it
    finds none or refuses (callers without the Providers.Manage permission)."""
    found: list[dict[str, str]] = []
    try:
        providers, _ = await find_providers(config, wanted, None)
        found = [{"id": p["id"], "name": p["name"]} for p in providers]
    except BootogApiError as exc:
        if exc.status_code not in (401, 403):
            raise
    if not found:
        session = await get_provider_list.ainvoke({}, config=config)
        found = [{"id": p["providerId"], "name": p["providerName"]} for p in session
                 if wanted.casefold() in p["providerName"].casefold()]
    key = " ".join(wanted.casefold().split())
    exact = [p for p in found if " ".join(p["name"].casefold().split()) == key]
    return exact or found


@tracked_step("resolve_provider", "Finding the company", "Unable to look up companies.")
async def resolve_provider(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    """The company whose requests are listed (`targetProviderId`). Names are matched in the
    list the Contact Us screen's company filter uses (SearchProviders), then in the
    session's own companies. One match is used; several are offered to choose from; none
    lists nothing — an unresolved company is never replaced by another or dropped."""
    wanted = (state.get("company_name") or "").strip()
    pick = state.get("company_pick")
    if pick:
        provider_id, provider_name = pick["id"], pick["name"]
    elif wanted.casefold() in ("all", "all companies", "any", "any company", "every company"):
        provider_id, provider_name = None, None
    else:
        found = await _find_companies(config, wanted)
        if len(found) != 1:
            status = "ambiguous" if found else "not_found"
            candidates = found[:_COMPANY_OPTIONS]
            note = {"company": {
                "status": status, "requested": wanted,
                "candidates": [{"number": i + 1, "name": p["name"]} for i, p in enumerate(candidates)],
                "more": len(found) > len(candidates),
                "kept": state.get("provider_name") or "all companies",
            }}
            update: dict[str, Any] = {
                "last_tool_result": note,
                "filter_problems": [*(state.get("filter_problems") or []), f"company '{wanted}'"],
                "company_candidates": candidates or None,
            }
            if candidates:
                update.update(pending_question={"type": "choose_company", "options": [p["name"] for p in candidates]},
                              awaiting_user_input=True)
            message = f"{len(found)} companies match “{wanted}” — asking which one" if found else f"No company matches “{wanted}”"
            return StepResult(message, update)
        provider_id, provider_name = found[0]["id"], found[0]["name"]

    update = {"provider_id": provider_id, "provider_name": provider_name, "company_candidates": None}
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
            # "Sam A" means the people named exactly that when there are any; an
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
    """Merges this turn's values into the one draft of the selected request.

    - Empty values never reach here (dropped when understood), so nothing is cleared.
    - A value that would make the draft invalid is not stored; the previous value stays
      and the rejection is reported ("Reject invalid values without destroying the
      previous valid value").
    - Once the account exists, nothing changes: there is no BOOTOG API to update it, and
      the draft must keep matching what was created.
    - `completed_this_turn`: the draft was incomplete before and is complete now — the
      user's message supplied the last missing details."""
    previous = state.get("user_customer_fields") or {}
    updates = dict(state.get("turn_field_updates") or {})
    extra_new = state.get("turn_extra_information") or {}
    update: dict[str, Any] = {"extra_information": {**(state.get("extra_information") or {}), **extra_new}}
    noted = f"Noted: {', '.join(k.replace('_', ' ') for k in extra_new)}" if extra_new else ""

    if updates and account_exists(state):
        labels = [FIELD_LABELS[k] for k in updates]
        update["post_creation_edit"] = labels
        message = "The account already exists — changes were not applied: " + ", ".join(labels)
        return StepResult(" · ".join(p for p in (message, noted) if p), update, data={"fields": []})

    record = conversion_record(state)
    rejected: dict[str, str] = {}
    candidate = {**previous, **updates}
    completed = False
    if record:
        ctype = draft_type(state)
        _, missing_before, invalid_before = validate_draft(ctype, build_draft(ctype, record, previous))
        _, _, invalid = validate_draft(ctype, build_draft(ctype, record, candidate))
        for key in updates:
            reason = invalid.get(FIELD_LABELS[key])
            if reason:
                rejected[FIELD_LABELS[key]] = reason
                if key in previous:
                    candidate[key] = previous[key]
                else:
                    candidate.pop(key)
        _, missing_after, invalid_after = validate_draft(ctype, build_draft(ctype, record, candidate))
        applied = {k for k in updates if candidate.get(k) == updates[k]}
        completed = bool(applied) and bool(missing_before or invalid_before) and not (missing_after or invalid_after)
        new_email = build_draft(ctype, record, candidate).get("email")
        if new_email != build_draft(ctype, record, previous).get("email"):
            conv = _forget_email_checks(conversion_of(state), record)
            if conv is not None:
                update["conversion"] = conv

    applied_keys = [k for k in updates if candidate.get(k) == updates[k]]
    corrected = [FIELD_LABELS[k] for k in applied_keys if previous.get(k) and previous[k] != updates[k]]
    received = [FIELD_LABELS[k] for k in applied_keys if FIELD_LABELS[k] not in corrected]
    parts = []
    if received:
        parts.append(f"Received: {', '.join(received)}")
    if corrected:
        parts.append(f"Corrected: {', '.join(corrected)}")
    if rejected:
        parts.append(f"Not accepted: {', '.join(rejected)}")
    if noted:
        parts.append(noted)
    update.update(user_customer_fields=candidate, rejected_field_updates=rejected, completed_this_turn=completed)
    return StepResult(" · ".join(parts) or "Nothing new to record", update, data={"fields": received + corrected})


def _forget_email_checks(conv: dict[str, Any], record: dict[str, Any]) -> dict[str, Any] | None:
    """The email changed: results that were about the previous email (the existing-account
    check, "already registered", BOOTOG's "email already exists" rejection) no longer
    apply. The new email is checked afresh before anything is created."""
    if not conv or conv.get("request_id") != record.get("id"):
        return None
    conv["account"] = None
    if conv.get("blocked") == "existing_role":
        conv["blocked"] = None
    if (conv.get("failed") or {}).get("code") == "EMAIL_EXISTS":
        conv["failed"] = None  # a definitive rejection: nothing was created
    return conv


def email_conflict(state: ContactUsChatState) -> bool:
    """The current email is what stops the conversion: it already has the role, or BOOTOG
    rejected it as already registered. The card then opens with the Email input."""
    conv = conversion_of(state)
    return conv.get("blocked") == "existing_role" or (conv.get("failed") or {}).get("code") == "EMAIL_EXISTS"


def account_exists(state: ContactUsChatState) -> bool:
    """The account for the selected request and conversion type was created (in this
    conversation), so its details can no longer change through this workflow."""
    record = state.get("contact_request") or {}
    conv = conversion_of(state)
    if conv.get("request_id") == record.get("id") and "create_account" in (conv.get("completed") or []):
        return True
    ctype = active_type(state)
    return bool(ctype and (state.get("created_customers") or {}).get(converted_key(conversion_ref(state), ctype.key)))


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
    conv = conversion_of(state)
    return bool(ctype) and set(conversion_steps(ctype, conv)) <= set(conv.get("completed") or [])


def finalize_turn(state: ContactUsChatState) -> dict[str, Any]:
    """Deterministic end-of-turn bookkeeping: what was shown, what was created, status."""
    update: dict[str, Any] = {}
    if listed_this_turn(state):
        update["listed_request_ids"] = [r.get("id") for r in listed_records(state)[:_LIST_LIMIT]]

    conv = conversion_of(state)
    # The request, or for a partner without one its form id (each such partner is its own).
    ref = conv.get("request_id") or (conv.get("form_id") if conv.get("standalone") else None)
    ctype = active_type(state)
    last_step = conversion_steps(ctype, conv)[-1] if ctype else None
    if last_step in (state.get("turn_steps") or []) and conversion_finished(state) and ref:
        created_id = conv.get("user_id") or conv.get("created_vendor_id") or "created"
        key = converted_key(ref, conv.get("type"))
        update["created_customers"] = {**(state.get("created_customers") or {}), key: str(created_id)}
        update["operation"] = None
        update["is_complete"] = True
        # Offered, not awaited: the turn is complete; the next message is read against it.
        update["pending_question"] = {"type": "next_action", "options": list(NEXT_ACTIONS)}

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
    pending = [s for s in conversion_steps(ctype, conv) if s not in done] if ctype else []
    if "create_account" in done:
        return {
            "status": "created",
            "customer_id": conv.get("user_id") or conv.get("created_vendor_id"),
            "this_turn": bool(state.get("is_complete")) or "create_account" in (state.get("turn_steps") or []),
            "pending_steps": [STEP_LABELS[s] for s in pending],
            "message": (conv.get("failed") or {}).get("message"),
        }
    created = (state.get("created_customers") or {}).get(converted_key(conversion_ref(state), ctype.key if ctype else None))
    if created:
        return {"status": "created", "customer_id": created, "this_turn": bool(state.get("is_complete")), "pending_steps": []}
    if conv.get("blocked") in ("existing_role", "already_converted", "type_locked"):
        return {"status": "blocked", "message": conv["blocked"]}
    if conv.get("failed"):
        return {"status": "failed", "message": conv["failed"].get("message")}
    return {"status": "not_attempted"}


_NOT_TYPED = ("latitude", "longitude")  # never typed in: from the request only
# Form key of the card's "Select Service Provider" (B2B / Partners); the value is a provider id.
PROVIDER_FORM_KEY = "targetProviderId"
PROVIDER_LABEL = "Service Provider"
# Form key of the Partner card's "Vendor Company Type"; the value is a partner type key.
COMPANY_TYPE_FORM_KEY = "partnerType"
COMPANY_TYPE_LABEL = "Vendor Company Type"


def provider_missing(state: ContactUsChatState) -> bool:
    ctype = active_type(state)
    return bool(ctype and ctype.needs_target_provider and not conversion_of(state).get("target_provider"))


def conversion_state(state: ContactUsChatState, missing: list[str], invalid: dict[str, str]) -> str:
    """Where the operation stands, from structured state only (never from the reply):
    collecting · ready · confirming · existing_customer · failed · outcome_unknown ·
    status_pending (account created, request not yet CRM Completed) · partial (account
    created, another step pending) · completed."""
    record = state.get("contact_request") or {}
    conv = conversion_of(state)
    if conv.get("request_id") != record.get("id"):
        conv = {}
    ctype = active_type(state)
    done = set(conv.get("completed") or [])
    if (ctype and set(conversion_steps(ctype, conv)) <= done) or conv.get("blocked") == "already_converted" or (
        ctype and (state.get("created_customers") or {}).get(converted_key(conversion_ref(state), ctype.key))
    ):
        return "completed"
    if "create_account" in done:
        pending = [s for s in conversion_steps(ctype, conv) if s not in done] if ctype else []
        return "status_pending" if pending == ["update_request_status"] else "partial"
    failed = conv.get("failed") or {}
    if failed.get("step") == "create_account":
        return "outcome_unknown" if failed.get("outcome_unknown") else "failed"
    if conv.get("blocked") == "existing_role":
        return "existing_customer"
    if missing or invalid:
        return "collecting"
    if (conv.get("account") or {}).get("verified") is False or conv.get("role_unverified"):
        return "unverified"  # the account or role check failed: nothing can be created yet
    question = (state.get("pending_question") or {}).get("type")
    if question == "confirm_conversion":
        return "confirming"
    return "ready"


def existing_account_warning(ctype_label: str) -> str:
    return f"A {ctype_label} account already exists for this email. No duplicate account was created."


VERIFICATION_FAILED = "I couldn't verify this email because BOOTOG returned an error. Please retry once the issue is resolved."


def customer_view(state: ContactUsChatState) -> dict[str, Any] | None:
    """The conversion card for the UI, from the one authoritative draft: every field with
    its value (as it would be sent: FL, US, phone mask), source and status, what the
    form should ask for, and where the operation stands."""
    raw = customer_draft(state)
    if raw is None:
        return None
    ctype = draft_type(state)
    draft, missing, invalid = validate_draft(ctype, raw)
    # A value the user just tried to set and that was not accepted is shown as that
    # field's error (the previous value stays in place).
    errors = {**invalid, **(state.get("rejected_field_updates") or {})}
    user_fields = state.get("user_customer_fields") or {}
    conv = conversion_of(state)
    if conv.get("request_id") != (state.get("contact_request") or {}).get("id"):
        conv = {}
    # The service provider is required like a field: nothing is "ready" without it.
    if provider_missing(state) and conv:
        missing = [PROVIDER_LABEL, *missing]
    # So is a partner's company type (the Partner form shows until it is chosen).
    if ctype is PARTNER_FORM:
        missing = [COMPANY_TYPE_LABEL, *missing]
    status = conversion_state(state, missing, errors)
    editable = not account_exists(state) and status not in ("completed", "status_pending", "partial")
    problem_keys = {k for k in ctype.fields if FIELD_LABELS[k] in missing or FIELD_LABELS[k] in errors}
    if editable and status == "collecting":
        form_fields = [k for k in ctype.fields if k in ADDRESS_FORM_FIELDS or k in problem_keys]
    elif editable and email_conflict(state):
        form_fields = ["email"]  # a different email is the way forward; the rest is kept
    else:
        form_fields = []
    record = state.get("contact_request") or {}
    email = draft.get("email")
    notice = None
    if (conv.get("failed") or {}).get("kind") == "created_with_error":
        notice = {
            "level": "warning",
            "text": f"BOOTOG returned an error while creating this account, but {email} now has the "
            f"{ctype.role_name} role, which it did not have just before: the account was most likely created by "
            "this attempt and may be incomplete. CRM completion is not verified — the request is not marked CRM "
            "Completed. Retry to verify the account and finish.",
        }
    elif status == "existing_customer" and (conv.get("account") or {}).get("role_removed"):
        notice = {
            "level": "warning",
            "text": f"{email} has a removed {ctype.role_name} role in BOOTOG, and BOOTOG does not allow adding it "
            "again, so no account was created and the request was not changed. An administrator has to restore "
            "the role in BOOTOG, or enter a different email.",
        }
    elif status == "existing_customer":
        notice = {
            "level": "warning",
            "text": f"{existing_account_warning(ctype.label)} {email} already has the {ctype.role_name} role. "
            "Enter a different email to continue.",
        }
    elif status == "failed" and email_conflict(state):
        notice = {
            "level": "warning",
            "text": f"BOOTOG reports that {email} is already registered, so no account was created. "
            "Enter a different email to continue.",
        }
    elif status == "unverified" and conv.get("role_unverified"):
        notice = {
            "level": "warning",
            "text": f"The {ctype.label} role could not be verified in BOOTOG, so the account was not created. Please try again.",
        }
    elif status == "unverified":
        notice = {"level": "warning", "text": f"{VERIFICATION_FAILED} No account was created."}
    elif standalone(state):
        notice = {
            "level": "info",
            "text": "This partner is not linked to a Contact Us request, so no request status will change. "
            "To link it, name the request in the chat.",
        }
    provider_field = None
    if ctype.needs_target_provider:
        chosen = conv.get("target_provider")
        rejected = (state.get("pending_question") or {}).get("status") == "invalid_choice"
        provider_field = {
            "key": PROVIDER_FORM_KEY,
            "label": PROVIDER_LABEL,
            "required": True,
            "value": {"id": chosen["id"], "name": chosen["name"]} if chosen else None,
            "status": "valid" if chosen else "missing",
            "error": "Choose a provider from the list" if rejected else None,
            # Only while it can still be chosen or changed; ids come from SearchProviders.
            "options": [{"id": p["id"], "name": p["name"], "address": p.get("address")}
                        for p in conv.get("provider_options") or []] if editable else [],
            "optionsTruncated": bool(conv.get("provider_options_truncated")),
        }
    company_type_field = None
    if ctype is PARTNER_FORM or ctype in PARTNER_TYPES:
        chosen = None if ctype is PARTNER_FORM else ctype
        company_type_field = {
            "key": COMPANY_TYPE_FORM_KEY,
            "label": COMPANY_TYPE_LABEL,
            "required": True,
            "value": {"id": chosen.key, "name": partner_type_name(chosen)} if chosen else None,
            "status": "valid" if chosen else "missing",
            # Only while it can still change: not once an account may exist (an attempt
            # whose answer was lost locks the type until it is settled).
            "options": [{"id": t.key, "name": partner_type_name(t)} for t in PARTNER_TYPES]
            if editable and status != "outcome_unknown" else [],
        }
    attached = conversion_uploads(state)
    rejected_uploads = state.get("rejected_uploads") or {}
    uploads = [
        {
            "key": key,
            "label": UPLOAD_SPECS[key].label,
            "file": {"name": attached[key]["name"], "size": attached[key]["size"]} if key in attached else None,
            "error": rejected_uploads.get(UPLOAD_SPECS[key].label),
        }
        for key in ctype.uploads
    ]
    return {
        # The request the card belongs to; for a partner without one, its form id.
        "requestId": conversion_ref(state),
        "linkedToRequest": bool(record.get("id")),
        "notice": notice,
        "conversionType": ctype.key,
        "conversionLabel": ctype.label,
        "serviceProvider": (conv.get("target_provider") or {}).get("name"),
        "serviceProviderField": provider_field,
        "companyTypeField": company_type_field,
        # Optional images sent with the account (Partners); only names and sizes, never content.
        "uploads": uploads,
        "uploadRules": {"extensions": list(UPLOAD_EXTENSIONS), "maxBytes": UPLOAD_MAX_BYTES} if uploads else None,
        "state": status,
        "editable": editable,
        # Inputs the card opens with (missing/invalid details plus the whole address);
        # "Edit" opens every editable field.
        "formFields": [k for k in form_fields if k not in _NOT_TYPED],
        "editableFields": [k for k in ctype.fields if k not in _NOT_TYPED] if editable else [],
        "fields": [
            {
                "key": key,
                "label": FIELD_LABELS[key],
                "value": draft.get(key),
                "required": key in ctype.required,
                "source": "user" if key in user_fields else ("request" if draft.get(key) else None),
                "status": "invalid" if FIELD_LABELS[key] in errors else "missing" if FIELD_LABELS[key] in missing else "valid",
                "error": errors.get(FIELD_LABELS[key]),
            }
            for key in ctype.fields
        ],
        "missingRequiredFields": missing,
        "invalidFields": errors,
        "creation": creation_status(state),
    }


def _conversion_facts(state: ContactUsChatState) -> dict[str, Any] | None:
    conv = conversion_of(state)
    unlinked = standalone(state)
    if not unlinked and (
        not conv.get("request_id") or conv.get("request_id") != (state.get("contact_request") or {}).get("id")
    ):
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
            for s in (conversion_steps(ctype, conv) if ctype else [])
        },
    }
    if unlinked:
        facts["without_contact_us_request"] = (
            "This partner is created on its own, like on BOOTOG's Create Partner screen: it is not linked to "
            "a Contact Us request, so no request is needed and no request status is changed. Do NOT ask for a "
            "request. If the user wants it linked to one, they can name the request."
        )
    if conv.get("account"):
        account = conv["account"]
        facts["existing_account"] = {
            "email": account.get("email"),
            "registered": account.get("exists"),
            "already_has_required_role": account.get("has_role"),
            "roles": account.get("roles"),
        }
    if (conv.get("account") or {}).get("verified") is False:
        facts["verification_failed"] = VERIFICATION_FAILED + " No account was created and the request was not changed."
    if conv.get("blocked") == "existing_role" and (conv.get("account") or {}).get("role_removed"):
        email = conv["account"].get("email")
        facts["blocked"] = (
            f"{email} has a removed (deleted) {ctype.role_name} role in BOOTOG. BOOTOG refuses to add that role "
            "again while the removed one is on record, so nothing was sent, no account was created and the request "
            "was not changed. This is NOT an active existing customer: do not say one exists."
        )
        facts["how_to_continue"] = (
            "An administrator has to restore the role in BOOTOG, or the user can enter a different email (the card "
            "is open on the Email field); a new email is checked again before anything is created."
        )
    elif conv.get("blocked") == "existing_role":
        record = state.get("contact_request") or {}
        already_completed = record.get("status") == 11
        found_later = bool((conv.get("account") or {}).get("found_after_failed_attempt"))
        email = (conv.get("account") or {}).get("email")
        facts["existing_account_warning"] = "Warning: " + existing_account_warning(ctype.label)
        facts["blocked"] = (
            f"The email {email} is already registered with the {ctype.role_name} role, so no new account was created. "
            + (
                "BOOTOG had answered an earlier creation attempt with an error, and a check afterwards found "
                "this role: it cannot be confirmed whether that attempt created it, so do NOT say it was "
                "created. Nothing will be retried. "
                if found_later
                else ""
            )
            + (
                "This Contact Us request is already marked CRM Completed."
                if already_completed
                else "That does NOT mean this Contact Us request was converted. The agent did not change the "
                "request's status; it stays as it was in BOOTOG"
                + (f" ({CONTACT_US_STATUS_LABELS[record['status']]})." if record.get("status") in CONTACT_US_STATUS_LABELS else ".")
            )
            + " The collected details are kept."
        )
        facts["how_to_continue"] = (
            "Enter a different email in the card (it is open on the Email field) or type the new email in the "
            "chat; the new email is checked again before anything is created. Other options: convert to a "
            "different account type, or cancel."
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
    if question == "ready_to_create":
        draft = state.get("customer_payload") or {}
        reason = (state.get("pending_question") or {}).get("reason")
        facts["ready_for_creation"] = {
            "type": ctype.label,
            "details": {FIELD_LABELS[k]: v for k, v in draft.items() if v},
            "why_waiting": {
                "review_requested": "the user asked to review the details first",
                "after_failure": "the last attempt failed; the details can be corrected first",
            }.get(reason, "the details were changed after they were complete"),
            "next": "Ask the user to check the details and say 'create' (or press Create customer) to create the account.",
        }
    if question == "confirm_conversion":
        draft = state.get("customer_payload") or {}
        record = state.get("contact_request") or {}
        facts["awaiting_confirmation"] = {
            "will_create": ctype.label,
            "from_request": (
                {"id": record.get("id"), "name": full_name(record), "email": record.get("emailId")}
                if record else "none — not linked to a Contact Us request"
            ),
            "service_provider": (conv.get("target_provider") or {}).get("name"),
            "service_categories": [s["name"] for s in conv.get("subcategories") or []],
            "carried_over_from_previous_conversion": conv.get("carried_over") or {},
            "role": conv.get("role_name"),
            "details": {FIELD_LABELS[k]: v for k, v in draft.items() if v},
            "coordinates": (
                ("from the request" if record else "from the address picked") if draft.get("latitude")
                else "not available — left empty"
            ),
            "then": [STEP_LABELS[s] for s in conversion_steps(ctype, conv)[1:]],
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
            "card": "The details card has a 'Select Service Provider' dropdown with every provider; the user can "
            "pick it there (together with any missing details) or reply with a number or a provider name.",
        }
    if question == "choose_subcategory":
        facts["service_category_choice"] = {
            **(state.get("pending_question") or {}),
            "options": [{"position": i + 1, "name": s["name"]} for i, s in enumerate(conv.get("subcategory_candidates") or [])],
        }
    if conversion_finished(state):
        facts["result"] = {
            "completed": True,
            "verified": f"BOOTOG shows the account with the {ctype.role_name} role"
            + (", linked to the service provider; no Contact Us request was involved, so none was updated."
               if unlinked else ", and the request read back as CRM Completed after the update."),
            "request_status": None if unlinked else "CRM Completed",
            "user_id": conv.get("user_id"),
            "company_id": conv.get("created_vendor_id") if ctype and ctype.skip_tenant else None,
        }
        if state.get("is_complete"):  # finished in this turn
            facts["result"]["follow_up"] = _follow_up(state)
    return facts


# What the agent offers after a verified conversion. More next actions are added here.
NEXT_ACTIONS = ("convert_another",)


def _follow_up(state: ContactUsChatState) -> str:
    noun = (state.get("agent_name") or "Contact Us Agent").removesuffix(" Agent")
    return f"Would you like to convert another {noun} request?"


def _failure_advice(failed: dict[str, Any]) -> str:
    if failed.get("code") == "CSRF_INVALID":
        return (
            "Bootog's gateway refused the write (anti-forgery check). Nothing was created. An administrator "
            "must fix the agent's Bootog connection; then 'retry' sends it again."
        )
    if failed.get("kind") == "created_with_error":
        return (
            "BOOTOG answered the creation with an error, but the email had no account with this role just "
            "before and has it now: the account was most likely created by this attempt (BOOTOG does not undo a "
            "failed creation), so it is NOT an existing customer and must not be called one. Say that account "
            "creation may be incomplete and that CRM completion is unverified: the request was not marked CRM "
            "Completed. Do not suggest another email (that would create a second account). 'retry' verifies "
            "this account and then marks the request CRM Completed, without creating anything again; the "
            "BOOTOG team may need to check this account."
        )
    if failed.get("step") == "update_request_status":
        return (
            "The account was created and verified, but the request could not be confirmed as CRM Completed, so "
            "the conversion is NOT complete — never call it complete. 'retry' applies the status again and "
            "reads it back, without creating anything."
        )
    if failed.get("code") == "EMAIL_EXISTS":
        return (
            "BOOTOG rejected the email as already registered, so no account was created and the request was "
            "not changed. Enter a different email (the card opens on the Email field, or type it in the chat); "
            "it is checked again before anything is sent. The same email is not resent."
        )
    if failed.get("reconciled") == "absent":
        return (
            "BOOTOG failed internally while creating the account (a server problem, not the details). "
            "A check right afterwards found no account with this role for the email, so nothing usable was "
            "created. 'retry' checks again before sending anything."
        )
    if failed.get("reconciled") == "unverified":
        return (
            "BOOTOG failed internally, and the follow-up check could not be completed either, so it is unknown "
            "whether part of the account exists. Nothing is sent again until a check succeeds ('retry' checks first)."
        )
    if failed.get("code") == "SERVER_FAULT":
        return (
            "BOOTOG failed internally while creating the account; this is a server problem, not a "
            "problem with the details. Part of the account (login and role) may already exist. "
            "'retry' checks BOOTOG first and never creates a second account; if it still fails, "
            "the BOOTOG team should look at this email's account."
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
        if state.get("intent") == "list_requests":
            view = request_filters_view(state)
            facts["requests"]["active_filters"] = _describe_filters(state, view)
            facts["requests"]["page"] = {"page": view["page"], "of": view["pages"]}
            if not state.get("contact_requests_total"):
                facts["requests"]["empty"] = (
                    "No requests matched the selected filters. Say exactly that, name the active filters, and offer "
                    "to broaden them (another period, status or company). Do NOT drop any filter yourself."
                )
    history = state.get("history")
    if history:
        facts["history"] = {
            "request": history.get("name"),
            "total_entries": history.get("total"),
            "error": history.get("error"),
            "cut_off": bool(history.get("truncated")),
            # Newest first, as BOOTOG orders them; the card shows every entry.
            "latest": [
                {k: v for k, v in e.items() if v not in (None, "", False) and k not in ("id", "statusId")}
                for e in (history.get("entries") or [])[:_FACTS_LIST_LIMIT]
            ],
        }
    status_change = status_change_facts(state)
    if status_change:
        facts["status_change"] = status_change
    if state.get("filter_problems"):
        facts["filters_not_applied"] = {
            "could_not_apply": state["filter_problems"],
            "nothing_listed": True,
            "filters_kept": _describe_filters(state, request_filters_view(state)),
            "valid_statuses": list(CONTACT_US_STATUS_LABELS.values()),
            "next": "Say which filter could not be applied and why, and ask for a valid one (for a company that "
            "matched several, list the numbered options). Nothing was listed, so no results may be described.",
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
    if selected or standalone(state):
        if selected:
            facts["selected_request"] = request_summary(selected)
        if converting or state.get("is_complete") or intent in ("show_request_details", "provide_information", *CONVERSION_OPERATIONS):
            view = customer_view(state) or {}
            facts["form"] = {
                "converting": converting,
                "form": view.get("conversionLabel"),
                "operation_state": view.get("state"),
                "fields": {f["label"]: f["value"] for f in view.get("fields", [])},
                "missing_required": view.get("missingRequiredFields", []),
                "invalid": view.get("invalidFields", {}),
                "provided_this_turn": [
                    FIELD_LABELS[k] for k in state.get("turn_field_updates") or {}
                    if FIELD_LABELS[k] not in (state.get("rejected_field_updates") or {})
                ],
                "submitted_from_form": bool(state.get("form_submitted")),
                "the_card_shows_a_form": bool(view.get("formFields")),
            }
            if view.get("uploads"):
                facts["form"]["images"] = {u["label"]: (u["file"] or {}).get("name") for u in view["uploads"]}
        if state.get("rejected_uploads"):
            facts["rejected_images"] = {"not_accepted": state["rejected_uploads"], "previous_image_kept": True}
        if state.get("rejected_field_updates"):
            facts["rejected_changes"] = {
                "not_accepted": state["rejected_field_updates"],
                "previous_values_kept": True,
                "account_not_created_this_turn": True,
            }
        if state.get("post_creation_edit") or (state.get("edit_request") and account_exists(state)):
            fields = state.get("post_creation_edit") or [
                FIELD_LABELS[k] for k in (state.get("edit_request") or {}).get("fields") or []
            ]
            facts["post_creation_edit"] = {
                "fields": fields,
                "saved": False,
                "why": "The account already exists. This workflow has no BOOTOG API to update an existing "
                "account, so the change was not saved anywhere; it has to be made in BOOTOG directly.",
            }
        if state.get("edit_request") and not account_exists(state):
            facts["edit_request"] = {
                "fields_offered": [FIELD_LABELS[k] for k in state["edit_request"].get("fields") or []],
                "next": "Ask which of these to change and the new value (the card shows them as a form).",
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

