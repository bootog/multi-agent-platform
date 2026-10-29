"""Business rules of the Frontend Order Enquiry Agent. Pure functions only: no API
calls, no LLM and no database, so the same input always gives the same answer and every
rule is easy to test.

Sections:
1. Dates       — "today", "last 7 days", "Oct 1 to Oct 10" -> a validated DateRange.
2. Enquiries   — validate records, converted (orderAction 6) or not, enquiry context,
                 follow-up detection from the Contact Log history.
3. Replies     — the chat reply text and the `result` payload sent to the browser."""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.agents.frontorder_enquiry.state import FrontOrderEnquiryAgentState
from app.core.config import get_settings
from app.integrations.bootog.order_enquiry import ORDER_ACTION_CONVERTED_TO_ORDER

# =====================================================================================
# 1. Dates
# The LLM only says WHICH kind of period the user asked for; the actual dates are
# computed and validated here in the business timezone. Anything that cannot be turned
# into a clear, valid range becomes a clarification question instead of a guess.
# =====================================================================================


@dataclass(frozen=True)
class DateRange:
    start: date  # inclusive, business-timezone calendar day
    end: date  # inclusive, business-timezone calendar day

    def utc_bounds(self, tz: ZoneInfo) -> tuple[datetime, datetime]:
        """[start of first day, start of the day after the last day) as aware datetimes.
        The API layer converts them to UTC."""
        start = datetime.combine(self.start, time.min, tzinfo=tz)
        end = datetime.combine(self.end + timedelta(days=1), time.min, tzinfo=tz)
        return start, end


class DateRangeError(ValueError):
    """The requested period is unclear or invalid. The message is a question/explanation
    that can be shown to the user as-is."""


def today_in(tz: ZoneInfo) -> date:
    return datetime.now(tz).date()


def parse_iso_date(value: str | None, what: str) -> date:
    if not value:
        raise DateRangeError(f"Which {what} date do you mean? Please give it as a full date, e.g. 2026-10-01.")
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise DateRangeError(f"I couldn't read the {what} date '{value}'. Please give it as YYYY-MM-DD.") from None


def resolve_period(
    period: str,
    today: date,
    max_range_days: int,
    last_n_days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> DateRange | None:
    """Turn the interpreted period into a validated DateRange.


    period:
      - "none" / "all" -> None (no date filter). The graph treats "none" (no dates in the
                          message) as "keep the conversation's dates" before calling this.
      - "today"       -> today
      - "yesterday"   -> yesterday
      - "last_n_days" -> the last N calendar days INCLUDING today (last 7 days = today and the 6 before)
      - "date_range"  -> start_date..end_date (inclusive); a single day when end_date is missing
    """
    if period in ("none", "all"):
        return None
    if period == "today":
        return DateRange(today, today)
    if period == "yesterday":
        day = today - timedelta(days=1)
        return DateRange(day, day)
    if period == "last_n_days":
        if not last_n_days or last_n_days < 1:
            raise DateRangeError("How many days back should I look?")
        if last_n_days > max_range_days:
            raise DateRangeError(f"Please ask for {max_range_days} days or fewer.")
        return DateRange(today - timedelta(days=last_n_days - 1), today)
    if period == "date_range":
        start = parse_iso_date(start_date, "start")
        end = parse_iso_date(end_date, "end") if end_date else start
        return validate_range(DateRange(start, end), today, max_range_days)
    raise DateRangeError("Which dates should I look at? For example: today, the last 7 days, or October 1 to October 10.")


def validate_range(requested: DateRange, today: date, max_range_days: int) -> DateRange:
    if requested.start > requested.end:
        raise DateRangeError(
            f"The start date {requested.start.isoformat()} is after the end date {requested.end.isoformat()}. "
            "Which range did you mean?"
        )
    if requested.start > today:
        raise DateRangeError(
            f"{requested.start.isoformat()} is in the future, so there can't be enquiries yet. Which dates did you mean?"
        )
    if (requested.end - requested.start).days + 1 > max_range_days:
        raise DateRangeError(f"That range is longer than {max_range_days} days. Please choose a shorter range.")
    # An end date in the future is harmless (no enquiries exist yet); cap it at today so
    # the summary reports the range that was actually searched.
    return DateRange(requested.start, min(requested.end, today))


# =====================================================================================
# 2. Enquiries
# =====================================================================================

# Request actions (see understanding.py) that are ABOUT converted enquiries ("show
# converted enquiries", "how many were converted"). Every other action — new
# enquiries, follow-up lists, sending follow-ups — is about enquiries that are still
# open, so converted ones are left out of those results (but never deleted or hidden
# from a converted-enquiry request).
CONVERTED_ACTIONS = {"list_converted"}
# Actions that need each enquiry's Contact Log history.
FOLLOW_UP_ACTIONS = {"list_followed_up", "list_not_followed_up"}


# --- validation -----------------------------------------------------------


def enquiry_id(record: Any) -> str | None:
    """The enquiry's id as a string, or None when the record has no usable id."""
    if not isinstance(record, dict):
        return None
    value = record.get("id")
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_enquiries(records: list[Any]) -> tuple[list[dict[str, Any]], int, int]:
    """Keep dict records that carry an id, first occurrence per id.
    Returns (valid, malformed count, duplicate count)."""
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    malformed = duplicates = 0
    for record in records:
        record_id = enquiry_id(record)
        if record_id is None:
            malformed += 1
        elif record_id in seen:
            duplicates += 1
        else:
            seen.add(record_id)
            valid.append(record)
    return valid, malformed, duplicates


# --- orderAction classification and intent selection ------------------------
# Three separate steps, so each stays simple:
#   1. classify_enquiries          -> which records are converted / not converted
#   2. filter_enquiries_for_intent -> which of those the user asked about
#   3. later workflow stages act only on the selected records


def parse_order_action(value: Any) -> int | None:
    """orderAction as an int, or None when missing/unreadable. Accepts 6, 6.0 and "6"."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


@dataclass
class ClassifiedEnquiries:
    """Validated records grouped by orderAction. The records are not modified."""

    not_converted: list[dict[str, Any]] = field(default_factory=list)
    converted: list[dict[str, Any]] = field(default_factory=list)
    # orderAction missing/null/unreadable: we cannot tell whether the enquiry was
    # converted, so no request selects it; its id is reported instead.
    unknown: list[dict[str, Any]] = field(default_factory=list)


def classify_enquiries(records: list[dict[str, Any]]) -> ClassifiedEnquiries:
    """Group records by orderAction: 6 = converted to order, any other number = not
    converted, missing/unreadable = unknown."""
    classified = ClassifiedEnquiries()
    for record in records:
        action = parse_order_action(record.get("orderAction"))
        if action is None:
            classified.unknown.append(record)
        elif action == ORDER_ACTION_CONVERTED_TO_ORDER:
            classified.converted.append(record)
        else:
            classified.not_converted.append(record)
    return classified


def wants_converted(action: str | None) -> bool:
    """True when the request is about converted enquiries."""
    return action in CONVERTED_ACTIONS


def filter_enquiries_for_intent(action: str | None, classified: ClassifiedEnquiries) -> list[dict[str, Any]]:
    """The records the user asked about: converted ones for a converted-enquiry
    request, otherwise only those not converted yet."""
    return classified.converted if wants_converted(action) else classified.not_converted


# --- enquiry context ------------------------------------------------------
# Fields read from an OrderEnquery record. None of them is assumed present; a
# missing field is simply None in the context. The full record stays available in
# state (selected_enquiries) for later stages that need provider/fee/industry data.

CONTEXT_FIELDS: tuple[str, ...] = (
    "orderId",
    "jobId",
    "providerId",
    "providerName",
    "industryName",
    "serviceCategoryName",
    "assignedToUserName",
    "assignmentOrigin",
    "customerId",
    "customerFirstName",
    "customerLastName",
    "customerEmail",
    "customerPhone",
    "isSmsAgreed",
    "status",
    "mainServiceTypeId",
    "additionalServiceIds",
    "address",
    "city",
    "state",
    "zipCode",
    "county",
    "date",
    "fromTime",
    "toTime",
    "createdDateTime",
    "modifiedDateTime",
)


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def main_service_name(record: dict[str, Any]) -> str | None:
    """Name of the main service: the fee row flagged isMainService (the same lookup the
    .NET OrderEnqueryService uses for its emails)."""
    fees = record.get("fees")
    for fee in fees if isinstance(fees, list) else []:
        if isinstance(fee, dict) and fee.get("isMainService") is True:
            return _clean(fee.get("serviceTypeName"))
    return None


def build_enquiry_context(record: dict[str, Any]) -> dict[str, Any]:
    """The enquiry context handed to the next workflow stage."""
    context: dict[str, Any] = {"enquiryId": enquiry_id(record)}
    for name in CONTEXT_FIELDS:
        context[name] = _clean(record.get(name))
    context["orderAction"] = parse_order_action(record.get("orderAction"))
    context["mainServiceName"] = main_service_name(record)
    name = " ".join(p for p in (context["customerFirstName"], context["customerLastName"]) if p)
    context["customerName"] = name or None
    return context


def summarize_enquiry(context: dict[str, Any]) -> dict[str, Any]:
    """The subset returned in the run result for the chat to display. No email/phone.
    `status` and `orderAction` stay numeric: the Angular app already maps them to labels."""
    return {
        "enquiryId": context.get("enquiryId"),
        "jobId": context.get("jobId"),
        "customerName": context.get("customerName"),
        "createdDateTime": context.get("createdDateTime"),
        "mainService": context.get("mainServiceName"),
        "serviceCategory": context.get("serviceCategoryName"),
        "industry": context.get("industryName"),
        "providerName": context.get("providerName"),
        "assignedTo": context.get("assignedToUserName"),
        "status": context.get("status"),
        "orderAction": context.get("orderAction"),
        "city": context.get("city"),
        "state": context.get("state"),
        # Only set when the request asked about follow-up (None = not checked).
        "followedUp": context.get("followedUp"),
        "lastContactedAt": context.get("lastContactedAt"),
    }


# --- follow-up detection (OrderEnqueryTrackingHistory) ----------------------
# Verified in OrderEnqueryService.cs: the history table gets three kinds of rows.
#   1. the creation row ("Order Inquiry Initialize"),
#   2. ChangeStatus rows (status change only),
#   3. Contact Log rows written by OrderEnqueryLogUpdate (a person logged a call,
#      SMS, email or note).
# Only kind 3 ever fills the fields below, so a row with any of them is a real contact.
# `followUpNeeded` / `dateOfFollowUp` only mean a follow-up is *planned*, so they are
# deliberately not used.
CONTACT_LOG_FIELDS: tuple[str, ...] = ("userName", "contactName", "contactNumber", "emailTo", "smsTo", "note")


def is_contact_log(row: dict[str, Any]) -> bool:
    return any(isinstance(row.get(f), str) and row[f].strip() for f in CONTACT_LOG_FIELDS)


def latest_contact(history: list[dict[str, Any]]) -> str | None:
    """createdDateTime of the newest Contact Log row, or None when nobody has contacted
    the customer about this enquiry yet."""
    dates = [str(row.get("createdDateTime")) for row in history if is_contact_log(row) and row.get("createdDateTime")]
    return max(dates) if dates else None


# =====================================================================================
# 3. Replies
# The chat reply text and the `result` payload of a run, built from the final graph
# state. `result` keeps every field the Angular results card reads (`eligibleEnquiries`,
# `eligibleCount`, `request`, ...) plus the provider-first fields (`view`,
# `providerSummary`, `providerCandidates`, `pagination`, `scope`). Counts are always the
# API's own numbers; a count that is not complete is flagged, never shown as a total.
# =====================================================================================

NO_PROVIDER_LABEL = "No provider assigned"


def _noun(count: int) -> str:
    return f"enquir{'y' if count == 1 else 'ies'}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _period_text(scope: dict[str, Any]) -> str:
    # Dates always refer to when the enquiry was CREATED: Bootog records no separate
    # conversion date (conversion only updates modifiedDateTime, as any edit does).
    date_from, date_to = scope.get("date_from"), scope.get("date_to")
    if not date_from:
        return "(all creation dates)"
    return f"created on {date_from}" if date_from == date_to else f"created from {date_from} to {date_to}"


def _kind_text(action: str | None) -> str:
    return "converted to orders" if wants_converted(action) else "not converted to orders"


def _total_pages(total: int, page_size: int) -> int:
    return max(1, -(-total // page_size)) if page_size > 0 else 1


# --- provider detail -----------------------------------------------------------


def matching_count(state: FrontOrderEnquiryAgentState) -> tuple[int, bool]:
    """(number of enquiries matching the request, whether that number is complete).

    - List / converted requests: the query is scoped to the same orderAction class, so
      the server's count is the exact total, whichever page is shown.
    - Follow-up requests: only the records read were checked; if the page cap stopped
      the read, the count is incomplete and reported as such."""
    shown = len(state.get("enquiry_contexts", []))
    scope = state.get("scope") or {}
    if state.get("action") in FOLLOW_UP_ACTIONS:
        return shown, not state.get("enquiries_truncated")
    if not state.get("enquiries_truncated") and (scope.get("page") or 1) == 1:
        return shown, True
    return state.get("enquiries_total") or shown, True


def read_range(records: list[Any]) -> tuple[str | None, str | None]:
    """Creation dates (YYYY-MM-DD) of the oldest and newest record read, so a partial read
    can say which dates it covered ("the latest 250 of 1662" = created from .. to ..)."""
    days = [str(r["createdDateTime"])[:10] for r in records if isinstance(r, dict) and r.get("createdDateTime")]
    return (min(days), max(days)) if days else (None, None)


def _detail_pagination(state: FrontOrderEnquiryAgentState, page_size: int) -> dict[str, Any] | None:
    scope = state.get("scope") or {}
    if state.get("action") in FOLLOW_UP_ACTIONS:
        return None  # every page (up to the cap) was read; there are no pages to move through
    total = state.get("enquiries_total") or 0
    page = scope.get("page") or 1
    return {
        "itemType": "enquiries",
        "page": page,
        "pageSize": page_size,
        "totalItems": total,
        "totalPages": _total_pages(total, page_size),
        "hasMore": bool(state.get("enquiries_truncated")),
    }


def _detail_reply(state: FrontOrderEnquiryAgentState, page_size: int) -> str:
    scope = state.get("scope") or {}
    action = state.get("action") or ""
    shown = len(state.get("enquiry_contexts", []))
    count, complete = matching_count(state)
    provider = scope.get("provider_name") or "the selected provider"
    kind = {"list_followed_up": "followed-up ", "list_not_followed_up": "not-yet-followed-up "}.get(action, "")
    message = f"{provider}: {count} {kind}{_noun(count)} {_kind_text(action)} {_period_text(scope)}"

    if action in FOLLOW_UP_ACTIONS:
        if complete:
            return message + "."
        read = len(state.get("raw_enquiries", []))
        oldest, newest = read_range(state.get("raw_enquiries", []))
        covered = f" (created {oldest} to {newest})" if oldest else ""
        return (message + f" among the latest {read} of {state.get('enquiries_total')} read{covered}, so there may be "
                "more. Ask for a shorter date range to check all of them.")
    pagination = _detail_pagination(state, page_size) or {}
    if pagination.get("totalPages", 1) <= 1:
        return message + "."
    page = pagination["page"]
    first = (page - 1) * page_size + 1
    if shown == 0:
        return message + f". Page {page} is past the last page ({pagination['totalPages']})."
    more = ' Say "next page" for more.' if pagination["hasMore"] else ""
    return message + f". Showing {first}–{first + shown - 1} (page {page} of {pagination['totalPages']}).{more}"


# --- provider summary ------------------------------------------------------------


def summary_counts(state: FrontOrderEnquiryAgentState) -> tuple[int, bool]:
    """(enquiries across the providers shown, whether that is the total of ALL providers).
    It is the overall total only when every provider fitted on the first page."""
    providers = state.get("provider_summary") or []
    shown_total = sum(p["count"] for p in providers)
    page = (state.get("scope") or {}).get("page") or 1
    complete = page == 1 and len(providers) >= (state.get("providers_total") or 0)
    return shown_total, complete


def _summary_pagination(state: FrontOrderEnquiryAgentState, page_size: int) -> dict[str, Any]:
    total = state.get("providers_total") or 0
    page = (state.get("scope") or {}).get("page") or 1
    return {
        "itemType": "providers",
        "page": page,
        "pageSize": page_size,
        "totalItems": total,
        "totalPages": _total_pages(total, page_size),
        "hasMore": page * page_size < total,
    }


def _summary_reply(state: FrontOrderEnquiryAgentState, page_size: int) -> str:
    scope = state.get("scope") or {}
    action = scope.get("action")
    providers = state.get("provider_summary") or []
    period = _period_text(scope)
    total_providers = state.get("providers_total") or 0
    pagination = _summary_pagination(state, page_size)
    if total_providers == 0:
        return f"No order enquiries {_kind_text(action)} {period}."
    if not providers:
        return f"Page {pagination['page']} is past the last page of providers ({pagination['totalPages']})."

    shown_total, complete = summary_counts(state)
    first = (pagination["page"] - 1) * page_size + 1
    if complete:
        header = (
            f"{shown_total} {_noun(shown_total)} {_kind_text(action)} {period}, "
            f"from {_plural(total_providers, 'provider')} (most enquiries first):"
        )
    else:
        header = (
            f"Enquiries {_kind_text(action)} {period}: providers {first}–{first + len(providers) - 1} "
            f"of {total_providers} (most enquiries first). These providers have {shown_total} {_noun(shown_total)}; "
            "the overall total needs every page of providers, so it is not shown here."
        )
    lines = [header]
    names = [p["providerName"] for p in providers]
    for number, provider in enumerate(providers, start=first):
        label = provider["providerName"] or NO_PROVIDER_LABEL
        if provider["providerName"] and names.count(provider["providerName"]) > 1:
            # Different providers with the same name: show the end of the id to tell them apart.
            label += f" (id …{provider['providerId'][-4:]})"
        lines.append(f"{number}. {label} — {provider['count']}")
    hints = ["Reply with a number or a provider name to see its enquiries."]
    if pagination["hasMore"]:
        hints.append('Say "next page" for more providers.')
    if action in FOLLOW_UP_ACTIONS:
        hints.insert(0, "Follow-up status is checked per provider.")
    lines.append(" ".join(hints))
    return "\n".join(lines)


# --- reply and result ------------------------------------------------------------


def build_reply(state: FrontOrderEnquiryAgentState) -> str:
    settings = get_settings()
    status = state.get("execution_status")
    if status in ("failed", "blocked"):
        return state.get("error") or "The Frontend Order Enquiry Agent stopped."
    if state.get("clarification"):
        return state["clarification"]
    if state.get("view") == "provider_summary":
        return _summary_reply(state, settings.frontorder_enquiry_summary_page_size)
    return _detail_reply(state, settings.frontorder_enquiry_page_size)


def build_result(state: FrontOrderEnquiryAgentState) -> dict[str, Any]:
    """The `result` of the final run event. Field names before `view` are kept stable
    because the Angular results card reads them: `eligibleEnquiries` / `eligibleCount`
    hold the enquiries this request selected — converted ones when `selection` is
    "converted" — and are empty for a provider summary."""
    settings = get_settings()
    scope = state.get("scope") or {}
    view = state.get("view") or "none"
    failed = state.get("execution_status") in ("failed", "blocked")
    detail = view == "provider_detail" and not failed and not state.get("clarification")
    summary = view == "provider_summary" and not failed and not state.get("clarification")
    matched_count, count_complete = matching_count(state) if detail else (0, True)
    read_from, read_to = read_range(state.get("raw_enquiries", [])) if detail else (None, None)
    summary_total, summary_complete = summary_counts(state) if summary else (None, None)
    action = state.get("action") or scope.get("action")
    if detail:
        pagination = _detail_pagination(state, settings.frontorder_enquiry_page_size)
    elif summary:
        pagination = _summary_pagination(state, settings.frontorder_enquiry_summary_page_size)
    else:
        pagination = None
    return {
        "request": {
            "action": action,
            "summary": state.get("request_summary"),
            "dateFrom": scope.get("date_from"),
            "dateTo": scope.get("date_to"),
            "targetProviderId": scope.get("provider_id"),
            "providerName": scope.get("provider_name"),
        },
        "clarificationQuestion": state.get("clarification"),
        "selection": "converted" if wants_converted(action) else "not_converted",
        "matchedCount": matched_count,
        "countComplete": count_complete,
        "eligibleCount": len(state.get("enquiry_contexts", [])),
        "totalCount": state.get("enquiries_total") if detail else None,
        "retrievedCount": len(state.get("raw_enquiries", [])),
        # Creation dates of the oldest / newest record read (provider detail only).
        "readFrom": read_from,
        "readTo": read_to,
        "truncated": bool(state.get("enquiries_truncated")) if detail else False,
        "malformedCount": state.get("malformed_count", 0),
        "duplicateCount": state.get("duplicate_count", 0),
        "convertedEnquiryIds": state.get("converted_enquiry_ids", []),
        "invalidOrderActionIds": state.get("invalid_order_action_ids", []),
        "eligibleEnquiries": [summarize_enquiry(c) for c in state.get("enquiry_contexts", [])],
        "failedStep": state.get("current_step") if failed else None,
        # --- provider-first fields ---
        "view": "clarification" if state.get("clarification") and not failed else view,
        "scope": {
            "action": scope.get("action"),
            "dateFrom": scope.get("date_from"),
            "dateTo": scope.get("date_to"),
            "providerId": scope.get("provider_id"),
            "providerName": scope.get("provider_name"),
            "page": scope.get("page"),
        },
        "providerSummary": [
            {"providerId": p["providerId"], "providerName": p["providerName"] or NO_PROVIDER_LABEL, "count": p["count"]}
            for p in (state.get("provider_summary") or [])
        ]
        if summary
        else [],
        # Enquiries across the providers in providerSummary; summaryCountComplete says
        # whether that is also the total across all providers.
        "summaryEnquiryCount": summary_total,
        "summaryCountComplete": summary_complete,
        "providerCandidates": state.get("provider_candidates") or [],
        "pagination": pagination,
    }


def answer_facts(reply: str, result: dict[str, Any]) -> dict[str, Any]:
    """What a later chat question about this answer may use ("what date range was that?").
    Counts, dates, scope and provider names only: no customer names or contact details."""
    facts = {"reply": reply}
    for name in ("view", "scope", "matchedCount", "countComplete", "totalCount", "retrievedCount", "truncated",
                 "readFrom", "readTo", "summaryEnquiryCount", "summaryCountComplete", "pagination"):
        facts[name] = result.get(name)
    facts["providerSummary"] = result.get("providerSummary") or []
    return facts

