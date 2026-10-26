"""The chat reply text and the `result` payload of a run. Pure functions of the final
graph state, so they are easy to test and never call an API.

`result` keeps every field the Angular results card already reads (`eligibleEnquiries`,
`eligibleCount`, `request`, ...) and adds the provider-first fields (`view`,
`providerSummary`, `providerCandidates`, `pagination`, `scope`). Counts are always the
API's own numbers; a count that is not complete is flagged, never presented as a total."""

from typing import Any

from app.agents.frontorder_enquiry.config import get_frontorder_enquiry_settings
from app.agents.frontorder_enquiry.schemas import FOLLOW_UP_ACTIONS, summarize_enquiry, wants_converted
from app.agents.frontorder_enquiry.state import FrontOrderEnquiryAgentState

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
        return message + f" among the latest {read} of {state.get('enquiries_total')} read, so there may be more."
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
    settings = get_frontorder_enquiry_settings()
    status = state.get("execution_status")
    if status in ("failed", "blocked"):
        return state.get("error") or "The Frontend Order Enquiry Agent stopped."
    if state.get("clarification"):
        return state["clarification"]
    if state.get("view") == "provider_summary":
        return _summary_reply(state, settings.summary_page_size)
    return _detail_reply(state, settings.page_size)


def build_result(state: FrontOrderEnquiryAgentState) -> dict[str, Any]:
    """The `result` of the final run event. Field names before `view` are kept stable
    because the Angular results card reads them: `eligibleEnquiries` / `eligibleCount`
    hold the enquiries this request selected — converted ones when `selection` is
    "converted" — and are empty for a provider summary."""
    settings = get_frontorder_enquiry_settings()
    scope = state.get("scope") or {}
    view = state.get("view") or "none"
    failed = state.get("execution_status") in ("failed", "blocked")
    detail = view == "provider_detail" and not failed and not state.get("clarification")
    summary = view == "provider_summary" and not failed and not state.get("clarification")
    matched_count, count_complete = matching_count(state) if detail else (0, True)
    summary_total, summary_complete = summary_counts(state) if summary else (None, None)
    action = state.get("action") or scope.get("action")
    if detail:
        pagination = _detail_pagination(state, settings.page_size)
    elif summary:
        pagination = _summary_pagination(state, settings.summary_page_size)
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
