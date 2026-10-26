"""Frontend Order Enquiry graph nodes.

Every node except `respond` is wrapped by `tracked_step`, which publishes running /
completed / failed / blocked events on the run channel, logs the step and records it
in the run monitor. Logs carry ids, counts and status only — never tokens or customer
details."""

import asyncio
import functools
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.frontorder_enquiry.config import get_frontorder_enquiry_settings
from app.agents.frontorder_enquiry.dates import DateRange, DateRangeError, resolve_period, today_in
from app.agents.frontorder_enquiry.monitoring import RunMonitor
from app.agents.frontorder_enquiry.responses import build_reply, build_result
from app.agents.frontorder_enquiry.schemas import (
    FOLLOW_UP_ACTIONS,
    build_enquiry_context,
    classify_enquiries,
    enquiry_id,
    filter_enquiries_for_intent,
    is_contact_log,
    latest_contact,
    validate_enquiries,
    wants_converted,
)
from app.agents.frontorder_enquiry.state import EnquiryScope, FrontOrderEnquiryAgentState
from app.agents.frontorder_enquiry.tools import (
    get_enquiry_contact_history,
    get_order_enquiries,
    get_provider_enquiry_summary,
    resolve_provider_by_name,
    verify_provider,
)
from app.agents.frontorder_enquiry.understanding import DEFAULT_REQUEST, understand_request
from app.agents.runtime import RunEmitter, utc_now
from app.core.logging import get_logger, log_step
from app.integrations.bootog import BootogApiError
from app.llm import LlmNotConfiguredError

AGENT = "FRONTORDER_ENQUIRY_AGENT"
logger = get_logger(__name__)

DEFAULT_ACTION = "list_enquiries"
# History reads run in parallel, a few at a time, so a long list stays quick without
# flooding the Bootog API.
HISTORY_CONCURRENCY = 5

HELP_MESSAGE = (
    "I can count and list order enquiries per provider, for example: \"How many enquiries "
    'came in today?", "Show me enquiries for <provider name>", "Show converted enquiries this '
    'month" or "Which enquiries from October 1 to October 10 have not been followed up?".'
)
FOLLOW_UP_EMAIL_UNAVAILABLE = (
    "Sending follow-up emails is not available yet: no email-sending service has been connected "
    "to this agent, so nothing was sent. You can ask me to list the enquiries that have not been "
    "followed up yet instead."
)


@dataclass
class StepResult:
    message: str
    update: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] | None = None


class StepFailed(Exception):
    """Expected failure with a user-safe message."""


class StepBlocked(Exception):
    """The step cannot run yet (e.g. an API is not configured). Not a success."""


NodeFn = Callable[[FrontOrderEnquiryAgentState, RunnableConfig], Awaitable[StepResult]]


def _monitor(config: RunnableConfig) -> RunMonitor | None:
    return config["configurable"].get("monitor")


def tracked_step(step_id: str, running_message: str, failure_message: str):
    def decorator(fn: NodeFn):
        @functools.wraps(fn)
        async def node(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> dict[str, Any]:
            emitter: RunEmitter = config["configurable"]["emitter"]
            monitor = _monitor(config)
            run_id = state["run_id"]
            await emitter.step(step_id, "running", running_message)
            log_step(logger, AGENT, run_id, step_id, "running")
            started = time.perf_counter()

            def elapsed() -> int:
                return int((time.perf_counter() - started) * 1000)

            async def stop(status: str, message: str) -> dict[str, Any]:
                await emitter.step(step_id, status, message, duration_ms=elapsed())
                log_step(logger, AGENT, run_id, step_id, status, elapsed())
                if monitor:
                    monitor.step(step_id, status, elapsed(), message)
                return {"current_step": step_id, "execution_status": status, "error": message}

            try:
                result = await fn(state, config)
            except StepBlocked as exc:
                return await stop("blocked", str(exc))
            except StepFailed as exc:
                return await stop("failed", str(exc))
            except BootogApiError as exc:
                return await stop("failed", f"{failure_message} {exc.message}")
            except Exception as exc:  # unexpected: keep details in logs only
                logger.exception("[%s] run=%s step=%s unexpected %s", AGENT, run_id, step_id, type(exc).__name__)
                return await stop("failed", failure_message)

            await emitter.step(step_id, "completed", result.message, data=result.data, duration_ms=elapsed())
            log_step(logger, AGENT, run_id, step_id, "completed", elapsed())
            if monitor:
                monitor.step(step_id, "completed", elapsed())
            return {**result.update, "current_step": step_id}

        return node

    return decorator


def _enquiries(count: int) -> str:
    return f"{count} order enquir{'y' if count == 1 else 'ies'}"


def _clarify(question: str, **update: Any) -> StepResult:
    """Stop before any enquiry API call and ask the user instead of guessing."""
    return StepResult("Need more information", {"clarification": question, "view": "clarification", **update})


def _utc_window(scope: EnquiryScope) -> dict[str, str]:
    """created_from / created_before (UTC ISO) for the scope's business-day range."""
    if not (scope.get("date_from") and scope.get("date_to")):
        return {}
    date_range = DateRange(date.fromisoformat(scope["date_from"]), date.fromisoformat(scope["date_to"]))
    start, end = date_range.utc_bounds(get_frontorder_enquiry_settings().timezone)
    return {"created_from": start.isoformat(), "created_before": end.isoformat()}


# --- nodes -----------------------------------------------------------------


@tracked_step("initialize_run", "Initializing Frontend Order Enquiry Agent", "Unable to start the Frontend Order Enquiry Agent.")
async def initialize_run(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    logger.info("[%s] run=%s session=%s agent started", AGENT, state["run_id"], state.get("session_id"))
    return StepResult("Agent started", {"execution_status": "running", "started_at": utc_now()})


@tracked_step("understand_request", "Understanding the request", "Unable to understand the request.")
async def understand_request_node(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    """Interpret THIS message: action, dates, provider name, page. The LLM classifies the
    message; the dates themselves are computed and validated in dates.py. Merging with
    the conversation's earlier scope happens in resolve_scope."""
    settings = get_frontorder_enquiry_settings()
    today = today_in(settings.timezone)
    message = (state.get("message") or "").strip()
    if not message:
        request = DEFAULT_REQUEST
    else:
        try:
            request, usage = await understand_request(config, message, today)
        except LlmNotConfiguredError as exc:
            raise StepFailed(str(exc)) from exc
        if _monitor(config):
            _monitor(config).llm_call(usage)

    logger.info(
        "[%s] run=%s action=%s period=%s provider_named=%s all_providers=%s page_change=%s",
        AGENT, state["run_id"], request.action, request.period, bool(request.provider_name),
        request.all_providers, request.page_change,
    )
    if request.action == "send_follow_up":
        raise StepBlocked(FOLLOW_UP_EMAIL_UNAVAILABLE)
    if request.action == "other":
        return _clarify(HELP_MESSAGE, action=request.action)
    if request.needs_clarification:
        return _clarify(request.clarification_question or "Which dates should I look at?", action=request.action)

    # "none" = the message names no dates: keep the conversation's dates (resolve_scope).
    dates: dict[str, Any] = {"dates": "inherit"}
    if request.period != "none":
        try:
            date_range = resolve_period(
                request.period,
                today,
                settings.max_range_days,
                last_n_days=request.last_n_days,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        except DateRangeError as exc:
            return _clarify(str(exc), action=request.action)
        dates = {
            "dates": "explicit",
            "date_from": date_range.start.isoformat() if date_range else None,
            "date_to": date_range.end.isoformat() if date_range else None,
        }

    requested = {
        "action": request.action,
        "provider_name": (request.provider_name or "").strip() or None,
        "all_providers": request.all_providers,
        "choice_number": request.choice_number,
        "page_change": request.page_change,
        "page_number": request.page_number,
        **dates,
    }
    label = "inherited dates" if dates["dates"] == "inherit" else (
        f"{dates['date_from']} to {dates['date_to']}" if dates.get("date_from") else "all dates"
    )
    return StepResult(
        f"{request.request_summary or 'Order enquiries'} ({label})",
        {"requested": requested, "request_summary": request.request_summary, "clarification": None},
        data={"action": request.action, "dateFrom": dates.get("date_from"), "dateTo": dates.get("date_to")},
    )


@tracked_step("resolve_scope", "Resolving the provider and dates", "Unable to resolve the provider.")
async def resolve_scope(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    """Combine this message with the conversation's previous scope:
    - action / dates / provider not mentioned now -> kept from the previous message;
    - a new provider or new dates replace the old ones;
    - a provider is identified by Bootog's provider list only (never by the LLM).
    An ambiguous or unknown provider name ends the run with a question; no enquiry API
    is called for it, and no unrestricted request is made instead."""
    previous: EnquiryScope = state.get("scope") or {}
    requested = state.get("requested") or {}

    action = requested.get("action")
    if action in (None, "continue"):
        action = previous.get("action") or DEFAULT_ACTION

    if requested.get("dates") == "explicit":
        date_from, date_to = requested.get("date_from"), requested.get("date_to")
    else:
        date_from, date_to = previous.get("date_from"), previous.get("date_to")

    provider_id, provider_name = previous.get("provider_id"), previous.get("provider_name")
    offered = state.get("offered_providers") or []
    # The user may type back a name from the last numbered list ("Termite Depot (Jacksonville, FL)").
    picked_by_name = offered_provider_by_name(requested.get("provider_name"), offered)
    if state.get("target_provider_id"):
        # A provider picked in the UI: confirm it exists before using it.
        found = await verify_provider.ainvoke({"provider_id": state["target_provider_id"]}, config=config)
        if not found:
            return _clarify("I couldn't find that provider in Bootog. Which provider do you mean?", action=action)
        provider_id, provider_name = found["providerId"], found["providerName"]
    elif picked_by_name:
        # A name copied from the list wins over a number the model may also have read
        # from it ("Termite Depot — 1" is the provider, not row 1).
        provider_id, provider_name = picked_by_name["providerId"], picked_by_name["providerName"]
    elif requested.get("choice_number") and offered:
        # "the second one": a row of the numbered list the previous reply showed. Its id
        # came from Bootog in that earlier turn.
        picked = next((p for p in offered if p["number"] == requested["choice_number"]), None)
        if picked is None:
            return _clarify(f"There is no number {requested['choice_number']} in the last list. Which provider do you mean?",
                            action=action)
        if not picked.get("providerId"):
            return _clarify("That row counts enquiries that have no provider, so it can't be opened by provider. "
                            "Which provider do you mean?", action=action)
        provider_id, provider_name = picked["providerId"], picked["providerName"]
    elif requested.get("provider_name"):
        match = await resolve_provider_by_name.ainvoke({"provider_name": requested["provider_name"]}, config=config)
        name = requested["provider_name"]
        if match["status"] == "not_found":
            return _clarify(
                f'I couldn\'t find a provider named "{name}". Please check the name, or ask '
                '"how many enquiries" to see the providers that have enquiries.',
                action=action,
            )
        if match["status"] == "ambiguous":
            shown = [{"number": n, **p} for n, p in enumerate(match["candidates"], start=1)]
            more = match["totalMatches"] - len(shown)
            lines = [f'Several providers match "{name}". Which one do you mean? Reply with the number.']
            lines += [f"{p['number']}. {p['providerName']}" + (f" ({p['location']})" if p.get("location") else "") for p in shown]
            if more > 0:
                lines.append(f"{more} more match; type more of the name to narrow it down.")
            return _clarify("\n".join(lines), action=action, provider_candidates=shown, offered_providers=shown)
        provider_id, provider_name = match["provider"]["providerId"], match["provider"]["providerName"]
    elif requested.get("all_providers"):
        provider_id = provider_name = None

    scope: EnquiryScope = {
        "action": action,
        "date_from": date_from,
        "date_to": date_to,
        "provider_id": provider_id,
        "provider_name": provider_name,
    }
    scope["page"] = _page(state, previous, scope)
    view = "provider_detail" if provider_id else "provider_summary"
    logger.info(
        "[%s] run=%s scope action=%s from=%s to=%s provider=%s page=%s view=%s",
        AGENT, state["run_id"], action, date_from, date_to, provider_id, scope["page"], view,
    )
    return StepResult(
        f"{provider_name or 'All providers'} — {date_from or 'all dates'}"
        + (f" to {date_to}" if date_to and date_to != date_from else "")
        + f", page {scope['page']}",
        {"scope": scope, "action": action, "view": view},
        data={"providerId": provider_id, "dateFrom": date_from, "dateTo": date_to, "page": scope["page"], "view": view},
    )


def _offered_labels(row: dict[str, Any]) -> list[str]:
    """The ways a row of a numbered list was written in a reply."""
    name = row.get("providerName") or ""
    labels = [name, f"{name} (id …{row['providerId'][-4:]})"]
    if row.get("location"):
        labels.append(f"{name} ({row['location']})")
    return labels


def offered_provider_by_name(name: str | None, offered: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The row of the last numbered list whose name the user typed back, e.g.
    "Termite Depot (Jacksonville, FL)" or "Termite Depot — 1". Only a single match is
    used; otherwise None, and the name is looked up in Bootog as usual."""
    if not name:
        return None
    typed = re.sub(r"\s*[—–]\s*\d+\s*$", "", name).strip()  # drop a copied " — 12" count
    rows = [row for row in offered if row.get("providerId")]
    # Exact spelling first: "Termite Depot" and "Termite depot" are different providers.
    exact = [row for row in rows if typed in _offered_labels(row)]
    if exact:
        return exact[0] if len(exact) == 1 else None
    folded = [row for row in rows if typed.casefold() in [label.casefold() for label in _offered_labels(row)]]
    return folded[0] if len(folded) == 1 else None


def _page(state: FrontOrderEnquiryAgentState, previous: EnquiryScope, scope: EnquiryScope) -> int:
    """Page of the view: an explicit page wins; "next"/"previous" move from the previous
    page only while the view (action, dates, provider) is unchanged; otherwise page 1."""
    requested = state.get("requested") or {}
    if state.get("requested_page"):
        return state["requested_page"]
    if requested.get("page_number") and requested["page_number"] >= 1:
        return min(int(requested["page_number"]), 10000)
    same_view = all(previous.get(k) == scope.get(k) for k in ("action", "date_from", "date_to", "provider_id"))
    current = previous.get("page") or 1
    if same_view and requested.get("page_change") == "next":
        return current + 1
    if same_view and requested.get("page_change") == "previous":
        return max(1, current - 1)
    return 1


@tracked_step("summarize_providers", "Counting enquiries per provider", "Unable to count enquiries per provider.")
async def summarize_providers(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    """Provider summary (Stage A): one GetAllOrderEnquery call returns providers ranked by
    their number of matching enquiries. No enquiry records are kept, and no per-provider
    calls are made."""
    scope = state["scope"]
    converted = wants_converted(scope.get("action"))
    args: dict[str, Any] = {"page": scope["page"], "converted": converted, **_utc_window(scope)}
    summary = await get_provider_enquiry_summary.ainvoke(args, config=config)
    providers = summary["providers"]
    shown_total = sum(p["count"] for p in providers)
    # The reply numbers rows across pages (page 2 starts at 11 with 10 per page).
    first = (scope["page"] - 1) * summary["pageSize"] + 1
    offered = [
        {"number": n, "providerId": p["providerId"], "providerName": p["providerName"]}
        for n, p in enumerate(providers, start=first)
    ]
    logger.info(
        "[%s] run=%s provider summary providers=%s of %s enquiries_on_page=%s",
        AGENT, state["run_id"], len(providers), summary["totalProviders"], shown_total,
    )
    return StepResult(
        f"{summary['totalProviders']} provider{'s' if summary['totalProviders'] != 1 else ''} with matching enquiries"
        f" — showing {len(providers)}",
        {"provider_summary": providers, "providers_total": summary["totalProviders"], "offered_providers": offered},
        data={"providers": len(providers), "totalProviders": summary["totalProviders"], "enquiries": shown_total},
    )


@tracked_step("fetch_enquiries", "Fetching the provider's order enquiries", "Unable to retrieve order enquiries.")
async def fetch_enquiries(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    """Provider detail (Stage B): only the selected provider's enquiries, with the scope's
    dates. Ask the server for the same orderAction class the request is about, so its
    count is the exact number of matching enquiries. Converted enquiries are not
    discarded: a converted-enquiry request asks for exactly those. select_enquiries
    re-checks every record locally."""
    run_id = state["run_id"]
    scope = state["scope"]
    converted = wants_converted(scope.get("action"))
    args: dict[str, Any] = {"target_provider_id": scope["provider_id"], "converted": converted, **_utc_window(scope)}
    # Follow-up requests must check every record's Contact Log before they can say which
    # ones match, so they read every page (bounded); other requests read one page.
    if scope.get("action") not in FOLLOW_UP_ACTIONS:
        args["page"] = scope["page"]
    logger.info(
        "[%s] run=%s fetching enquiries provider=%s from=%s to=%s converted=%s page=%s",
        AGENT, run_id, scope["provider_id"], scope.get("date_from"), scope.get("date_to"), converted, args.get("page"),
    )
    page = await get_order_enquiries.ainvoke(args, config=config)
    records, total, truncated = page["records"], page["total"], page["truncated"]
    logger.info("[%s] run=%s enquiries received=%s total=%s truncated=%s", AGENT, run_id, len(records), total, truncated)

    if not records:
        message = "No order enquiries returned"
    elif truncated:
        message = f"Retrieved {len(records)} of {total} order enquiries"
    else:
        message = f"Retrieved {_enquiries(len(records))}"
    return StepResult(
        message,
        {"raw_enquiries": records, "enquiries_total": total, "enquiries_truncated": truncated},
        data={"retrieved": len(records), "total": total, "truncated": truncated},
    )


@tracked_step("validate_enquiries", "Validating enquiry records", "Unable to validate order enquiries.")
async def validate_enquiries_node(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    valid, malformed, duplicates = validate_enquiries(state.get("raw_enquiries", []))
    if malformed:
        logger.warning("[%s] run=%s skipped %s malformed enquiry record(s) (not an object or no id)", AGENT, state["run_id"], malformed)
    if duplicates:
        logger.info("[%s] run=%s dropped %s duplicate enquiry record(s)", AGENT, state["run_id"], duplicates)

    message = f"{len(valid)} valid enquiry record{'s' if len(valid) != 1 else ''}"
    skipped = [part for part in (f"{malformed} malformed" if malformed else "", f"{duplicates} duplicate" if duplicates else "") if part]
    if skipped:
        message += f" — skipped {', '.join(skipped)}"
    return StepResult(
        message,
        {"valid_enquiries": valid, "malformed_count": malformed, "duplicate_count": duplicates},
        data={"valid": len(valid), "malformed": malformed, "duplicates": duplicates},
    )


@tracked_step("select_enquiries", "Selecting the enquiries the request is about", "Unable to select order enquiries.")
async def select_enquiries(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    """Classify every record by orderAction, then keep the class the request is about.
    The fetch was already scoped the same way; this local check guards against a
    response that ignored the filter and handles a missing/unreadable orderAction."""
    run_id = state["run_id"]
    action = state.get("action")
    classified = classify_enquiries(state.get("valid_enquiries", []))
    selected = filter_enquiries_for_intent(action, classified)
    converted_ids = [enquiry_id(r) or "" for r in classified.converted]
    invalid_ids = [enquiry_id(r) or "" for r in classified.unknown]
    logger.info(
        "[%s] run=%s converted=%s not_converted=%s invalid_order_action=%s selected=%s",
        AGENT, run_id, len(classified.converted), len(classified.not_converted), len(invalid_ids), len(selected),
    )
    if invalid_ids:
        logger.warning("[%s] run=%s enquiries with missing/invalid orderAction set aside: %s", AGENT, run_id, invalid_ids)
    if _monitor(config):
        if wants_converted(action):
            _monitor(config).outcome(converted_included=len(selected))
        else:
            _monitor(config).outcome(converted_excluded=len(classified.converted))
        _monitor(config).outcome(unknown_order_action_excluded=len(invalid_ids))

    if wants_converted(action):
        message = f"{len(selected)} converted to order"
        left_out = len(classified.not_converted)
        left_out_text = "not converted"
    else:
        message = f"{len(selected)} not converted to order"
        left_out = len(classified.converted)
        left_out_text = "already converted"
    if left_out:
        message += f" — left out {left_out} {left_out_text}"
    if invalid_ids:
        message += f", set aside {len(invalid_ids)} with missing/invalid orderAction"
    return StepResult(
        message,
        {"selected_enquiries": selected, "converted_enquiry_ids": converted_ids, "invalid_order_action_ids": invalid_ids},
        data={
            "selected": len(selected),
            "converted": len(classified.converted),
            "notConverted": len(classified.not_converted),
            "invalidOrderAction": len(invalid_ids),
        },
    )


@tracked_step("check_follow_up", "Checking Contact Log history", "Unable to check the Contact Log history.")
async def check_follow_up(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    """Keep only followed-up (or only not-followed-up) enquiries, using each enquiry's
    Contact Log history. A failed history read fails the step: an enquiry is never
    reported as "not followed up" just because its history could not be read."""
    eligible = state.get("selected_enquiries", [])
    semaphore = asyncio.Semaphore(HISTORY_CONCURRENCY)

    async def read_history(record: dict[str, Any]) -> list[dict[str, Any]]:
        async with semaphore:
            return await get_enquiry_contact_history.ainvoke({"enquiry_id": str(record["id"])}, config=config)

    histories = await asyncio.gather(*(read_history(r) for r in eligible))

    follow_up_by_id: dict[str, dict[str, Any]] = {}
    for record, history in zip(eligible, histories):
        follow_up_by_id[str(record["id"])] = {
            "followedUp": any(is_contact_log(row) for row in history),
            "lastContactedAt": latest_contact(history),
        }
    want_followed_up = state.get("action") == "list_followed_up"
    selected = [r for r in eligible if follow_up_by_id[str(r["id"])]["followedUp"] == want_followed_up]
    followed = sum(1 for v in follow_up_by_id.values() if v["followedUp"])
    logger.info("[%s] run=%s follow_up checked=%s followed_up=%s", AGENT, state["run_id"], len(eligible), followed)
    label = "followed up" if want_followed_up else "not followed up yet"
    return StepResult(
        f"{followed} of {_enquiries(len(eligible))} already followed up — showing {len(selected)} {label}",
        # selected_enquiries is narrowed to the requested follow-up state.
        {"selected_enquiries": selected, "follow_up_by_id": follow_up_by_id},
        data={"checked": len(eligible), "followedUp": followed, "selected": len(selected)},
    )


@tracked_step("prepare_enquiries", "Preparing enquiry context", "Unable to prepare order enquiries.")
async def prepare_enquiries(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    run_id = state["run_id"]
    follow_up_by_id = state.get("follow_up_by_id") or {}
    contexts = []
    for record in state.get("selected_enquiries", []):
        context = build_enquiry_context(record)
        context.update(follow_up_by_id.get(context["enquiryId"], {}))
        contexts.append(context)
    logger.info("[%s] run=%s prepared %s enquiries", AGENT, run_id, len(contexts))
    return StepResult(
        f"Prepared {_enquiries(len(contexts))} for processing",
        {"enquiry_contexts": contexts},
        data={"prepared": len(contexts)},
    )


async def respond(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Last node of every run: builds the chat reply and the result, records business
    outcomes, and drops the raw enquiry records from state so the conversation
    checkpoint keeps only the scope and the display result — no customer contact data."""
    status = state.get("execution_status") or "completed"
    if status == "running":
        status = "completed"
    state = {**state, "execution_status": status}
    monitor = _monitor(config)
    if monitor and status == "completed":
        if state.get("view") == "provider_summary":
            monitor.outcome(providers_summarized=len(state.get("provider_summary") or []))
        elif state.get("view") == "provider_detail":
            monitor.outcome(enquiries_retrieved=len(state.get("raw_enquiries") or []))
    return {
        "execution_status": status,
        "reply": build_reply(state),
        "result": build_result(state),
        "raw_enquiries": [],
        "valid_enquiries": [],
        "selected_enquiries": [],
        "enquiry_contexts": [],
        "follow_up_by_id": {},
    }
