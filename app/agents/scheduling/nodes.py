"""Scheduling Agent nodes. Each tool node does one job against live Bootog data and
publishes every API step to Live Activity (running -> completed / failed / blocked).

  understand_message   LLM (or a chat card's structured action) -> intent + references
  order_summary        GET Order/GetOrderHeaders -> dashboard groups and counts
  list_orders          GET Order/GetOrderDetails with filters resolved from live lookups
  order_details        resolve the order -> GET Order/GetOrderDetailById
  find_producers       GET UserDetail/GetOrderUsers for the order's location
  prepare_assignment   -> pending "assign" action (waits for confirmation)
  check_availability   working hours + blocked days + booked appointments -> open slots
  prepare_schedule     -> pending "schedule" action (waits for confirmation)
  prepare_log          collects the Add Log details -> pending "add_log" action
  list_logs            GET ClientContactLog for the order
  list_reference       service types, programs, time zones, statuses, providers
  execute_action       re-validates, POSTs, verifies (only after an explicit confirmation)
  cancel_action        drops the pending action / log draft
  respond              LLM reply written only from the turn's facts

Nothing is written to Bootog unless the user confirmed exactly the action they were shown
in an earlier turn; the order (and, for a booking, the producer's calendar) is re-checked
right before the write, and the result is verified by reading the order back."""

import asyncio
import functools
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.contact_us.agent_nodes import ChatEmitter
from app.agents.scheduling import domain
from app.agents.scheduling.state import SchedulingChatState
from app.agents.scheduling.understanding import INTENT_LABELS, TurnUnderstanding, compose_reply, understand
from app.core.config import get_settings
from app.core.logging import get_logger, log_step
from app.integrations.bootog import BootogApiError, BootogClient, BootogResultError, OrdersApi, SchedulingApi
from app.integrations.bootog.scheduling import PRODUCER_GROUPS
from app.llm import LlmNotConfiguredError

AGENT = "SCHEDULING_AGENT"
logger = get_logger(__name__)

ACTIVITY_TITLES: dict[str, str] = {
    "start_turn": "Agent Session",
    "understand_message": "Understanding Request",
    "load_order_headers": "Order Status Counts",
    "load_lookups": "Reference Data",
    "load_orders": "Order Search",
    "load_order": "Order Details",
    "load_producers": "Producer Search",
    "load_service_requirements": "Service Requirements",
    "load_producer_workload": "Producer Workload",
    "load_calendar": "Producer Availability",
    "recheck_order": "Order Re-check",
    "recheck_calendar": "Availability Re-check",
    "assign_or_schedule": "Assign / Schedule",
    "create_contact_log": "Add Log",
    "verify_action": "Verification",
    "load_logs": "Contact Logs",
    "awaiting_user": "Waiting for User",
    "awaiting_confirmation": "Waiting for Confirmation",
}

CAPABILITIES = [
    "Order status summary matching the Orders dashboard (New, Assigned, Scheduled, QA, TPQA, ...)",
    "List orders by status, service type, program, state, created or scheduled date, rush, or text search — with paging",
    "Find an order by job number, policy number, customer name or address, and show its details",
    "Find eligible producers near an order and assign the order (after your confirmation)",
    "Show a producer's open appointment slots and schedule the inspection (after your confirmation)",
    "Add contact logs (phone call, SMS, email, note) for the order and show its log history",
    "List service types, programs, time zones, order statuses, sub-statuses and providers",
]

_WRITE_TYPES = {"assign", "schedule", "add_log", "assign_batch"}


class NeedInput(Exception):
    """The agent needs something from the user; the turn ends with a question."""

    def __init__(self, text: str, kind: str = "clarify", update: dict[str, Any] | None = None, **details: Any):
        super().__init__(text)
        self.question = {"type": kind, "text": text, **details}
        self.update = update or {}


class Blocked(Exception):
    """The request can't go ahead (e.g. the order is closed). Not a failure of the agent."""

    def __init__(self, text: str, update: dict[str, Any] | None = None):
        super().__init__(text)
        self.update = update or {}


# --- per-turn context --------------------------------------------------------------


@dataclass
class Turn:
    state: SchedulingChatState
    config: RunnableConfig
    # The node's state update, kept even when the node ends with a question or a block
    # (e.g. the order resolved before "which producer?" stays selected).
    update: dict[str, Any] = field(default_factory=dict)

    def show(self, *cards: str) -> None:
        """Result cards this turn produced (the UI renders them under the reply)."""
        self.update["shown"] = [*self.update.get("shown", self.state.get("shown") or []), *cards]

    def note(self, text: str) -> None:
        self.update["notices"] = [*self.update.get("notices", self.state.get("notices") or []), text]

    @property
    def emitter(self) -> ChatEmitter:
        return self.config["configurable"]["emitter"]

    @property
    def run_id(self) -> str:
        return self.state["run_id"]

    @functools.cached_property
    def client(self) -> BootogClient:
        return self.config["configurable"]["bootog_client"]

    @functools.cached_property
    def orders(self) -> OrdersApi:
        return OrdersApi(self.client)

    @functools.cached_property
    def scheduling(self) -> SchedulingApi:
        return SchedulingApi(self.client)

    @functools.cached_property
    def clock(self) -> domain.LocalClock:
        return domain.LocalClock(int(self.state.get("utc_offset_minutes") or 0))

    @functools.cached_property
    def u(self) -> dict[str, Any]:
        """This turn's understanding (a copy: nodes may fill in values carried from an answered question)."""
        return dict(self.state.get("understanding") or {})

    @asynccontextmanager
    async def step(self, step_id: str, running: str, failure: str):
        """One tracked API step. Set `record.message` (and `.data`) before leaving the block."""
        record = _StepRecord()
        await self.emitter.step(step_id, "running", running)
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            yield record
        except (NeedInput, Blocked) as exc:
            status = "blocked" if isinstance(exc, Blocked) else "completed"
            await self.emitter.step(step_id, status, str(exc) if status == "blocked" else record.message or "Done", duration_ms=elapsed())
            raise
        except BootogApiError as exc:
            await self.emitter.step(step_id, "failed", f"{failure} {exc.message}", duration_ms=elapsed())
            log_step(logger, AGENT, self.run_id, step_id, "failed", elapsed())
            raise
        except Exception:
            await self.emitter.step(step_id, "failed", failure, duration_ms=elapsed())
            log_step(logger, AGENT, self.run_id, step_id, "failed", elapsed())
            raise
        await self.emitter.step(step_id, "completed", record.message or "Done", data=record.data, duration_ms=elapsed())
        log_step(logger, AGENT, self.run_id, step_id, "completed", elapsed())


class _StepRecord:
    message: str = ""
    data: dict[str, Any] | None = None


NodeFn = Callable[[Turn], Awaitable[dict[str, Any]]]


def agent_node(name: str):
    """Runs a node with a Turn and turns its outcome into a state update: a question for the
    user, a blocked request, or a failure with a user-safe message."""

    def decorator(fn: NodeFn):
        # Not functools.wraps: LangGraph reads the wrapped function's type hints and would
        # coerce the state into a Turn.
        async def node(state: SchedulingChatState, config: RunnableConfig) -> dict[str, Any]:
            turn = Turn(state, config)
            steps = [*(state.get("turn_steps") or []), name]
            try:
                result = await fn(turn)
                update = {**turn.update, **result}
            except NeedInput as exc:
                if turn.u.get("intent") == "assign_order":
                    # Answering "which order/producer?" continues the same assignment request.
                    exc.question["resume"] = {k: turn.u[k] for k in ("intent", "producer", "max_distance_miles", "reassign")
                                              if turn.u.get(k) not in (None, {}, False)}
                update = {**turn.update, **exc.update, "question": exc.question, "pending_question": exc.question,
                          "awaiting_user_input": True}
            except Blocked as exc:
                update = {**turn.update, **exc.update, "execution_status": "blocked", "error": str(exc)}
            except BootogApiError as exc:
                update = {**turn.update, "execution_status": "failed", "error": exc.message}
            except Exception as exc:  # unexpected: details in logs only
                logger.exception("[%s] run=%s node=%s unexpected %s", AGENT, state.get("run_id"), name, type(exc).__name__)
                update = {**turn.update, "execution_status": "failed", "error": "Something went wrong while working on that."}
            return {**update, "turn_steps": steps}

        node.__name__ = node.__qualname__ = fn.__name__
        node.__doc__ = fn.__doc__
        return node

    return decorator


# --- turn start + understanding ----------------------------------------------------


async def start_turn(state: SchedulingChatState, config: RunnableConfig) -> dict[str, Any]:
    if not state.get("workflow_status"):
        await config["configurable"]["emitter"].step("start_turn", "completed", "Scheduling Agent session started")
    return {
        "intent": None,
        "understanding": None,
        "turn_steps": ["start_turn"],
        "shown": [],
        "notices": [],
        "question": None,
        "answered_question": None,
        "action_result": None,
        "error": None,
        "execution_status": "running",
        "agent_response": None,
        "workflow_status": "running",
        "awaiting_user_input": False,
    }


def _context_for_llm(turn: Turn) -> dict[str, Any]:
    s = turn.state
    order = s.get("selected_order")
    listing = s.get("orders") or {}
    producers = (s.get("producers") or {}).get("items") or []
    availability = s.get("availability") or {}
    now = turn.clock.now()
    return {
        "today": now.date().isoformat(),
        "weekday": now.strftime("%A"),
        "user_time": now.strftime("%I:%M %p"),
        "current_order": {k: order.get(k) for k in ("jobNumber", "policyNumber", "customer", "address", "service")}
        | {"status": domain.order_status_text(order), "assigned_to": (domain.assigned_producer(order) or {}).get("name")}
        if order else None,
        "last_order_list": [
            {"n": i + 1, "jobNumber": o.get("jobNumber"), "policyNumber": o.get("policyNumber"), "customer": o.get("customer")}
            for i, o in enumerate((listing.get("items") or [])[:60])
        ] or None,
        "last_order_list_size": len(listing.get("items") or []) or None,
        "batch_being_prepared": [
            {"n": i, **{k: v for k, v in spec.items()}}
            for i, spec in sorted((s.get("batch_draft") or {}).get("mapping", {}).items())
        ] or None,
        "last_order_list_page": listing.get("page"),
        "last_producer_list": [{"n": i + 1, "name": p["name"]} for i, p in enumerate(producers[:10])] or None,
        "last_slots": [
            {"n": i + 1, "date": d, "start": slot["start"]} for i, (d, slot) in enumerate(_flat_slots(availability)[:15])
        ] or None,
        "pending_action": _pending_view(s.get("pending_action")),
        "log_being_prepared": {k: v for k, v in (s.get("log_draft") or {}).items() if k != "orderId"} or None,
        "pending_question": (s.get("pending_question") or {}).get("text"),
    }


_ACTION_INTENTS = {
    "confirm": "confirm_action", "cancel": "cancel_action", "select_order": "order_details", "assign": "assign_order",
    "producers": "find_producers", "availability": "check_availability", "schedule": "schedule_order",
    "list_status": "list_orders", "page": "list_orders", "add_log": "add_log", "logs": "list_logs", "summary": "order_summary",
    "reassign": "assign_order",
}


def _understand_action(action: dict[str, Any]) -> dict[str, Any]:
    """A chat card's button: exact ids, no interpretation needed."""
    kind = action.get("type")
    intent = _ACTION_INTENTS.get(kind or "")
    if intent is None:
        raise Blocked("That action isn't supported.")
    u: dict[str, Any] = {"intent": intent, "summary": INTENT_LABELS[intent], "action_id": action.get("actionId")}
    order_id = action.get("orderId")
    if order_id:
        if not domain.is_uuid(order_id):
            raise Blocked("That order reference isn't valid.")
        u["order"] = {"order_id": order_id}
    if action.get("producerId"):
        u["producer"] = {"id": action["producerId"]}
    if kind == "schedule":
        u["slot"] = {"date": action.get("date"), "start_time": action.get("start")}
    if kind == "list_status" and action.get("statusName"):
        u["filters"] = {"status_names": [str(action["statusName"])[:60]]}
    if kind == "page":
        u["filters"] = {"page": action.get("page")}
    if kind == "reassign":
        u["reassign"] = True
    if action.get("maxDistanceMiles"):
        u["max_distance_miles"] = action["maxDistanceMiles"]
    if kind == "add_log" and action.get("channel"):
        u["log"] = {"channel": action["channel"]}
    return u


@agent_node("understand_message")
async def understand_message(turn: Turn) -> dict[str, Any]:
    async with turn.step("understand_message", "Understanding your message", "I couldn't interpret that message.") as rec:
        action = turn.state.get("action")
        if action:
            u = _understand_action(action)
            resume = (turn.state.get("pending_question") or {}).get("resume")
            if action.get("type") == "select_order" and resume:
                u = {**resume, **u, "intent": resume["intent"], "summary": INTENT_LABELS[resume["intent"]]}
        else:
            try:
                result: TurnUnderstanding = await understand(
                    turn.config, turn.state.get("conversation_history") or [], _context_for_llm(turn)
                )
            except LlmNotConfiguredError as exc:
                raise Blocked("The AI service is not configured, so I can't process messages yet.") from exc
            u = result.model_dump(exclude_none=True)
            asked = turn.state.get("pending_question") or {}
            if u["intent"] == "confirm_action" and not turn.state.get("pending_action") and asked.get("type") == "confirm_reassign":
                # "Yes" to "already assigned — reassign?": nothing to confirm yet, it continues the assignment.
                u.update({"intent": "assign_order", "reassign": True})
        rec.message = f"Intent: {INTENT_LABELS.get(u['intent'], u['intent'])}"
        rec.data = {"intent": u["intent"]}
    return {"intent": u["intent"], "understanding": u, "pending_question": None,
            "answered_question": turn.state.get("pending_question")}


# --- order dashboard + listing -------------------------------------------------------


async def _headers(turn: Turn) -> list[dict[str, Any]]:
    async with turn.step("load_order_headers", "Loading order status counts", "Unable to load the order status counts.") as rec:
        headers = await turn.orders.order_headers()
        rec.message = f"{len(headers)} status cards loaded"
    return headers


@agent_node("order_summary")
async def order_summary(turn: Turn) -> dict[str, Any]:
    headers = await _headers(turn)
    summary = domain.dashboard(headers)
    wanted = (turn.u.get("filters") or {}).get("status_names") or []
    cards, unknown = domain.match_status_cards(headers, wanted)
    summary["asked"] = [{"label": domain.card_label(c["name"]), "count": domain._int(c.get("count"))} for c in cards]
    update: dict[str, Any] = {"summary": summary}
    turn.show("summary")
    if unknown:
        turn.note(f"Unknown status name(s): {', '.join(unknown)}")
    return update


def _match_lookup(items: list[dict[str, Any]], names: list[str], what: str) -> tuple[list[str], list[str]]:
    ids, missing = [], []
    for name in names:
        key = domain._norm(name)
        exact = [i for i in items if key in {domain._norm(i.get("text")), domain._norm(i.get("aliasName"))}]
        hits = exact or [i for i in items if key and key in domain._norm(i.get("text"))]
        if not hits:
            missing.append(name)
        for item in hits:
            if item.get("value") and item["value"] not in ids:
                ids.append(item["value"])
    return ids, missing


async def _resolve_filters(turn: Turn, filters: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Understanding filters -> GetOrderDetails parameters, ids looked up live."""
    api_filters: dict[str, Any] = {}
    described: list[str] = []
    if filters.get("status_names"):
        headers = await _headers(turn)
        cards, unknown = domain.match_status_cards(headers, filters["status_names"])
        if unknown:
            known = ", ".join(domain.card_label(h["name"]) for h in headers if h.get("name"))
            raise NeedInput(f"I couldn't find the status {', '.join(unknown)}. Known statuses: {known}.", kind="choose_status")
        api_filters.update(domain.status_filters(cards))
        described.append("status " + ", ".join(domain.card_label(c["name"]) for c in cards))
    lookups = [k for k in ("service_type_names", "program_names") if filters.get(k)]
    if lookups:
        async with turn.step("load_lookups", "Looking up service types / programs", "Unable to load the lookup data.") as rec:
            if filters.get("service_type_names"):
                ids, missing = _match_lookup(await turn.orders.service_types(), filters["service_type_names"], "service type")
                if missing:
                    raise NeedInput(f"I couldn't find the service type {', '.join(missing)}.", kind="choose_service_type")
                api_filters["mainServiceTypeIds"] = ids
                described.append("service " + ", ".join(filters["service_type_names"]))
            if filters.get("program_names"):
                ids, missing = _match_lookup(await turn.orders.programs(), filters["program_names"], "program")
                if missing:
                    raise NeedInput(f"I couldn't find the program {', '.join(missing)}.", kind="choose_program")
                api_filters["programIds"] = ids
                described.append("program " + ", ".join(filters["program_names"]))
            rec.message = "Lookups resolved"
    if filters.get("state_codes"):
        api_filters["propertyStates"] = [s.strip().upper()[:2] for s in filters["state_codes"] if s.strip()]
        described.append("state " + ", ".join(api_filters["propertyStates"]))
    if filters.get("rush_only"):
        api_filters["isRushOrder"] = True
        described.append("rush only")
    if filters.get("customer_name"):
        api_filters["customerName"] = filters["customer_name"].strip()[:100]
        described.append(f'customer "{api_filters["customerName"]}"')
    if filters.get("search_text"):
        api_filters["allField"] = filters["search_text"].strip()[:100]
        described.append(f'matching "{api_filters["allField"]}"')
    sched_from, sched_to = domain.parse_day(filters.get("scheduled_from")), domain.parse_day(filters.get("scheduled_to"))
    if sched_from or sched_to:
        first, last = sched_from or sched_to, sched_to or sched_from
        api_filters["ScheduleFromDate"], api_filters["ScheduleToDate"] = turn.clock.day_bounds_utc(first, last)
        described.append(f"scheduled {first.isoformat()}" + (f" to {last.isoformat()}" if last != first else ""))
    return api_filters, described


@agent_node("list_orders")
async def list_orders(turn: Turn) -> dict[str, Any]:
    settings = get_settings()
    filters = dict(turn.u.get("filters") or {})
    previous = turn.state.get("orders_query") or {}
    given = {k for k, v in filters.items() if v not in (None, "", [], False)}
    paging = bool(filters.get("page_action")) or (bool(filters.get("page")) and not given - {"page", "page_action"})
    if paging and previous:
        query = dict(previous)
        current = int(previous.get("page") or 1)
        page = int(filters["page"]) if filters.get("page") else {"next": current + 1, "previous": max(1, current - 1)}.get(
            filters.get("page_action"), 1
        )
    else:
        api_filters, described = await _resolve_filters(turn, filters)
        created_from, created_to = domain.parse_day(filters.get("created_from")), domain.parse_day(filters.get("created_to"))
        if created_from or created_to:
            described.append(
                f"created {(created_from or created_to).isoformat()}"
                + (f" to {created_to.isoformat()}" if created_to and created_from and created_to != created_from else "")
            )
        query = {
            "filters": api_filters,
            "created_from": created_from.isoformat() if created_from else None,
            "created_to": created_to.isoformat() if created_to else None,
            "description": "; ".join(described) or "all orders, newest first",
        }
        page = int(filters.get("page") or 1)
    page = max(1, page)
    page_size = settings.scheduling_page_size

    # The chat shows every matching order: all of GetOrderDetails' pages are read
    # (bounded by bootog_max_pages; hitting the bound is reported as truncated).
    page = 1
    async with turn.step("load_orders", "Searching orders", "Unable to load orders.") as rec:
        if query.get("created_from") or query.get("created_to"):
            items, total, truncated = await _orders_created_between(turn, query, settings.bootog_max_pages)
        else:
            items, total, truncated = await _all_orders(turn, query["filters"], settings.bootog_max_pages)
        rec.message = f"{total} order(s) found" + (f" — showing {len(items)}" if len(items) < total else "")
        rec.data = {"total": total, "shown": len(items)}

    summaries = [domain.order_summary(r) for r in items]
    listing = {
        "items": summaries, "total": total, "page": page, "pageSize": max(len(summaries), page_size), "context": "listed",
        "description": query["description"], "truncated": truncated,
    }
    update: dict[str, Any] = {"orders": listing, "orders_query": {**query, "page": page}}
    turn.show("orders")
    if page > 1 and not summaries:
        turn.note(f"There is no page {page}; the list has {total} order(s).")
    return update


def _fetch_page_size() -> int:
    return max(1, get_settings().scheduling_order_fetch_page_size)


async def _all_orders(turn: Turn, filters: dict[str, Any], max_pages: int) -> tuple[list[dict[str, Any]], int, bool]:
    """Every order matching the filters, read page by page until GetOrderDetails' count is
    reached. An order seen on an earlier page (the list shifted while paging) is kept once."""
    batch, records, seen, total = _fetch_page_size(), [], set(), 0
    for page in range(1, max_pages + 1):
        result = await turn.orders.list_orders(page=page, page_size=batch, filters=filters)
        total = result.total
        for record in result.records:
            key = record.get("id")
            if key and key in seen:
                continue
            seen.add(key)
            records.append(record)
        if len(result.records) < batch or page * batch >= total:
            return records, total, False
    return records, total, True


async def _orders_created_between(turn: Turn, query: dict[str, Any], max_pages: int) -> tuple[list[dict[str, Any]], int, bool]:
    """GetOrderDetails has no created-date filter: read newest-first pages and keep the
    orders created in the user's local date range, stopping once past it."""
    first = domain.parse_day(query.get("created_from")) or date.min
    last = domain.parse_day(query.get("created_to")) or date.max
    batch, matched = _fetch_page_size(), []
    for page in range(1, max_pages + 1):
        result = await turn.orders.list_orders(page=page, page_size=batch, filters=query["filters"])
        older = False
        for record in result.records:
            day = turn.clock.local_date_of(record.get("createdDateTime"))
            if day is None:
                continue
            if day < first:
                older = True
                break
            if day <= last:
                matched.append(record)
        if older or len(result.records) < batch or page * batch >= result.total:
            return matched, len(matched), False
    return matched, len(matched), True


# --- order resolution ---------------------------------------------------------------


async def _load_order(turn: Turn, order_id: str, step_id: str = "load_order") -> dict[str, Any]:
    async with turn.step(step_id, "Loading the order", "Unable to load the order.") as rec:
        order = domain.order_summary(await turn.orders.get_order(order_id))
        rec.message = f"Order {order.get('jobNumber') or order.get('policyNumber') or ''} — {domain.order_status_text(order)}".strip()
    return order


async def _search_orders(turn: Turn, filters: dict[str, Any], label: str) -> list[dict[str, Any]]:
    async with turn.step("load_orders", f"Searching orders for {label}", "Unable to search orders.") as rec:
        result = await turn.orders.list_orders(page=1, page_size=10, filters=filters)
        rec.message = f"{result.total} order(s) match {label}"
    return [domain.order_summary(r) for r in result.records]


async def resolve_order(turn: Turn, update: dict[str, Any]) -> dict[str, Any]:
    """The order the user means, freshly loaded. Asks when it is unknown or ambiguous;
    a newly selected order resets producer/slot/log context that belonged to another one."""
    ref = turn.u.get("order") or {}
    current = turn.state.get("selected_order")
    order_id: str | None = None

    if ref.get("order_id"):
        order_id = ref["order_id"].strip()
        if not domain.is_uuid(order_id):
            raise NeedInput("That doesn't look like a valid order id. Please give the job or policy number.", kind="choose_order")
    elif ref.get("position"):
        items = (turn.state.get("orders") or {}).get("items") or []
        pos = ref["position"]
        if not items:
            raise NeedInput("There is no order list yet — which order do you mean (job or policy number)?", kind="choose_order")
        index = len(items) - 1 if pos == -1 else pos - 1
        if not 0 <= index < len(items):
            raise NeedInput(f"The last list has {len(items)} orders — which number do you mean?", kind="choose_order")
        order_id = items[index]["id"]
    else:
        number = ref.get("job_number") or ref.get("policy_number")
        text = ref.get("search_text")
        customer = (ref.get("customer_name") or "").strip()[:100]
        if customer:
            order_id = await _order_by_customer(turn, customer, ref.get("status_names") or [])
        elif number:
            key = "jobNumber" if ref.get("job_number") else "policyNumber"
            found = domain.match_orders(await _search_orders(turn, {key: number}, number), number)
            if not found:
                found = domain.match_orders(await _search_orders(turn, {"allField": number}, number), number)
            if not found:
                raise NeedInput(f"I couldn't find an order with number {number}. Please check it.", kind="choose_order")
            if len(found) > 1:
                _candidates(turn, found, f"orders numbered {number}")
                raise NeedInput(f"{len(found)} orders have number {number} — which one?", kind="choose_order", update=update)
            order_id = found[0]["id"]
        elif text:
            status = await _status_filter(turn, ref.get("status_names") or [])
            found = await _search_orders(turn, {"allField": text[:100], **status}, f'"{text}"')
            if not found:
                raise NeedInput(f'No order matches "{text}". Try a job or policy number.', kind="choose_order")
            if len(found) > 1:
                _candidates(turn, found, f'orders matching "{text}"')
                raise NeedInput(f'{len(found)} orders match "{text}" — which one?', kind="choose_order", update=update)
            order_id = found[0]["id"]
        elif current:
            order_id = current["id"]
        else:
            raise NeedInput("Which order? Give me its job number or policy number, or pick one from a list.", kind="choose_order")

    order = await _load_order(turn, order_id)
    update["selected_order"] = order
    if not current or current.get("id") != order["id"]:
        update.update({"producers": None, "ranking": None, "availability": None, "log_draft": None, "logs": None})
        pending = turn.state.get("pending_action")
        if pending and pending.get("orderId") != order["id"]:
            update["pending_action"] = None
    return order


async def _status_filter(turn: Turn, names: list[str]) -> dict[str, Any]:
    """GetOrderDetails status parameters for status names, from the live status cards."""
    if not names:
        return {}
    headers = await _headers(turn)
    cards, unknown = domain.match_status_cards(headers, names)
    if unknown:
        known = ", ".join(domain.card_label(h["name"]) for h in headers if h.get("name"))
        raise NeedInput(f"I couldn't find the status {', '.join(unknown)}. Known statuses: {known}.", kind="choose_status")
    return domain.status_filters(cards)


async def _order_by_customer(turn: Turn, customer: str, status_names: list[str]) -> str:
    """The one order of a customer (optionally in a status). Several -> the user picks."""
    status = await _status_filter(turn, status_names)
    label = f"{customer}'s " + (f"{' / '.join(status_names)} " if status_names else "") + "orders"
    found = domain.match_customer(await _search_orders(turn, {"customerName": customer, **status}, label), customer)
    if not found:  # the name may only be stored on the contact: search all fields too
        found = domain.match_customer(await _search_orders(turn, {"allField": customer, **status}, label), customer)
    if not found:
        raise NeedInput(f"I couldn't find {label}. Check the name, or give the job or policy number.", kind="choose_order")
    if len(found) > 1:
        _candidates(turn, found, label)
        raise NeedInput(f"{customer} has {len(found)} matching orders — which one should I use?", kind="choose_order",
                        update=turn.update)
    return found[0]["id"]


def _candidates(turn: Turn, orders: list[dict[str, Any]], description: str) -> None:
    turn.update["orders"] = {
        "items": orders, "total": len(orders), "page": 1, "pageSize": len(orders), "context": "candidates",
        "description": description, "truncated": False,
    }
    turn.show("orders")


@agent_node("order_details")
async def order_details(turn: Turn) -> dict[str, Any]:
    update = turn.update
    await resolve_order(turn, update)
    turn.show("order")
    return update


# --- producers ----------------------------------------------------------------------


def _distance_limit(turn: Turn) -> float:
    default = float(get_settings().scheduling_producer_radius_miles)
    try:
        limit = float(turn.u.get("max_distance_miles") or default)
    except (TypeError, ValueError):
        limit = default
    return max(1.0, min(limit, 500.0))


async def _state_name(turn: Turn, order: dict[str, Any]) -> str | None:
    """The state's name for the county filter, from BOOTOG's own state list (as the screen does)."""
    state, country = (order.get("state") or "").strip().upper(), (order.get("country") or "").strip().upper()
    if not state or not country:
        return None
    try:
        states = await turn.orders.states(country)
    except BootogApiError:
        return None
    return next((str(s["stateName"]) for s in states if str(s.get("stateCode") or "").strip().upper() == state
                 and s.get("stateName")), None)


async def _load_producers(
    turn: Turn, order: dict[str, Any], limit: float | None = None, filters: dict[str, list[str]] | None = None
) -> list[dict[str, Any]]:
    county = (order.get("county") or "").replace(" County", "").strip() or None
    state = (order.get("state") or "").strip().upper() or None
    limit = limit if limit is not None else _distance_limit(turn)
    state_name = await _state_name(turn, order) if state and county else None
    async with turn.step("load_producers", f"Searching producers within {limit:g} miles", "Unable to load producers.") as rec:
        groups = await turn.scheduling.order_users(
            latitude=order.get("latitude"), longitude=order.get("longitude"), state_code=state, state_name=state_name,
            county=county, target_provider_id=order.get("providerId"), program_id=order.get("programId"),
            max_distance_miles=int(-(-limit // 1)), skillset_ids=(filters or {}).get("skillsetIds"),
            credential_ids=(filters or {}).get("credentialIds"),
        )
        producers = domain.merge_producers(groups, PRODUCER_GROUPS)
        rec.message = f"{len(producers)} producer(s) found"
    assigned = domain.assigned_producer(order)
    for p in producers:
        p["assigned"] = bool(assigned and assigned["id"] == p["id"])
    return producers


# Producers kept for an order (the list the user picks from by name or number).
_PRODUCERS_KEPT = 25


async def _producers_for(
    turn: Turn, order: dict[str, Any], update: dict[str, Any], limit: float | None = None,
    filters: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    cached = update.get("producers") if "producers" in update else turn.state.get("producers")
    if (cached and cached.get("orderId") == order["id"] and (limit is None or cached.get("maxDistanceMiles") == limit)
            and (filters is None or cached.get("filters") == filters)):
        return cached["items"]
    limit = limit if limit is not None else _distance_limit(turn)
    items = await _load_producers(turn, order, limit, filters)
    update["producers"] = {"orderId": order["id"], "items": items[:_PRODUCERS_KEPT], "total": len(items), "maxDistanceMiles": limit,
                           "filters": filters}
    return update["producers"]["items"]


def _show_producers(turn: Turn) -> None:
    turn.show("producers")


@agent_node("find_producers")
async def find_producers(turn: Turn) -> dict[str, Any]:
    update = turn.update
    order = await resolve_order(turn, update)
    update["producers"] = None  # always a fresh search when asked
    limit = _distance_limit(turn)
    items = await _load_producers(turn, order, limit)
    update["producers"] = {"orderId": order["id"], "items": items[:_PRODUCERS_KEPT], "total": len(items), "maxDistanceMiles": limit}
    _show_producers(turn)
    reason = domain.eligibility(order, "assign")
    if reason:
        turn.note(f"This order can't be assigned right now: {reason}.")
    if not items:
        turn.note("No producers were found for this order's location.")
    return update


async def resolve_producer(turn: Turn, order: dict[str, Any], update: dict[str, Any], *, allow_assigned: bool) -> dict[str, Any]:
    """The producer the user means, always one of the order's eligible producers (or the
    producer already assigned to it). Asks when it is unknown or ambiguous."""
    ref = turn.u.get("producer") or {}
    assigned = domain.assigned_producer(order)
    named = any(ref.get(k) for k in ("id", "name", "position"))

    if allow_assigned and not named and not ref.get("use_assigned"):
        # Scheduling without naming anyone: the producer whose slots were just shown for
        # this order, else the producer already assigned to it.
        shown = turn.state.get("availability") or {}
        if shown.get("orderId") == order["id"] and (shown.get("producer") or {}).get("id"):
            return shown["producer"]
    if ref.get("use_assigned") or (allow_assigned and not named and assigned):
        if not assigned:
            raise NeedInput("This order isn't assigned to anyone yet — which producer should do it?", kind="choose_producer")
        items = await _producers_for(turn, order, update)
        found = next((p for p in items if p["id"] == assigned["id"]), None)
        return found or {**assigned, "address": None, "headshot": None}

    items = await _producers_for(turn, order, update)
    if ref.get("id"):
        found = next((p for p in items if p["id"] == ref["id"]), None)
        if found is None and assigned and assigned["id"] == ref["id"]:
            return {**assigned, "address": None, "headshot": None}
        if found is None:
            raise Blocked("That producer isn't in the eligible list for this order any more. Please pick from the current list.")
        return found
    if ref.get("position"):
        pos = ref["position"]
        index = len(items) - 1 if pos == -1 else pos - 1
        if not 0 <= index < len(items):
            _show_producers(turn)
            raise NeedInput(f"The producer list has {len(items)} entries — which number?", kind="choose_producer", update=update)
        return items[index]
    if ref.get("name"):
        hits = domain.match_producers(items, ref["name"])
        if len(hits) == 1:
            return hits[0]
        _show_producers(turn)
        if not hits:
            raise NeedInput(f"{ref['name']} isn't among the eligible producers for this order. Pick one from the list.",
                            kind="choose_producer", update=update)
        raise NeedInput(f"{len(hits)} producers match {ref['name']} — which one?", kind="choose_producer", update=update)
    _show_producers(turn)
    if not items:
        raise Blocked("No eligible producers were found for this order's location.", update=update)
    raise NeedInput("Which producer should get this order? Pick one from the list.", kind="choose_producer", update=update)


# --- assign ---------------------------------------------------------------------------


def _pending(turn: Turn, kind: str, order: dict[str, Any], payload: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "type": kind,
        "orderId": order["id"],
        "orderLabel": order.get("jobNumber") or order.get("policyNumber"),
        "payload": payload,
        "details": details,
        "askedRunId": turn.run_id,
    }


def _ask_confirmation(turn: Turn, pending: dict[str, Any], text: str) -> dict[str, Any]:
    update = turn.update
    update["pending_action"] = pending
    question = {"type": "confirm", "text": text, "actionId": pending["id"]}
    update.update({"question": question, "pending_question": question, "awaiting_user_input": True})
    turn.show("pending")
    return update


async def _service_requirements(turn: Turn, order: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
    """The order's service requirements and the producer matching filters the Assign/Schedule
    screen would send (skill set / credential ids from the service type or the program, kept
    when valid for the provider). If they can't be loaded the assignment stops: matching
    without them would treat every producer of the provider as "matched"."""
    service_type_id = order.get("serviceTypeId")
    if not service_type_id:
        raise Blocked("The order has no service type, so its producer requirements can't be checked.")
    async with turn.step("load_service_requirements", "Checking the service requirements",
                         "Unable to load the service requirements.") as rec:
        service_type = await turn.orders.service_type(service_type_id)
        program = await turn.orders.program_detail(order["programId"]) if order.get("programId") else None
        provider = order.get("providerId")
        skills, docs = await asyncio.gather(turn.orders.skill_set_names(provider), turn.orders.document_type_names(provider))
        filters = domain.matching_filters(
            service_type, program, {str(x.get("value")) for x in skills}, {str(x.get("value")) for x in docs},
        )
        requirements = domain.service_requirements(service_type)
        if requirements is not None:
            requirements["filters"] = {"skillSets": len(filters["skillsetIds"]), "credentials": len(filters["credentialIds"]),
                                       "fromProgram": bool(program and (program.get("programSkillSets") or program.get("programDocumentTypes")))}
        rec.message = (f"Matching on {len(filters['skillsetIds'])} skill set(s) and {len(filters['credentialIds'])} credential(s)"
                       if filters["skillsetIds"] or filters["credentialIds"] else "No skill / credential filters for this service")
    return requirements, filters


async def _workload(turn: Turn, producer: dict[str, Any]) -> dict[str, Any] | None:
    try:
        async with turn.step("load_producer_workload", f"Checking {producer['name']}'s workload",
                             "Unable to load the producer's workload.") as rec:
            body = await turn.scheduling.producer_service_count(producer["id"])
            workload = {k: body.get(k) for k in ("assigned", "scheduled", "submitted", "rejected")}
            rec.message = ", ".join(f"{v} {k}" for k, v in workload.items() if v is not None) or "No workload data"
        return workload
    except BootogApiError:
        return None


@agent_node("prepare_assignment")
async def prepare_assignment(turn: Turn) -> dict[str, Any]:
    """Order -> eligibility -> (already assigned?) -> service requirements -> producers within the
    distance limit, ranked on BOOTOG's data -> recommended (or named) producer -> confirmation."""
    update = turn.update
    answered = turn.state.get("answered_question") or {}
    if answered.get("type") == "confirm_reassign" and turn.u.get("reassign"):
        # "Yes, reassign": keep what the original request asked for.
        if turn.u.get("max_distance_miles") is None and answered.get("maxDistanceMiles"):
            turn.u["max_distance_miles"] = answered["maxDistanceMiles"]
        if not any((turn.u.get("producer") or {}).get(k) for k in ("id", "name", "position")) and answered.get("producer"):
            turn.u["producer"] = answered["producer"]
    order = await resolve_order(turn, update)
    label = order.get("jobNumber") or order.get("policyNumber") or "the order"
    turn.show("order")
    reason = domain.eligibility(order, "assign")
    if reason:
        raise Blocked(f"Order {label} can't be assigned: {reason}.", update=update)
    current = domain.assigned_producer(order)
    if current and not turn.u.get("reassign"):
        raise NeedInput(f"Order {label} is already assigned to {current.get('name') or 'a producer'}. "
                        "Do you want to reassign it?", kind="confirm_reassign", update=update,
                        orderId=order["id"], maxDistanceMiles=turn.u.get("max_distance_miles"),
                        producer=turn.u.get("producer"))

    requirements, filters = await _service_requirements(turn, order)
    filters_sent = bool(filters["skillsetIds"] or filters["credentialIds"] or order.get("programId"))
    limit = _distance_limit(turn)
    ref = turn.u.get("producer") or {}
    named = any(ref.get(k) for k in ("id", "name", "position"))
    ranking = None
    if domain.has_coordinates(order):
        producers = await _producers_for(turn, order, update, limit, filters)
        ranking = domain.rank_producers(producers, max_distance_miles=limit, assigned_id=(current or {}).get("id"),
                                        filters_sent=filters_sent)
        update["ranking"] = {"orderId": order["id"], **ranking}
        turn.show("ranking")
    elif not named:
        raise Blocked(f"Order {label} has no property coordinates, so producers can't be matched by distance. "
                      "Name the producer to assign, or fix the property address first.", update=update)

    if named:
        producer = await resolve_producer(turn, order, update, allow_assigned=False)
        evaluated = next((p for p in (ranking or {}).get("items", []) if p["id"] == producer["id"]), None)
    else:
        evaluated = ranking.get("recommended")
        if evaluated is None:
            summary = (f"No producer within {limit:g} miles matches the service's skill / credential filters for order {label}"
                       if ranking["withinDistanceCount"] else f"No producers were found within {limit:g} miles of order {label}")
            raise NeedInput(f"{summary}. You can widen the distance or name a producer from the list.",
                            kind="choose_producer", update=update)
        producer = evaluated

    workload = await _workload(turn, producer)
    shown = evaluated or producer
    warnings = []
    if named and evaluated is None:
        warnings.append("This producer was not in BOOTOG's candidate list for the distance limit.")
    elif evaluated and not evaluated.get("eligible"):
        warnings.append(f"Not auto-eligible: {evaluated.get('excludedReason')}.")
    details = {
        "order": label, "service": order.get("service"), "customer": order.get("customer"), "address": order.get("address"),
        "producer": producer["name"], "producerPhone": producer.get("phone"), "producerEmail": producer.get("email"),
        "distanceMiles": shown.get("distanceMiles"), "maxDistanceMiles": limit,
        "matched": domain.MATCHED_GROUP in (shown.get("groups") or []) if filters_sent else None,
        "groups": ", ".join(shown.get("groups") or []) or None,
        "rating": shown.get("rating"), "openAssignments": shown.get("openAssignments"),
        "workload": workload, "selection": "named by you" if named else "nearest eligible producer",
        "requirements": requirements, "replaces": current["name"] if current else None,
        "warnings": warnings or None,
    }
    pending = _pending(turn, "assign", order, domain.assign_payload(order, producer), details)
    pending["producerId"] = producer["id"]
    pending["expectedAssigneeId"] = (current or {}).get("id")
    await turn.emitter.step("awaiting_confirmation", "waiting", f"Confirm assigning to {producer['name']}")
    return _ask_confirmation(turn, pending, f"Assign order {label} to {producer['name']}?")


# --- availability + schedule ---------------------------------------------------------


def _flat_slots(availability: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    return [(d["date"], s) for d in availability.get("days") or [] for s in d.get("slots") or []]


def _window(turn: Turn, start: date | None = None, days: int | None = None) -> list[date]:
    settings = get_settings()
    slot = turn.u.get("slot") or {}
    first = start or domain.parse_day(slot.get("from_date")) or turn.clock.today()
    first = max(first, turn.clock.today())
    count = max(1, min(int(days or slot.get("days") or settings.scheduling_slot_days), settings.scheduling_max_slot_days))
    return [first + timedelta(days=i) for i in range(count)]


async def _availability(turn: Turn, order: dict[str, Any], producer: dict[str, Any], days: list[date], step_id: str) -> dict[str, Any]:
    start_iso, end_iso = turn.clock.day_bounds_utc(days[0], days[-1])
    async with turn.step(step_id, f"Checking {producer['name']}'s calendar", "Unable to check the producer's availability.") as rec:
        hours, blocks, booked = await asyncio.gather(
            turn.scheduling.working_hours(producer["id"]),
            turn.scheduling.block_days(producer["id"]),
            turn.scheduling.booked_appointments(producer["id"], start_iso, end_iso),
        )
        duration = domain.order_duration_minutes(order)
        result = domain.available_slots(
            days=days, working_hours=hours, booked=booked, blocks=blocks, duration_minutes=duration,
            clock=turn.clock, ignore_order_id=order["id"], per_day_limit=max(1, get_settings().scheduling_slots_per_day),
        )
        open_count = sum(len(d["slots"]) for d in result)
        rec.message = f"{open_count} open slot(s) over {len(days)} day(s)"
        rec.data = {"openSlots": open_count}
    availability = {
        "orderId": order["id"], "producer": {k: producer.get(k) for k in ("id", "name", "phone", "email", "address", "headshot")},
        "durationMinutes": duration, "days": result, "hasWorkingHours": bool(hours),
    }
    return availability


@agent_node("check_availability")
async def check_availability(turn: Turn) -> dict[str, Any]:
    update = turn.update
    order = await resolve_order(turn, update)
    producer = await resolve_producer(turn, order, update, allow_assigned=True)
    availability = await _availability(turn, order, producer, _window(turn), "load_calendar")
    update["availability"] = availability
    turn.show("availability")
    if not availability["hasWorkingHours"]:
        turn.note(f"{producer['name']} has no working hours set up, so no slots can be offered.")
    reason = domain.eligibility(order, "schedule")
    if reason:
        turn.note(f"This order can't be scheduled right now: {reason}.")
    return update


def _chosen_slot(turn: Turn, order: dict[str, Any], producer: dict[str, Any]) -> tuple[str, str] | None:
    slot = turn.u.get("slot") or {}
    if slot.get("date") and slot.get("start_time"):
        day = domain.parse_day(slot["date"])
        start = domain.parse_clock(slot["start_time"])
        if day and start is not None:
            return day.isoformat(), domain.format_clock(start)
    if slot.get("position"):
        last = turn.state.get("availability") or {}
        if last.get("orderId") == order["id"] and (last.get("producer") or {}).get("id") == producer["id"]:
            flat = _flat_slots(last)
            pos = slot["position"]
            index = len(flat) - 1 if pos == -1 else pos - 1
            if 0 <= index < len(flat):
                return flat[index][0], flat[index][1]["start"]
    return None


@agent_node("prepare_schedule")
async def prepare_schedule(turn: Turn) -> dict[str, Any]:
    update = turn.update
    order = await resolve_order(turn, update)
    reason = domain.eligibility(order, "schedule")
    if reason:
        raise Blocked(f"Order {order.get('jobNumber') or ''} can't be scheduled: {reason}.".replace("  ", " "), update=update)
    producer = await resolve_producer(turn, order, update, allow_assigned=True)
    chosen = _chosen_slot(turn, order, producer)
    if chosen is None:
        availability = await _availability(turn, order, producer, _window(turn), "load_calendar")
        update["availability"] = availability
        turn.show("availability")
        raise NeedInput(f"Which slot should I book with {producer['name']}?", kind="choose_slot", update=update)

    day, start = chosen
    availability = await _availability(turn, order, producer, [date.fromisoformat(day)], "load_calendar")
    slot = domain.slot_is_open(availability["days"], day, start)
    if slot is None:
        update["availability"] = await _availability(turn, order, producer, _window(turn), "load_calendar")
        turn.show("availability")
        raise NeedInput(
            f"{start} on {day} isn't open for {producer['name']} (outside working hours, booked, blocked or too short "
            f"for a {availability['durationMinutes']}-minute inspection). Pick one of the open slots.",
            kind="choose_slot", update=update,
        )

    comments = (turn.u.get("schedule_comments") or "").strip()[:500] or None
    current = domain.assigned_producer(order)
    details = {
        "order": order.get("jobNumber") or order.get("policyNumber"), "service": order.get("service"),
        "customer": order.get("customer"), "address": order.get("address"), "producer": producer["name"],
        "date": day, "weekday": date.fromisoformat(day).strftime("%A"), "start": slot["start"], "end": slot["end"],
        "durationMinutes": availability["durationMinutes"], "comments": comments,
        "replaces": current["name"] if current and current["id"] != producer["id"] else None,
    }
    pending = _pending(turn, "schedule", order, domain.schedule_payload(order, producer, day, slot["start"], slot["end"], comments), details)
    pending["producerId"] = producer["id"]
    pending["producer"] = availability["producer"]
    await turn.emitter.step("awaiting_confirmation", "waiting", f"Confirm booking {day} {slot['start']}")
    return _ask_confirmation(turn, pending, f"Book {producer['name']} for order {details['order']} on {day} at {slot['start']}?")


# --- add log -------------------------------------------------------------------------


@agent_node("prepare_log")
async def prepare_log(turn: Turn) -> dict[str, Any]:
    update = turn.update
    order = await resolve_order(turn, update)
    draft = dict(turn.state.get("log_draft") or {}) if (turn.state.get("log_draft") or {}).get("orderId") == order["id"] else {}
    if "log_draft" in update and update["log_draft"] is None:
        draft = {}
    given = {k: v for k, v in (turn.u.get("log") or {}).items() if v not in (None, "")}
    if given.get("channel") and draft.get("channel") and given["channel"] != draft["channel"]:
        draft = {}  # a different kind of log: start over
    if given.get("party") and draft.get("party") and given["party"] != draft["party"]:
        for key in ("contact_name", "contact_number", "email_to"):
            if key not in given:
                draft.pop(key, None)
    draft.update(given)
    draft["orderId"] = order["id"]

    party = draft.get("party") or "customer"
    draft["party"] = party
    if party == "producer":
        assigned = domain.assigned_producer(order)
        if not assigned and not (draft.get("contact_name") and (draft.get("contact_number") or draft.get("email_to"))):
            update["log_draft"] = draft
            raise NeedInput("This order has no producer assigned — who did you contact (name and phone or email)?",
                            kind="log_details", update=update)
        defaults = {"contact_name": (assigned or {}).get("name"), "contact_number": (assigned or {}).get("phone"),
                    "email_to": (assigned or {}).get("email")}
    else:
        defaults = {"contact_name": order.get("customer"), "contact_number": order.get("contactPhone"),
                    "email_to": order.get("contactEmail")}
    for key, value in defaults.items():
        if not draft.get(key) and value:
            draft[key] = value
    if draft.get("channel") == "phone_call":
        draft.setdefault("direction", "outbound")
        draft.setdefault("caller_type", "policyholder" if party == "customer" else "other")
    draft["about_schedule"] = any(s.get("scheduleDate") for s in order.get("services") or [])

    missing = domain.missing_log_fields(draft)
    update["log_draft"] = draft
    if missing:
        kind = (draft.get("channel") or "log").replace("_", " ")
        raise NeedInput(f"To add the {kind} log I still need: {', '.join(missing)}.", kind="log_details", update=update,
                        missing=missing)

    payload = domain.log_payload(order["id"], draft, turn.clock)
    channel_label = domain.CONTACT_TYPE_LABELS[payload["contactType"]]
    details = {
        "order": order.get("jobNumber") or order.get("policyNumber"), "channel": channel_label,
        "with": "customer" if party == "customer" else "producer", "contact": draft.get("contact_name"),
        "number": draft.get("contact_number") if draft["channel"] in ("phone_call", "sms") else None,
        "emailTo": draft.get("email_to") if draft["channel"] == "email" else None,
        "subject": draft.get("subject") if draft["channel"] == "email" else None, "message": draft.get("message"),
        "result": domain.RESPONSE_LABELS.get(payload["responseResult"]) if draft["channel"] == "phone_call" else None,
        "direction": draft.get("direction") if draft["channel"] == "phone_call" else None,
        "time": f"{payload['dateOfContact']} {payload['timeOfContact']}",
        "sendsMessage": draft["channel"] in ("sms", "email"),
    }
    pending = _pending(turn, "add_log", order, payload, details)
    await turn.emitter.step("awaiting_confirmation", "waiting", f"Confirm the {channel_label} log")
    return _ask_confirmation(turn, pending, f"Add this {channel_label} log to order {details['order']}?")


@agent_node("list_logs")
async def list_logs(turn: Turn) -> dict[str, Any]:
    update = turn.update
    order = await resolve_order(turn, update)
    async with turn.step("load_logs", "Loading contact logs", "Unable to load the contact logs.") as rec:
        page = await turn.scheduling.contact_logs(order["id"])
        rec.message = f"{page.total} log(s)"
    update["logs"] = {"orderId": order["id"], "orderLabel": order.get("jobNumber") or order.get("policyNumber"),
                      "items": [domain.log_summary(r) for r in page.records], "total": page.total}
    turn.show("logs")
    return update


# --- reference data -----------------------------------------------------------------

@agent_node("list_reference")
async def list_reference(turn: Turn) -> dict[str, Any]:
    kind = turn.u.get("reference_kind")
    if kind not in domain.REFERENCE_TITLES:
        raise NeedInput("Which list do you need: service types, programs, time zones, order statuses, sub-statuses or providers?")
    async with turn.step("load_lookups", f"Loading {domain.REFERENCE_TITLES[kind].lower()}", "Unable to load that list.") as rec:
        if kind == "service_types":
            rows = await turn.orders.service_types()
            items = sorted({f"{r.get('text')}" + (f" ({r['categoryName'].strip()})" if r.get("categoryName") else "") for r in rows if r.get("text")})
        elif kind == "programs":
            rows = await turn.orders.programs()
            items = sorted({f"{r.get('text')}" + (f" [{r['aliasName']}]" if r.get("aliasName") else "") for r in rows if r.get("text")})
        elif kind == "time_zones":
            items = await turn.orders.time_zones()
        elif kind == "order_statuses":
            rows = await turn.orders.order_statuses()
            items = sorted({str(r["name"]) for r in rows if r.get("name") and r.get("isActive", True)})
        elif kind == "sub_statuses":
            rows = await turn.orders.order_sub_statuses()
            items = sorted({f"{r.get('orderStatusName')}: {r.get('name')}" for r in rows if r.get("name")})
        else:
            rows = await turn.orders.providers()
            items = sorted({str(r["text"]) for r in rows if r.get("text")})
        rec.message = f"{len(items)} item(s)"
    turn.show("reference")
    return {"reference": {"kind": kind, "title": domain.REFERENCE_TITLES[kind], "items": items[:200], "total": len(items)}}


# --- confirmation + execution ---------------------------------------------------------


@agent_node("execute_action")
async def execute_action(turn: Turn) -> dict[str, Any]:
    pending = turn.state.get("pending_action")
    action_id = turn.u.get("action_id")
    if not pending:
        return {"notices": [*(turn.state.get("notices") or []), "There is nothing waiting for confirmation."]}
    if action_id and action_id != pending["id"]:
        raise Blocked("That confirmation is for an older request, so nothing was changed. Please review the latest one.")
    if pending.get("askedRunId") == turn.run_id or pending.get("type") not in _WRITE_TYPES:
        raise Blocked("Nothing was changed: the action must be confirmed after you've seen it.")
    if pending["type"] == "assign_batch":
        return await _execute_batch(turn, pending)

    update = turn.update
    update.update({"pending_action": None, "pending_question": None})
    order = await _load_order(turn, pending["orderId"], "recheck_order")
    update["selected_order"] = order
    if pending["type"] in ("assign", "schedule"):
        reason = domain.eligibility(order, pending["type"])
        if reason:
            raise Blocked(f"Nothing was changed: the order changed since you confirmed ({reason}).", update=update)
    if pending["type"] == "assign":
        now = (domain.assigned_producer(order) or {}).get("id")
        if now == pending["producerId"]:  # already done (e.g. a repeated confirmation): never post twice
            update["action_result"] = {"type": "assign", "status": "success", "verified": True,
                                       "message": "The order was already assigned to this producer.",
                                       "details": pending["details"]}
            turn.show("order")
            return update
        if now != pending.get("expectedAssigneeId"):
            raise Blocked("Nothing was changed: the order's producer changed since you confirmed. Please review it again.",
                          update=update)
    if pending["type"] == "schedule":
        details = pending["details"]
        producer = pending["producer"]
        recheck = await _availability(turn, order, producer, [date.fromisoformat(details["date"])], "recheck_calendar")
        if domain.slot_is_open(recheck["days"], details["date"], details["start"]) is None:
            update["availability"] = await _availability(turn, order, producer, _window(turn), "load_calendar")
            turn.show("availability")
            raise NeedInput(
                f"Nothing was booked: {details['start']} on {details['date']} is no longer open for {producer['name']}. "
                "Please pick another slot.", kind="choose_slot", update=update,
            )

    step_id = "create_contact_log" if pending["type"] == "add_log" else "assign_or_schedule"
    label = {"assign": "Assigning the order", "schedule": "Booking the appointment", "add_log": "Adding the log"}[pending["type"]]
    outcome = "success"
    message = ""
    try:
        async with turn.step(step_id, label, "The request was not completed.") as rec:
            if pending["type"] == "add_log":
                response = await turn.scheduling.create_contact_log(pending["payload"])
            else:
                response = await turn.scheduling.assign_or_schedule(pending["payload"])
            message = (response.get("message") or "").strip() if isinstance(response, dict) else ""
            rec.message = message or "Accepted by BOOTOG"
    except BootogResultError as exc:
        update["action_result"] = {"type": pending["type"], "status": "failed", "message": exc.message, "details": pending["details"]}
        update.update({"execution_status": "failed", "error": exc.message})
        return update
    except BootogApiError as exc:
        if not exc.outcome_unknown:
            update["action_result"] = {"type": pending["type"], "status": "failed", "message": exc.message, "details": pending["details"]}
            update.update({"execution_status": "failed", "error": exc.message})
            return update
        outcome = "unknown"
        message = exc.message

    verified = await _verify(turn, pending, update)
    if outcome == "unknown" and verified:
        outcome = "success"
    if pending["type"] == "assign" and outcome == "success" and not verified:
        outcome = "unverified"  # accepted by BOOTOG, but the order doesn't show the producer yet
    update["action_result"] = {
        "type": pending["type"], "status": outcome, "verified": verified, "message": message, "details": pending["details"],
    }
    if pending["type"] == "add_log":
        update["log_draft"] = None
        turn.show("logs")
    turn.show("order")
    return update


async def _verify(turn: Turn, pending: dict[str, Any], update: dict[str, Any]) -> bool:
    """Reads the result back: the order's services carry the producer (and the booked time),
    or the order's newest log matches the one added."""
    try:
        async with turn.step("verify_action", "Verifying the change in BOOTOG", "Unable to verify the change.") as rec:
            if pending["type"] == "add_log":
                page = await turn.scheduling.contact_logs(pending["orderId"], page_size=5)
                items = [domain.log_summary(r) for r in page.records]
                wanted = (pending["payload"].get("emailContent") or "").strip()
                ok = any(r.get("contactType") == pending["payload"]["contactType"]
                         and (r.get("emailContent") or "").strip() == wanted for r in page.records)
                update["logs"] = {"orderId": pending["orderId"], "orderLabel": pending.get("orderLabel"), "items": items,
                                  "total": page.total}
            else:
                order = domain.order_summary(await turn.orders.get_order(pending["orderId"]))
                update["selected_order"] = order
                services = order.get("services") or []
                ok = any(s.get("assignedToId") == pending["producerId"] for s in services)
                if pending["type"] == "schedule" and ok:
                    details = pending["details"]
                    ok = any(
                        s.get("assignedToId") == pending["producerId"]
                        and domain.parse_day(s.get("scheduleDate")) == date.fromisoformat(details["date"])
                        and domain.parse_clock(s.get("scheduleTime")) == domain.parse_clock(details["start"])
                        for s in services
                    )
            rec.message = "Change confirmed in BOOTOG" if ok else "The change is not visible in BOOTOG yet"
        return ok
    except BootogApiError:
        return False


@agent_node("cancel_action")
async def cancel_action(turn: Turn) -> dict[str, Any]:
    had = turn.state.get("pending_action") or turn.state.get("log_draft") or turn.state.get("batch_draft")
    text = "Cancelled — nothing was changed." if had else "There was nothing in progress to cancel."
    return {"pending_action": None, "log_draft": None, "batch_draft": None, "pending_question": None, "notices": [text]}


# --- multi-order assignment -----------------------------------------------------------------
# A batch is the orders of the last list (or of the last batch, for "retry the failed ones")
# with a producer per position. Every order goes through the same checks as a single
# assignment; one confirmation covers the ready ones; each is then written and verified on
# its own with the single-assignment guards.

_BATCH_DONE = {"success"}


def _order_ref(order: dict[str, Any]) -> dict[str, Any]:
    return {k: order.get(k) for k in ("id", "jobNumber", "policyNumber", "customer")}


def _batch_source(turn: Turn) -> dict[str, Any]:
    """The batch to work on: the draft being prepared, or a new one from the last order list
    (or from the last batch's unfinished orders when retrying)."""
    listing = turn.state.get("orders") or {}
    listing_key = [o.get("id") for o in listing.get("items") or []]
    draft = turn.state.get("batch_draft")
    if turn.u.get("retry_failed"):
        last = (turn.state.get("last_batch") or {}).get("items") or []
        todo = [i for i in last if i.get("result") not in _BATCH_DONE]
        if not todo:
            raise NeedInput("There is nothing to retry: every order of the last batch was assigned.", kind="batch_mapping")
        return {"orders": [{k: i.get(k) for k in ("id", "jobNumber", "policyNumber", "customer")} for i in todo],
                "mapping": {n: {"producer_name": i["producer"]} for n, i in enumerate(todo, 1) if i.get("producer")},
                "selected": list(range(1, len(todo) + 1)), "listingKey": None}
    if draft and draft.get("listingKey") in (listing_key, None):
        return draft
    if not listing_key:
        raise NeedInput("Which orders? Show a list first, e.g. \"show me the last 7 days' new orders\".", kind="batch_mapping")
    if listing.get("truncated"):
        raise Blocked("The last order list is partial, so I can't assign \"these orders\" safely. Please narrow the search.")
    return {"orders": [_order_ref(o) for o in listing["items"]], "mapping": {}, "selected": None, "listingKey": listing_key}


def _batch_rows(draft: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for n in draft["selected"]:
        ref = draft["orders"][n - 1]
        spec = draft["mapping"].get(n) or {}
        rows.append({"position": n, "orderId": ref["id"], "jobNumber": ref.get("jobNumber"), "policyNumber": ref.get("policyNumber"),
                     "customer": ref.get("customer"), "producer": spec.get("producer_name") or ("Nearest eligible" if spec.get("nearest") else None),
                     "status": "missing" if not spec else "pending", "reason": None if spec else "No producer chosen yet"})
    return rows


def _positions_text(positions: list[int]) -> str:
    return ", ".join(f"#{p}" for p in positions[:15]) + (" …" if len(positions) > 15 else "")


async def _check_batch_item(turn: Turn, row: dict[str, Any], spec: dict[str, Any], limit: float) -> dict[str, Any]:
    """One order of a batch through the single-assignment checks: fresh order, status,
    current producer, coordinates, service filters, producers within the limit, ranking.
    Never substitutes another producer: a problem is reported on the row instead."""
    row = {**row, "status": "attention"}
    try:
        order = await _load_order(turn, row["orderId"])
        row.update({"jobNumber": order.get("jobNumber"), "policyNumber": order.get("policyNumber"),
                    "customer": order.get("customer")})
        reason = domain.eligibility(order, "assign")
        if reason:
            return {**row, "reason": f"Can't be assigned: {reason}"}
        current = domain.assigned_producer(order)
        if current and not turn.u.get("reassign"):
            same = spec.get("producer_name") and domain.match_producers([{"name": current.get("name") or ""}], spec["producer_name"])
            if same:
                return {**row, "status": "done", "producer": current.get("name"), "reason": "Already assigned to this producer"}
            return {**row, "reason": f"Already assigned to {current.get('name') or 'a producer'} — say \"reassign\" to replace"}
        if not domain.has_coordinates(order):
            return {**row, "reason": "The property has no coordinates, so distance can't be checked"}
        _, filters = await _service_requirements(turn, order)
        filters_sent = bool(filters["skillsetIds"] or filters["credentialIds"] or order.get("programId"))
        if spec.get("nearest"):
            chosen, radius = await _nearest_eligible(turn, order, filters, filters_sent, current, limit)
            if chosen is None:
                searched = f" (searched {limit:g}–{radius:g} miles)" if radius > limit else ""
                return {**row, "reason": f"No eligible producer within {radius:g} miles{searched}"}
            if radius > limit:
                row["reason"] = (f"No eligible producer within {limit:g} miles. "
                                 f"Found an eligible producer {chosen.get('distanceMiles')} miles away.")
            return {**row, "status": "ready", "producer": chosen["name"], "producerId": chosen["id"],
                    "distanceMiles": chosen.get("distanceMiles"), "searchRadiusMiles": radius,
                    "replaces": (current or {}).get("name"), "expectedAssigneeId": (current or {}).get("id"),
                    "payload": domain.assign_payload(order, chosen)}
        producers = await _load_producers(turn, order, limit, filters)
        ranking = domain.rank_producers(producers, max_distance_miles=limit, assigned_id=(current or {}).get("id"),
                                        filters_sent=filters_sent)
        hits = domain.match_producers(ranking["items"], spec["producer_name"])
        if not hits:
            return {**row, "reason": f"{spec['producer_name']} isn't among this order's producers"}
        if len(hits) > 1:
            return {**row, "reason": f"{len(hits)} producers match {spec['producer_name']} — use the full name"}
        chosen = hits[0]
        if not chosen.get("eligible"):
            return {**row, "producer": chosen["name"], "reason": f"{chosen['name']}: {chosen.get('excludedReason')}"}
        return {**row, "status": "ready", "reason": None, "producer": chosen["name"], "producerId": chosen["id"],
                "distanceMiles": chosen.get("distanceMiles"), "replaces": (current or {}).get("name"),
                "expectedAssigneeId": (current or {}).get("id"), "payload": domain.assign_payload(order, chosen)}
    except (Blocked, NeedInput) as exc:
        return {**row, "reason": str(exc)}
    except BootogApiError as exc:
        return {**row, "reason": exc.message}


async def _nearest_eligible(
    turn: Turn, order: dict[str, Any], filters: dict[str, list[str]], filters_sent: bool,
    current: dict[str, Any] | None, limit: float,
) -> tuple[dict[str, Any] | None, float]:
    """The recommended producer for one order: the same search and ranking as at `limit`,
    repeated at a wider radius only while nothing eligible is found. Every rule (service
    filters, matching, distance, ranking) applies at each radius. A distance the user stated
    ("within 30 miles") is respected: no expansion beyond it. Returns (producer, last radius)."""
    settings = get_settings()
    radii = [limit] if turn.u.get("max_distance_miles") else domain.search_radii(
        limit, float(settings.scheduling_radius_step_miles), float(settings.scheduling_max_radius_miles))
    for radius in radii:
        producers = await _load_producers(turn, order, radius, filters)
        ranking = domain.rank_producers(producers, max_distance_miles=radius, assigned_id=(current or {}).get("id"),
                                        filters_sent=filters_sent)
        if ranking["recommended"]:
            return ranking["recommended"], radius
    return None, radii[-1]


@agent_node("prepare_batch_assignment")
async def prepare_batch_assignment(turn: Turn) -> dict[str, Any]:
    """"Assign these orders …": map the user's groups onto the listed orders, ask for any
    missing producer, check every order like a single assignment, then ask ONE confirmation
    for all the ready ones."""
    update = turn.update
    draft = _batch_source(turn)
    groups = turn.u.get("assignments") or []
    listed = len(draft["orders"])
    mapping, invalid = domain.batch_mapping(listed, groups, draft.get("mapping"))
    selected = domain.batch_selection(listed, groups, draft.get("selected"))
    count = len(selected)
    max_orders = get_settings().scheduling_max_batch_orders
    if count > max_orders:
        raise Blocked(f"That selects {count} orders; one batch can cover at most {max_orders}. "
                      "Please pick fewer (e.g. \"the first 10\") or narrow the search.")
    draft = {**draft, "mapping": {p: spec for p, spec in mapping.items() if p in selected}, "selected": selected}
    mapping = draft["mapping"]
    update.update({"batch_draft": draft, "pending_action": None})
    rows = _batch_rows(draft)
    update["batch_view"] = {"rows": rows, "validated": False}
    turn.show("batch")
    if invalid:
        raise NeedInput(f"The list has {listed} orders, so there is no order {_positions_text(invalid)}.", kind="batch_mapping",
                        update=update)
    missing = [r["position"] for r in rows if r["status"] == "missing"]
    if missing:
        text = (f"Which producer should each of these {count} selected orders go to? For example \"the first 4 to <name>, the rest to "
                "<name>\" or \"the nearest eligible producers\"." if len(missing) == count else
                f"Orders {_positions_text(missing)} have no producer yet — who should take them?")
        raise NeedInput(text, kind="batch_mapping", update=update)

    limit = _distance_limit(turn)
    checked = [await _check_batch_item(turn, row, mapping[row["position"]], limit) for row in rows]
    ready = [r for r in checked if r["status"] == "ready"]
    attention = [r for r in checked if r["status"] == "attention"]
    update["batch_view"] = {"rows": [{k: v for k, v in r.items() if k not in ("payload", "expectedAssigneeId")} for r in checked],
                            "validated": True, "maxDistanceMiles": limit}
    if not ready:
        raise NeedInput("None of these orders can be assigned as mapped — see the reasons in the table. "
                        "Change the producers for those orders or cancel.", kind="batch_mapping", update=update)
    pending = {
        "id": uuid.uuid4().hex, "type": "assign_batch", "orderId": None, "orderLabel": None, "payload": None,
        "items": checked, "askedRunId": turn.run_id,
        "details": {"orders": count, "ready": len(ready), "attention": len(attention),
                    "done": len([r for r in checked if r["status"] == "done"])},
    }
    await turn.emitter.step("awaiting_confirmation", "waiting", f"Confirm {len(ready)} assignment(s)")
    skipped = f" {len(attention)} need attention and will not be assigned unless you change them." if attention else ""
    return _ask_confirmation(turn, pending, f"Assign all {len(ready)} ready order(s) as shown?{skipped}")


def _result_row(item: dict[str, Any], result: str, message: str | None = None, verified: bool = False) -> dict[str, Any]:
    return {**{k: item.get(k) for k in ("position", "orderId", "jobNumber", "policyNumber", "customer", "producer",
                                          "distanceMiles")},
            "id": item.get("orderId"), "result": result, "verified": verified, "message": message}


async def _assign_one(turn: Turn, item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Writes one order of a confirmed batch with the single-assignment guards: re-check the
    order, never post twice, post once, read back. Returns (result row, stop the batch)."""
    label = item.get("jobNumber") or item.get("policyNumber") or "the order"
    try:
        order = await _load_order(turn, item["orderId"], "recheck_order")
    except BootogApiError as exc:
        return _result_row(item, "failed", f"Couldn't re-check the order: {exc.message}"), exc.status_code in (401, 403)
    reason = domain.eligibility(order, "assign")
    if reason:
        return _result_row(item, "skipped", f"Not assigned — the order changed ({reason})"), False
    now = (domain.assigned_producer(order) or {}).get("id")
    if now == item["producerId"]:
        return _result_row(item, "success", "Already assigned to this producer", verified=True), False
    if now != item.get("expectedAssigneeId"):
        return _result_row(item, "skipped", "Not assigned — the order's producer changed since you confirmed"), False
    outcome, message = "success", ""
    try:
        async with turn.step("assign_or_schedule", f"Assigning {label} to {item['producer']}", "The request was not completed.") as rec:
            response = await turn.scheduling.assign_or_schedule(item["payload"])
            message = (response.get("message") or "").strip() if isinstance(response, dict) else ""
            rec.message = message or "Accepted by BOOTOG"
    except BootogResultError as exc:
        return _result_row(item, "failed", exc.message), False
    except BootogApiError as exc:
        if not exc.outcome_unknown:
            return _result_row(item, "failed", exc.message), exc.status_code in (401, 403)
        outcome, message = "unknown", exc.message
    verified = await _verify(turn, {"type": "assign", "orderId": item["orderId"], "producerId": item["producerId"]}, {})
    if verified:
        return _result_row(item, "success", None, verified=True), False
    if outcome == "unknown":
        return _result_row(item, "unknown", f"Outcome not confirmed: {message}"), False
    return _result_row(item, "unverified", "Accepted by BOOTOG, but the order doesn't show the producer yet"), False


async def _execute_batch(turn: Turn, pending: dict[str, Any]) -> dict[str, Any]:
    update = turn.update
    update.update({"pending_action": None, "pending_question": None, "batch_draft": None})
    results, stop = [], None
    for item in pending["items"]:
        if item["status"] == "done":
            results.append(_result_row(item, "success", item.get("reason"), verified=True))
        elif item["status"] != "ready":
            results.append(_result_row(item, "not_processed", item.get("reason")))
        elif stop:
            results.append(_result_row(item, "not_processed", stop))
        else:
            row, halt = await _assign_one(turn, item)
            results.append(row)
            if halt:
                stop = "Not processed — BOOTOG refused access, so the remaining orders were not sent"
    totals = {
        "selected": len(results),
        "succeeded": sum(r["result"] == "success" for r in results),
        "failed": sum(r["result"] in ("failed", "skipped") for r in results),
        "needsCheck": sum(r["result"] in ("unknown", "unverified") for r in results),
        "notProcessed": sum(r["result"] == "not_processed" for r in results),
    }
    status = "success" if totals["succeeded"] == totals["selected"] else "failed" if not totals["succeeded"] else "partial"
    update["action_result"] = {"type": "assign_batch", "status": status, "verified": status == "success",
                               "message": None, "items": results, "totals": totals, "details": {"orders": len(results)}}
    update["last_batch"] = {"items": results}
    return update


# --- reply ----------------------------------------------------------------------------


def _pending_view(pending: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pending:
        return None
    return {"type": pending["type"], **{k: v for k, v in pending["details"].items() if v not in (None, "")}}


def build_facts(state: SchedulingChatState) -> dict[str, Any]:
    shown = set(state.get("shown") or [])
    facts: dict[str, Any] = {
        "agent": "Scheduling Agent",
        "intent": state.get("intent"),
        "request": (state.get("understanding") or {}).get("summary"),
    }
    if state.get("intent") in ("greeting_or_help", "other", None):
        facts["capabilities"] = CAPABILITIES
    if "summary" in shown and state.get("summary"):
        s = state["summary"]
        facts["order_status_summary"] = {g["title"]: {i["label"]: i["count"] for i in g["items"]} for g in s["groups"]}
        if s.get("asked"):
            facts["asked_statuses"] = s["asked"]
    if "orders" in shown and state.get("orders"):
        o = state["orders"]
        facts["orders"] = {
            "context": o["context"], "description": o["description"], "total": o["total"], "page": o["page"],
            "pages": -(-o["total"] // o["pageSize"]) if o["pageSize"] else 1, "truncated_search": o.get("truncated"),
            "first_items": [
                {"jobNumber": i.get("jobNumber"), "policyNumber": i.get("policyNumber"), "customer": i.get("customer"),
                 "service": i.get("service"), "status": domain.order_status_text(i)}
                for i in o["items"][:5]
            ],
        }
    order = state.get("selected_order")
    if order and shown & {"order", "producers", "availability", "pending", "logs"}:
        facts["order"] = {
            "jobNumber": order.get("jobNumber"), "policyNumber": order.get("policyNumber"), "service": order.get("service"),
            "customer": order.get("customer"), "address": order.get("address"), "status": domain.order_status_text(order),
            "assigned_to": (domain.assigned_producer(order) or {}).get("name"),
            "scheduled": [f"{s.get('scheduleDate', '')[:10]} {s.get('scheduleTime')}-{s.get('scheduleEndTime')}"
                          for s in order.get("services") or [] if s.get("scheduleDate")],
        }
    if "producers" in shown and state.get("producers"):
        p = state["producers"]
        facts["producers"] = {"total": p.get("total"), "first": [
            {"name": x["name"], "distanceMiles": x.get("distanceMiles"), "rating": x.get("rating"), "assigned": x.get("assigned")}
            for x in p["items"][:5]
        ]}
    if "ranking" in shown and state.get("ranking"):
        r = state["ranking"]
        facts["producer_ranking"] = {
            "rule": r["rule"], "maxDistanceMiles": r["maxDistanceMiles"], "eligible": r["eligibleCount"],
            "withinDistance": r["withinDistanceCount"], "recommended": (r.get("recommended") or {}).get("name"),
            "top": [{"name": x["name"], "distanceMiles": x.get("distanceMiles"), "eligible": x["eligible"],
                     "why_not": x.get("excludedReason")} for x in r["items"][:5]],
        }
    if "availability" in shown and state.get("availability"):
        a = state["availability"]
        facts["availability"] = {
            "producer": a["producer"]["name"], "durationMinutes": a["durationMinutes"],
            "days": [{"date": d["date"], "weekday": d["weekday"], "open": len(d["slots"]),
                      "first": [s["start"] for s in d["slots"][:3]], "note": d.get("note")} for d in a["days"]],
        }
    if "logs" in shown and state.get("logs"):
        lg = state["logs"]
        facts["logs"] = {"total": lg["total"], "latest": lg["items"][:3]}
    if "reference" in shown and state.get("reference"):
        r = state["reference"]
        facts["reference"] = {"title": r["title"], "total": r["total"], "first": r["items"][:8]}
    if "batch" in shown and state.get("batch_view"):
        facts["batch"] = {"validated": state["batch_view"].get("validated"), "rows": [
            {k: r.get(k) for k in ("position", "jobNumber", "customer", "producer", "status", "reason")}
            for r in state["batch_view"]["rows"]
        ]}
    if "pending" in shown and state.get("pending_action"):
        facts["pending_action"] = _pending_view(state["pending_action"])
    if state.get("action_result"):
        facts["action_result"] = state["action_result"]
    if state.get("question"):
        facts["question"] = state["question"]["text"]
    if state.get("notices"):
        facts["notes"] = state["notices"]
    if state.get("error"):
        facts["error"] = state["error"]
    return facts


def fallback_reply(state: SchedulingChatState) -> str:
    """Used when the LLM can't write the reply: plain text from the same facts."""
    if state.get("error"):
        return state["error"]
    result = state.get("action_result")
    if result and result["type"] == "assign_batch":
        t = result["totals"]
        return (f"{t['succeeded']} of {t['selected']} order(s) assigned and verified; {t['failed']} failed, "
                f"{t['needsCheck']} need checking, {t['notProcessed']} not processed.")
    if result:
        what = {"assign": "assignment", "schedule": "appointment", "add_log": "log"}[result["type"]]
        if result["status"] == "success":
            return f"Done — the {what} was saved in BOOTOG."
        if result["status"] == "unverified":
            return f"BOOTOG accepted the {what}, but the order doesn't show it yet, so I can't confirm it. Please check the order."
        if result["status"] == "unknown":
            return f"I couldn't confirm whether the {what} was saved. Please check the order before trying again."
        return f"The {what} was not saved: {result.get('message')}"
    if state.get("question"):
        return state["question"]["text"]
    if state.get("notices"):
        return " ".join(state["notices"])
    shown = state.get("shown") or []
    if "orders" in shown and state.get("orders"):
        return f"I found {state['orders']['total']} order(s) — {state['orders']['description']}."
    if "summary" in shown:
        return "Here is the current order status summary."
    return "I can help with order counts, finding orders, assigning and scheduling inspections, and contact logs."


async def respond(state: SchedulingChatState, config: RunnableConfig) -> dict[str, Any]:
    emitter: ChatEmitter = config["configurable"]["emitter"]
    status = state.get("execution_status")
    awaiting = bool(state.get("awaiting_user_input"))
    workflow = "failed" if status == "failed" else "blocked" if status == "blocked" else "waiting_for_user" if awaiting else "completed"
    merged = {**state, "workflow_status": workflow}
    try:
        reply = await compose_reply(config, state.get("conversation_history") or [], build_facts(merged))  # type: ignore[arg-type]
    except Exception as exc:  # the turn's real results stand; keep details in logs
        logger.warning("[%s] run=%s reply generation failed: %s", AGENT, state.get("run_id"), type(exc).__name__)
        reply = ""
    reply = reply or fallback_reply(merged)  # type: ignore[arg-type]
    if awaiting and (state.get("question") or {}).get("type") != "confirm":  # confirmations emit their own step
        await emitter.step("awaiting_user", "waiting", "Waiting for your reply")
    return {
        "workflow_status": workflow,
        "agent_response": reply,
        "conversation_history": [{"role": "assistant", "content": reply}],
    }
