"""Scheduling Agent — LangGraph chat over the Orders module (Orders · Assign · Schedule ·
Add Log). One chat session = one checkpointed thread; one message = one turn:

    START -> start_turn -> understand_message (LLM, or a card's structured action)
          -> one tool node chosen from the intent:
               order_summary | list_orders | order_details | find_producers
               prepare_assignment | check_availability | prepare_schedule
               prepare_log | list_logs | list_reference | execute_action | cancel_action
          -> respond (LLM reply from the turn's facts) -> END

The three workflows share the conversation's selected order: an order found or listed
is the one assigned, scheduled (with the assigned producer by default) and logged.
Writes only run from `execute_action`, after the user confirmed the pending action.

Session bookkeeping (owner, one turn at a time, expiry), chat history and the run
event channel are the shared chat infrastructure of the Contact agents."""

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.contact_us.agent_graph import ChatSessions
from app.agents.contact_us.agent_nodes import ChatEmitter
from app.agents.runtime import RunChannel
from app.agents.scheduling import domain, nodes
from app.agents.scheduling.nodes import ACTIVITY_TITLES
from app.agents.scheduling.state import SchedulingChatState
from app.core.logging import get_logger
from app.integrations.bootog import BootogClient

logger = get_logger(__name__)

AGENT_KEY = "scheduling"

TOOLS = {
    "order_summary": nodes.order_summary,
    "list_orders": nodes.list_orders,
    "order_details": nodes.order_details,
    "find_producers": nodes.find_producers,
    "prepare_assignment": nodes.prepare_assignment,
    "prepare_batch_assignment": nodes.prepare_batch_assignment,
    "check_availability": nodes.check_availability,
    "prepare_schedule": nodes.prepare_schedule,
    "prepare_log": nodes.prepare_log,
    "list_logs": nodes.list_logs,
    "list_reference": nodes.list_reference,
    "execute_action": nodes.execute_action,
    "cancel_action": nodes.cancel_action,
}

INTENT_TOOLS = {
    "order_summary": "order_summary",
    "list_orders": "list_orders",
    "find_order": "order_details",
    "order_details": "order_details",
    "find_producers": "find_producers",
    "assign_order": "prepare_assignment",
    "assign_orders": "prepare_batch_assignment",
    "check_availability": "check_availability",
    "schedule_order": "prepare_schedule",
    "add_log": "prepare_log",
    "list_logs": "list_logs",
    "list_reference": "list_reference",
    "confirm_action": "execute_action",
    "cancel_action": "cancel_action",
}


def route(state: SchedulingChatState) -> str:
    if state.get("execution_status") in ("failed", "blocked"):
        return "respond"
    return INTENT_TOOLS.get(state.get("intent") or "", "respond")


def build_scheduling_graph(checkpointer=None):
    graph = StateGraph(SchedulingChatState)
    graph.add_node("start_turn", nodes.start_turn)
    graph.add_node("understand_message", nodes.understand_message)
    for name, fn in TOOLS.items():
        graph.add_node(name, fn)
    graph.add_node("respond", nodes.respond)
    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "understand_message")
    graph.add_conditional_edges("understand_message", route, [*TOOLS, "respond"])
    for name in TOOLS:
        graph.add_edge(name, "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)


# In-memory checkpointer, like the Contact agents: sessions live as long as the process;
# conversations reopened from chat history are re-seeded from their stored messages.
CHECKPOINTER = InMemorySaver()
SCHEDULING_GRAPH = build_scheduling_graph(CHECKPOINTER)
sessions = ChatSessions(CHECKPOINTER)

_RUN_STATUS = {"failed": "failed", "blocked": "blocked", "waiting_for_user": "waiting"}


# --- result for the UI ------------------------------------------------------------------


def _order_card(order: dict[str, Any]) -> dict[str, Any]:
    return {
        **{k: order.get(k) for k in ("id", "jobNumber", "policyNumber", "service", "programName", "customer", "contactPhone",
                                     "contactEmail", "address", "providerName", "createdDateTime", "isRushOrder")},
        "status": domain.order_status_text(order),
        "assignedTo": (domain.assigned_producer(order) or {}).get("name"),
        "services": [
            {k: s.get(k) for k in ("id", "name", "status", "subStatus", "assignedTo", "scheduleDate", "scheduleTime", "scheduleEndTime")}
            for s in order.get("services") or []
        ],
    }


def _producer_card(p: dict[str, Any]) -> dict[str, Any]:
    return {k: p.get(k) for k in ("id", "name", "email", "phone", "address", "distanceMiles", "rating", "ratingCount",
                                  "openAssignments", "scheduledAssignments", "groups", "assigned")}


def _ranking_card(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "orderId": r["orderId"], "maxDistanceMiles": r["maxDistanceMiles"], "rule": r["rule"],
        "eligibleCount": r["eligibleCount"], "withinDistanceCount": r["withinDistanceCount"], "filtersSent": r["filtersSent"],
        "recommendedId": (r.get("recommended") or {}).get("id"),
        "items": [{**_producer_card(p), **{k: p.get(k) for k in ("rank", "eligible", "excludedReason", "recommended", "isCurrentProducer")}}
                  for p in r["items"][:15]],
        "total": len(r["items"]),
    }


def chat_result(state: SchedulingChatState) -> dict[str, Any]:
    shown = set(state.get("shown") or [])
    order = state.get("selected_order")
    pending = state.get("pending_action") if "pending" in shown else None
    orders = state.get("orders") if "orders" in shown else None
    producers = state.get("producers") if "producers" in shown else None
    availability = state.get("availability") if "availability" in shown else None
    return {
        "sessionId": state.get("session_id"),
        "agentKey": AGENT_KEY,
        "reply": state.get("agent_response"),
        "workflowStatus": state.get("workflow_status"),
        "intent": state.get("intent"),
        "awaitingUserInput": bool(state.get("awaiting_user_input")),
        "pendingQuestion": state.get("question"),
        "summary": state.get("summary") if "summary" in shown else None,
        "orders": {
            **{k: orders.get(k) for k in ("total", "page", "pageSize", "context", "description", "truncated")},
            "items": [_order_card(o) for o in orders.get("items") or []],
        } if orders else None,
        "order": _order_card(order) if order and shown & {"order", "producers", "availability", "pending", "logs"} else None,
        "producers": {"orderId": producers["orderId"], "total": producers.get("total"),
                      "items": [_producer_card(p) for p in producers["items"]]} if producers else None,
        "ranking": _ranking_card(state.get("ranking")) if "ranking" in shown and state.get("ranking") else None,
        "availability": {k: availability.get(k) for k in ("orderId", "producer", "durationMinutes", "days", "hasWorkingHours")}
        if availability else None,
        "pendingAction": {"id": pending["id"], "type": pending["type"], "orderId": pending["orderId"],
                          "details": pending["details"]} if pending else None,
        "actionResult": state.get("action_result"),
        "batch": state.get("batch_view") if "batch" in shown else None,
        "logs": state.get("logs") if "logs" in shown else None,
        "reference": state.get("reference") if "reference" in shown else None,
    }


# --- running a turn -----------------------------------------------------------------------


async def run_scheduling_chat_turn(
    channel: RunChannel,
    session_id: str,
    message: str | None,
    client: BootogClient,
    action: dict[str, Any] | None = None,
    utc_offset_minutes: int = 0,
    graph=None,
    llm=None,
) -> SchedulingChatState:
    """One turn, streaming activity into `channel`. Always closes the channel and client."""
    graph = graph or SCHEDULING_GRAPH
    emitter = ChatEmitter(channel, titles=ACTIVITY_TITLES)
    configurable: dict[str, Any] = {"thread_id": session_id, "bootog_client": client, "emitter": emitter}
    if llm is not None:
        configurable["llm"] = llm
    config: RunnableConfig = {"configurable": configurable, "recursion_limit": 25}
    inputs: SchedulingChatState = {
        "run_id": channel.run_id,
        "session_id": session_id,
        "current_user_message": message,
        "action": action,
        "utc_offset_minutes": utc_offset_minutes,
        "conversation_history": [{"role": "user", "content": message}] if message else [],
    }
    try:
        state = await graph.ainvoke(inputs, config=config)
    except Exception as exc:  # graph-level crash: never report success
        logger.exception("[SCHEDULING_AGENT] session=%s run=%s crashed: %s", session_id, channel.run_id, type(exc).__name__)
        reply = "Something went wrong while I was working on that. Please try again."
        state = {**inputs, "workflow_status": "failed", "agent_response": reply}
        try:
            await graph.aupdate_state(
                config, {"conversation_history": [{"role": "assistant", "content": reply}], "workflow_status": "failed"},
                as_node="respond",
            )
        except Exception:
            logger.warning("[SCHEDULING_AGENT] session=%s could not record the failure", session_id)
    finally:
        await client.aclose()

    status = _RUN_STATUS.get(state.get("workflow_status") or "", "completed")
    await emitter.finish(status, state.get("agent_response") or "", chat_result(state))
    return state


async def _run_in_session(channel, session_id, message, client, action, utc_offset_minutes) -> None:
    try:
        await run_scheduling_chat_turn(channel, session_id, message, client, action, utc_offset_minutes)
    finally:
        sessions.end(session_id)


def start_turn_task(
    channel: RunChannel,
    session_id: str,
    message: str | None,
    client: BootogClient,
    action: dict[str, Any] | None = None,
    utc_offset_minutes: int = 0,
) -> asyncio.Task:
    return asyncio.create_task(_run_in_session(channel, session_id, message, client, action, utc_offset_minutes))
