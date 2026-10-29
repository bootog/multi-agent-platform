"""Frontend Order Enquiry Agent — LangGraph workflow.

START -> initialize_run -> understand_request -> resolve_scope
      ├─ no provider selected -> summarize_providers ─────────────────────────────┐
      └─ provider selected    -> fetch_enquiries -> validate_enquiries            │
                                 -> select_enquiries -> [check_follow_up]         │
                                 -> prepare_enquiries ────────────────────────────┤
                                                                     respond -> END

- understand_request: chat message -> action, dates, provider NAME, page (LLM + domain.py dates).
- resolve_scope: merges that with the conversation's previous scope and turns a
  provider name into a verified provider id (Bootog's provider list). Ambiguous or
  unknown names end the run with a question; no enquiry API is called for them.
- summarize_providers (Stage A): one GetAllOrderEnquery call -> providers ranked by
  their number of matching enquiries. No records, no per-provider calls.
- fetch_enquiries .. prepare_enquiries (Stage B): the selected provider's enquiries,
  one page, scoped to converted (orderAction 6) or not-converted enquiries.
- respond: builds the reply and the result; every path ends here (clarification,
  failure and blocked included).

One conversation = one checkpointed LangGraph thread (history.py); one chat message =
one run. `scope` survives between messages; every other state key is reset per run.
Each message and reply is also saved to the PostgreSQL chat history (history.py).

Extending: later stages (customer follow-up, Contact Log update, convert to order) are
added as new nodes after `prepare_enquiries`. Action stages must only run for
not-converted selections (`wants_converted(action)` is False), so a converted enquiry
is listed/counted but never processed again. See docs/frontend-order-inquiry-architecture.md."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.frontorder_enquiry import nodes
from app.agents.frontorder_enquiry.domain import FOLLOW_UP_ACTIONS, build_result
from app.agents.frontorder_enquiry.history import AGENT_KEY, CHECKPOINTER, save_agent_reply, save_user_message, sessions
from app.agents.frontorder_enquiry.monitoring import RunMonitor
from app.agents.frontorder_enquiry.state import FrontOrderEnquiryAgentState
from app.agents.runtime import RunChannel, RunEmitter, utc_now
from app.core.logging import get_logger
from app.history import frontorder_history_store
from app.integrations.bootog import BootogClient

logger = get_logger(__name__)

STEPS: dict[str, tuple[str, str]] = {
    "initialize_run": ("Frontend Order Enquiry Agent", "Initializing Frontend Order Enquiry Agent"),
    "understand_request": ("Request Understanding", "Understanding the request, provider and dates"),
    "resolve_scope": ("Provider Resolution", "Finding the provider in Bootog and applying the conversation's context"),
    "summarize_providers": ("Provider Summary", "Counting matching enquiries per provider (provider summary only)"),
    "fetch_enquiries": ("Enquiry Retrieval", "Fetching the selected provider's order enquiries"),
    "validate_enquiries": ("Response Validation", "Validating enquiry records"),
    "select_enquiries": ("Enquiry Selection", "Classifying enquiries as converted or not converted to orders"),
    "check_follow_up": ("Follow-up Check", "Checking Contact Log history (follow-up requests only)"),
    "prepare_enquiries": ("Enquiry Preparation", "Preparing enquiry context"),
}


def plan() -> list[dict[str, str]]:
    return [{"id": i, "title": title, "description": description} for i, (title, description) in STEPS.items()]


def _stopped(state: FrontOrderEnquiryAgentState) -> bool:
    return state.get("execution_status") in ("failed", "blocked")


def _continue_to(next_node: str) -> Callable[[FrontOrderEnquiryAgentState], str]:
    def route(state: FrontOrderEnquiryAgentState) -> str:
        return "respond" if _stopped(state) else next_node

    return route


def _after_understanding(state: FrontOrderEnquiryAgentState) -> str:
    # A clarification question ends the run before any enquiry API call.
    return "respond" if _stopped(state) or state.get("clarification") else "resolve_scope"


def _after_scope(state: FrontOrderEnquiryAgentState) -> str:
    if _stopped(state) or state.get("clarification"):
        return "respond"
    return "fetch_enquiries" if state.get("view") == "provider_detail" else "summarize_providers"


def _after_selection(state: FrontOrderEnquiryAgentState) -> str:
    if _stopped(state):
        return "respond"
    return "check_follow_up" if state.get("action") in FOLLOW_UP_ACTIONS else "prepare_enquiries"


def build_frontorder_enquiry_graph(checkpointer=None):
    graph = StateGraph(FrontOrderEnquiryAgentState)
    graph.add_node("initialize_run", nodes.initialize_run)
    graph.add_node("understand_request", nodes.understand_request_node)
    graph.add_node("resolve_scope", nodes.resolve_scope)
    graph.add_node("summarize_providers", nodes.summarize_providers)
    graph.add_node("fetch_enquiries", nodes.fetch_enquiries)
    graph.add_node("validate_enquiries", nodes.validate_enquiries_node)
    graph.add_node("select_enquiries", nodes.select_enquiries)
    graph.add_node("check_follow_up", nodes.check_follow_up)
    graph.add_node("prepare_enquiries", nodes.prepare_enquiries)
    graph.add_node("respond", nodes.respond)

    graph.add_edge(START, "initialize_run")
    graph.add_conditional_edges("initialize_run", _continue_to("understand_request"))
    graph.add_conditional_edges("understand_request", _after_understanding)
    graph.add_conditional_edges("resolve_scope", _after_scope)
    graph.add_edge("summarize_providers", "respond")
    graph.add_conditional_edges("fetch_enquiries", _continue_to("validate_enquiries"))
    graph.add_conditional_edges("validate_enquiries", _continue_to("select_enquiries"))
    graph.add_conditional_edges("select_enquiries", _after_selection)
    graph.add_conditional_edges("check_follow_up", _continue_to("prepare_enquiries"))
    graph.add_edge("prepare_enquiries", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)


FRONTORDER_ENQUIRY_GRAPH = build_frontorder_enquiry_graph(CHECKPOINTER)


def new_turn(
    run_id: str,
    session_id: str,
    message: str | None,
    target_provider_id: str | None,
    page: int | None,
) -> dict[str, Any]:
    """Graph input for one message. Every per-run key is reset here; only `scope` (not
    listed) carries over from the conversation's previous message."""
    return {
        "run_id": run_id,
        "session_id": session_id,
        "message": message,
        "target_provider_id": target_provider_id,
        "requested_page": page,
        "action": "",
        "request_summary": "",
        "requested": {},
        "clarification": None,
        "provider_candidates": [],
        "view": "none",
        "provider_summary": [],
        "providers_total": 0,
        "raw_enquiries": [],
        "enquiries_total": 0,
        "enquiries_truncated": False,
        "valid_enquiries": [],
        "malformed_count": 0,
        "duplicate_count": 0,
        "converted_enquiry_ids": [],
        "invalid_order_action_ids": [],
        "selected_enquiries": [],
        "follow_up_by_id": {},
        "enquiry_contexts": [],
        "reply": "",
        "result": {},
        "current_step": "",
        "execution_status": "running",
        "error": None,
    }


async def save_audit(record: dict[str, Any]) -> bool:
    """Default audit sink: the agent database (app.history). Another backend can pass its
    own `audit_sink` to run_frontorder_enquiry_agent."""
    return await frontorder_history_store.record_agent_run(**record)


async def run_frontorder_enquiry_agent(
    channel: RunChannel,
    client: BootogClient,
    message: str | None = None,
    target_provider_id: str | None = None,
    session_id: str | None = None,
    page: int | None = None,
    owner_id: str | None = None,
    graph=None,
    llm=None,
    audit_sink=save_audit,
) -> FrontOrderEnquiryAgentState:
    """Execute one run (one chat message), streaming events into `channel`. Always closes
    the channel and the client, and releases the conversation for its next message.
    `llm` is only passed by tests; normally the shared chat model is used."""
    session_id = session_id or channel.run_id
    emitter = RunEmitter(channel)
    monitor = RunMonitor(run_id=channel.run_id, agent=AGENT_KEY, session_id=session_id, owner_id=owner_id)
    configurable: dict[str, Any] = {
        "thread_id": session_id,
        "bootog_client": client,
        "emitter": emitter,
        "monitor": monitor,
    }
    if llm is not None:
        configurable["llm"] = llm
    state: FrontOrderEnquiryAgentState = new_turn(channel.run_id, session_id, message, target_provider_id, page)
    await save_user_message(session_id, owner_id, message)
    try:
        await emitter.plan(plan())
        state = await (graph or FRONTORDER_ENQUIRY_GRAPH).ainvoke(
            state, config={"configurable": configurable}, durability="exit"
        )
    except Exception as exc:  # graph-level crash: never report success
        logger.exception("[%s] run=%s crashed: %s", nodes.AGENT, channel.run_id, type(exc).__name__)
        state = {**state, "execution_status": "failed", "error": "The Frontend Order Enquiry Agent stopped unexpectedly."}
        state = {**state, "reply": state["error"], "result": build_result(state)}
    finally:
        await client.aclose()
        sessions.end(session_id)

    status = state.get("execution_status") or "failed"
    state = {**state, "finished_at": utc_now()}
    result = {**state.get("result", {}), "sessionId": session_id, "run": monitor.summary()}
    logger.info(
        "[%s] run=%s session=%s finished status=%s duration=%sms api_calls=%s tokens=%s",
        nodes.AGENT, channel.run_id, session_id, status, monitor.duration_ms(), len(monitor.api_calls),
        monitor.token_usage()["totalTokens"],
    )
    reply = state.get("reply") or state.get("error") or ""
    await emitter.finish(status, reply, result)
    await save_agent_reply(session_id, owner_id, reply, result, failed=status in ("failed", "blocked"))

    scope = state.get("scope") or {}
    record = monitor.audit_record(
        status=status,
        intent=state.get("action") or scope.get("action"),
        provider_id=scope.get("provider_id"),
        date_from=scope.get("date_from"),
        date_to=scope.get("date_to"),
        error=state.get("error"),
    )
    try:
        if not await audit_sink(record):
            logger.warning("[%s] run=%s audit record was not saved", nodes.AGENT, channel.run_id)
    except Exception as exc:  # monitoring must never turn a finished run into a failure
        logger.warning("[%s] run=%s audit record failed: %s", nodes.AGENT, channel.run_id, type(exc).__name__)
    return state
