"""Frontend Order Enquiry Agent — LangGraph workflow.

START -> initialize_run -> fetch_enquiries -> validate_enquiries
      -> filter_converted -> prepare_enquiries -> END

Any failed/blocked step ends the run; later steps never execute.

Extending: later stages (customer follow-up, response analysis, Contact Log update,
decision, convert to order) are added as new nodes after `prepare_enquiries`. Each
reads `enquiry_contexts` / `eligible_enquiries` from state — enquiries with
orderAction == 6 never reach them. Add the node + its STEPS entry, then replace the
final `prepare_enquiries -> END` edge with `_continue_to("<next node>")`."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.frontorder_enquiry import nodes
from app.agents.frontorder_enquiry.schemas import summarize_enquiry
from app.agents.frontorder_enquiry.state import FrontOrderEnquiryAgentState
from app.agents.runtime import RunChannel, RunEmitter, utc_now
from app.core.logging import get_logger
from app.integrations.bootog import BootogClient

logger = get_logger(__name__)

STEPS: dict[str, tuple[str, str]] = {
    "initialize_run": ("Frontend Order Enquiry Agent", "Initializing Frontend Order Enquiry Agent"),
    "fetch_enquiries": ("Enquiry Retrieval", "Fetching order enquiries"),
    "validate_enquiries": ("Response Validation", "Validating enquiry records"),
    "filter_converted": ("Converted Filter", "Ignoring enquiries already converted to orders"),
    "prepare_enquiries": ("Enquiry Preparation", "Preparing enquiry context"),
}


def plan() -> list[dict[str, str]]:
    return [{"id": i, "title": title, "description": description} for i, (title, description) in STEPS.items()]


def _continue_to(next_node: str) -> Callable[[FrontOrderEnquiryAgentState], str]:
    def route(state: FrontOrderEnquiryAgentState) -> str:
        return END if state.get("execution_status") in ("failed", "blocked") else next_node

    return route


def build_frontorder_enquiry_graph():
    graph = StateGraph(FrontOrderEnquiryAgentState)
    graph.add_node("initialize_run", nodes.initialize_run)
    graph.add_node("fetch_enquiries", nodes.fetch_enquiries)
    graph.add_node("validate_enquiries", nodes.validate_enquiries_node)
    graph.add_node("filter_converted", nodes.filter_converted)
    graph.add_node("prepare_enquiries", nodes.prepare_enquiries)

    graph.add_edge(START, "initialize_run")
    graph.add_conditional_edges("initialize_run", _continue_to("fetch_enquiries"))
    graph.add_conditional_edges("fetch_enquiries", _continue_to("validate_enquiries"))
    graph.add_conditional_edges("validate_enquiries", _continue_to("filter_converted"))
    graph.add_conditional_edges("filter_converted", _continue_to("prepare_enquiries"))
    graph.add_edge("prepare_enquiries", END)
    return graph.compile()


FRONTORDER_ENQUIRY_GRAPH = build_frontorder_enquiry_graph()


def _final_message(state: FrontOrderEnquiryAgentState, status: str) -> str:
    if status in ("failed", "blocked"):
        return state.get("error") or "The Frontend Order Enquiry Agent stopped."
    count = len(state.get("enquiry_contexts", []))
    if not count:
        return "Frontend Order Enquiry Agent completed — no enquiries to process"
    return f"Frontend Order Enquiry Agent completed — {count} enquir{'y' if count == 1 else 'ies'} ready for the next stage"


def _result(state: FrontOrderEnquiryAgentState) -> dict[str, Any]:
    return {
        "totalCount": state.get("enquiries_total"),
        "retrievedCount": len(state.get("raw_enquiries", [])),
        "truncated": state.get("enquiries_truncated", False),
        "malformedCount": state.get("malformed_count", 0),
        "duplicateCount": state.get("duplicate_count", 0),
        "convertedEnquiryIds": state.get("converted_enquiry_ids", []),
        "invalidOrderActionIds": state.get("invalid_order_action_ids", []),
        "eligibleEnquiries": [summarize_enquiry(c) for c in state.get("enquiry_contexts", [])],
        "failedStep": state.get("current_step") if state.get("execution_status") in ("failed", "blocked") else None,
    }


async def run_frontorder_enquiry_agent(
    channel: RunChannel,
    client: BootogClient,
    graph=None,
) -> FrontOrderEnquiryAgentState:
    """Execute one run, streaming events into `channel`. Always closes the channel and client."""
    emitter = RunEmitter(channel)
    state: FrontOrderEnquiryAgentState = {"run_id": channel.run_id, "execution_status": "running", "error": None}
    try:
        await emitter.plan(plan())
        state = await (graph or FRONTORDER_ENQUIRY_GRAPH).ainvoke(
            state,
            config={"configurable": {"bootog_client": client, "emitter": emitter}},
        )
    except Exception as exc:  # graph-level crash: never report success
        logger.exception("[%s] run=%s crashed: %s", nodes.AGENT, channel.run_id, type(exc).__name__)
        state = {**state, "execution_status": "failed", "error": "The Frontend Order Enquiry Agent stopped unexpectedly."}
    finally:
        await client.aclose()

    status = state.get("execution_status") or "completed"
    if status == "running":
        status = "completed"
    state = {**state, "execution_status": status, "finished_at": utc_now()}
    logger.info("[%s] run=%s agent finished status=%s", nodes.AGENT, channel.run_id, status)
    await emitter.finish(status, _final_message(state, status), _result(state))
    return state
