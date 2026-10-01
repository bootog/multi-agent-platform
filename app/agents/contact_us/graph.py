"""Contact Us Agent — LangGraph workflow.

START -> initialize_run -> select_provider -> get_contact_us_data
      -> (retrieve mode: END)
      -> select_contact_request -> prepare_customer_data -> create_customer -> verify_customer -> END

Any failed/blocked step ends the run; later steps never execute."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.contact_us import nodes
from app.agents.contact_us.schemas import ContactUsRunRequest, summarize_contact_request
from app.agents.contact_us.state import ContactUsAgentState, RunMode
from app.agents.runtime import RunChannel, RunEmitter, utc_now
from app.core.logging import get_logger
from app.integrations.bootog import BootogClient

logger = get_logger(__name__)

STEPS: dict[str, tuple[str, str]] = {
    "initialize_run": ("Contact Us Agent", "Initializing Contact Us Agent"),
    "select_provider": ("Provider Selection", "Applying company filter"),
    "get_contact_us_data": ("Contact Request Retrieval", "Pulling Contact Us requests"),
    "select_contact_request": ("Contact Request Selected", "Reading the selected request"),
    "prepare_customer_data": ("Customer Data Preparation", "Preparing customer information"),
    "create_customer": ("Customer Creation", "Creating customer account"),
    "verify_customer": ("Customer Verification", "Verifying customer"),
}
RETRIEVE_STEPS = ["initialize_run", "select_provider", "get_contact_us_data"]
CUSTOMER_STEPS = list(STEPS)


def plan_for(mode: RunMode) -> list[dict[str, str]]:
    ids = CUSTOMER_STEPS if mode == "create_customer" else RETRIEVE_STEPS
    return [{"id": i, "title": STEPS[i][0], "description": STEPS[i][1]} for i in ids]


def _continue_to(next_node: str) -> Callable[[ContactUsAgentState], str]:
    def route(state: ContactUsAgentState) -> str:
        return END if state.get("execution_status") in ("failed", "blocked") else next_node

    return route


def _after_retrieval(state: ContactUsAgentState) -> str:
    if state.get("execution_status") in ("failed", "blocked") or state.get("mode") != "create_customer":
        return END
    return "select_contact_request"


def build_contact_us_graph():
    graph = StateGraph(ContactUsAgentState)
    graph.add_node("initialize_run", nodes.initialize_run)
    graph.add_node("select_provider", nodes.select_provider)
    graph.add_node("get_contact_us_data", nodes.get_contact_us_data)
    graph.add_node("select_contact_request", nodes.select_contact_request)
    graph.add_node("prepare_customer_data", nodes.prepare_customer_data)
    graph.add_node("create_customer", nodes.create_customer_node)
    graph.add_node("verify_customer", nodes.verify_customer)

    graph.add_edge(START, "initialize_run")
    graph.add_conditional_edges("initialize_run", _continue_to("select_provider"))
    graph.add_conditional_edges("select_provider", _continue_to("get_contact_us_data"))
    graph.add_conditional_edges("get_contact_us_data", _after_retrieval)
    graph.add_conditional_edges("select_contact_request", _continue_to("prepare_customer_data"))
    graph.add_conditional_edges("prepare_customer_data", _continue_to("create_customer"))
    graph.add_conditional_edges("create_customer", _continue_to("verify_customer"))
    graph.add_edge("verify_customer", END)
    return graph.compile()


CONTACT_US_GRAPH = build_contact_us_graph()


def _final_message(state: ContactUsAgentState, status: str) -> str:
    if status in ("failed", "blocked"):
        return state.get("error") or "The Contact Us Agent stopped."
    if state.get("mode") == "create_customer":
        return "Contact Us Agent completed — customer created"
    return "Contact Us Agent completed — select a request to create a customer"


def _result(state: ContactUsAgentState) -> dict[str, Any]:
    selected = state.get("contact_request")
    return {
        "mode": state.get("mode"),
        "providerId": state.get("provider_id"),
        "providerName": state.get("provider_name"),
        "totalCount": state.get("contact_requests_total"),
        "contactRequests": [summarize_contact_request(r) for r in state.get("contact_requests", [])],
        "selectedRequest": summarize_contact_request(selected) if selected else None,
        "customerDraft": state.get("customer_payload"),
        "missingRequiredFields": state.get("missing_required_fields", []),
        "customerResult": state.get("customer_result"),
        "failedStep": state.get("current_step") if state.get("execution_status") in ("failed", "blocked") else None,
    }


async def run_contact_us_agent(
    channel: RunChannel,
    request: ContactUsRunRequest,
    client: BootogClient,
    graph=None,
) -> ContactUsAgentState:
    """Execute one run, streaming events into `channel`. Always closes the channel and client."""
    emitter = RunEmitter(channel)
    mode: RunMode = "create_customer" if request.contact_request_id else "retrieve"
    state: ContactUsAgentState = {
        "run_id": channel.run_id,
        "mode": mode,
        "provider_id": request.provider_id,
        "provider_name": request.provider_name,
        "contact_request_id": request.contact_request_id,
        "options": request.options.model_dump(by_alias=True),
        "execution_status": "running",
        "error": None,
    }
    try:
        await emitter.plan(plan_for(mode))
        state = await (graph or CONTACT_US_GRAPH).ainvoke(
            state,
            config={"configurable": {"bootog_client": client, "emitter": emitter}},
        )
    except Exception as exc:  # graph-level crash: never report success
        logger.exception("[CONTACT_US_AGENT] run=%s crashed: %s", channel.run_id, type(exc).__name__)
        state = {**state, "execution_status": "failed", "error": "The Contact Us Agent stopped unexpectedly."}
    finally:
        await client.aclose()

    status = state.get("execution_status") or "completed"
    if status == "running":
        status = "completed"
    state = {**state, "execution_status": status, "finished_at": utc_now()}
    await emitter.finish(status, _final_message(state, status), _result(state))
    return state
