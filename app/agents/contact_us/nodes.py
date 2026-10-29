"""Contact Us graph nodes.

Every node is wrapped by `tracked_step`, which publishes the real running /
completed / failed / blocked events for the Live Tracking panel and logs the
step. Nothing here is simulated: a step completes when its work returns."""

import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.contact_us.schemas import build_customer_draft
from app.agents.contact_us.state import ContactUsAgentState
from app.agents.contact_us.tools import create_customer, get_contact_us_requests, get_provider_list
from app.agents.runtime import RunEmitter, utc_now
from app.core.logging import get_logger, log_step
from app.integrations.bootog import BootogApiError, CustomerApiNotConfiguredError

AGENT = "CONTACT_US_AGENT"
logger = get_logger(__name__)


@dataclass
class StepResult:
    """What a tracked step returns when it succeeds.

    Attributes:
        message: Text of the step's "completed" event in the Live Tracking panel.
        update: Fields merged into the graph state.
        data: Optional extra details attached to the "completed" event.
    """

    message: str
    update: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] | None = None


class StepFailed(Exception):
    """Expected failure with a user-safe message. `update` is still applied to the state
    (e.g. "the account was created, then linking failed"), so a retry can resume safely."""

    def __init__(self, message: str, update: dict[str, Any] | None = None):
        super().__init__(message)
        self.update = update or {}


class StepBlocked(Exception):
    """The step must not go ahead (e.g. an API is not configured, or the account already
    exists). Not a success. `update` is still applied to the state, like StepFailed."""

    def __init__(self, message: str, update: dict[str, Any] | None = None):
        super().__init__(message)
        self.update = update or {}


NodeFn = Callable[[ContactUsAgentState, RunnableConfig], Awaitable[StepResult]]


def tracked_step(step_id: str, running_message: str | Callable[[ContactUsAgentState], str], failure_message: str):
    """Turn a step function into a graph node that reports its progress.

    The node publishes a "running" event, runs the step, then publishes how it ended
    and logs it. Outcomes:

    - returns a StepResult -> "completed"; its `update` is applied to the state.
    - raises StepBlocked / CustomerApiNotConfiguredError -> "blocked".
    - raises StepFailed -> "failed" with the exception's message.
    - raises BootogApiError -> "failed" with `failure_message` plus the API's message.
    - raises anything else -> "failed" with `failure_message`; details go to the log only.

    Blocked and failed outcomes set `execution_status` and `error` instead of raising,
    so the graph's routing decides what happens next.

    Args:
        step_id: Step name used in events and logs.
        running_message: Text of the "running" event, or a function that builds it
            from the state.
        failure_message: User-safe text used when the step fails unexpectedly.
    """

    def decorator(fn: NodeFn):
        @functools.wraps(fn)
        async def node(state: ContactUsAgentState, config: RunnableConfig) -> dict[str, Any]:
            emitter: RunEmitter = config["configurable"]["emitter"]
            run_id = state["run_id"]
            message = running_message(state) if callable(running_message) else running_message
            await emitter.step(step_id, "running", message)
            log_step(logger, AGENT, run_id, step_id, "running")
            started = time.perf_counter()

            def elapsed() -> int:
                return int((time.perf_counter() - started) * 1000)

            try:
                result = await fn(state, config)
            except (StepBlocked, CustomerApiNotConfiguredError) as exc:
                await emitter.step(step_id, "blocked", str(exc), duration_ms=elapsed())
                log_step(logger, AGENT, run_id, step_id, "blocked", elapsed())
                update = getattr(exc, "update", None) or {}
                return {**update, "current_step": step_id, "execution_status": "blocked", "error": str(exc)}
            except StepFailed as exc:
                return {**exc.update, **await _fail(emitter, run_id, step_id, str(exc), elapsed())}
            except BootogApiError as exc:
                return await _fail(emitter, run_id, step_id, f"{failure_message} {exc.message}", elapsed())
            except Exception as exc:  # unexpected: keep details in logs only
                logger.exception("[%s] run=%s step=%s unexpected %s", AGENT, run_id, step_id, type(exc).__name__)
                return await _fail(emitter, run_id, step_id, failure_message, elapsed())

            await emitter.step(step_id, "completed", result.message, data=result.data, duration_ms=elapsed())
            log_step(logger, AGENT, run_id, step_id, "completed", elapsed())
            return {**result.update, "current_step": step_id}

        return node

    return decorator


async def _fail(emitter: RunEmitter, run_id: str, step_id: str, message: str, duration_ms: int) -> dict[str, Any]:
    """Publish and log a "failed" step, and return the state update that marks the run failed."""
    await emitter.step(step_id, "failed", message, duration_ms=duration_ms)
    log_step(logger, AGENT, run_id, step_id, "failed", duration_ms)
    return {"current_step": step_id, "execution_status": "failed", "error": message}


# --- nodes -----------------------------------------------------------------


@tracked_step("initialize_run", "Initializing Contact Us Agent", "Unable to start the Contact Us Agent.")
async def initialize_run(state: ContactUsAgentState, config: RunnableConfig) -> StepResult:
    """First step of a run: mark it running and record the start time."""
    if state["mode"] == "create_customer":
        message = "Agent started — customer creation for the selected request"
    else:
        message = "Agent started — retrieving Contact Us requests"
    return StepResult(message, {"execution_status": "running", "started_at": utc_now()})


@tracked_step("select_provider", "Applying company filter", "Unable to verify the selected company.")
async def select_provider(state: ContactUsAgentState, config: RunnableConfig) -> StepResult:
    """Check that the selected company is one the caller's session can use.

    With no company selected, nothing is checked and the session's default access
    applies. Otherwise the company must be in the session's provider list; its name is
    then stored in `provider_name`.

    Raises:
        StepFailed: The selected company is not available to the session.
    """
    provider_id = state.get("provider_id")
    if not provider_id:
        return StepResult("No company selected — using your session's default access")

    providers = await get_provider_list.ainvoke({}, config=config)
    match = next((p for p in providers if p["providerId"] == provider_id), None)
    if match is None:
        raise StepFailed("The selected company is not available to your session.")
    return StepResult(f"Using {match['providerName']}", {"provider_name": match["providerName"]})


def _request_noun(state: ContactUsAgentState) -> str:
    """"Contact Us request" for the run workflow; the chat agent's own kind otherwise."""
    return (state.get("agent_name") or "Contact Us Agent").removesuffix(" Agent") + " request"


@tracked_step(
    "get_contact_us_data",
    lambda state: f"Pulling {_request_noun(state)}s",
    "Unable to retrieve Contact Us requests.",
)
async def get_contact_us_data(state: ContactUsAgentState, config: RunnableConfig) -> StepResult:
    """Retrieve Contact Us requests with the filters in the state.

    Shared by the run workflow and the chat agent. Stores the records in
    `contact_requests`, the server's total in `contact_requests_total`, and whether a
    search stopped at the page limit in `contact_requests_truncated`.
    """
    # `request_filters` / `module_filter` are only set by the chat agent; the run workflow
    # never sets them, so it keeps the documented Contact-Us filter.
    filters = state.get("request_filters") or {}
    page = await get_contact_us_requests.ainvoke(
        {
            "provider_id": state.get("provider_id"),
            "contact_name": filters.get("contact_name"),
            "email": filters.get("email"),
            "created_from": filters.get("created_from"),
            "created_to": filters.get("created_to"),
            "module_filter": state.get("module_filter"),
            "statuses": filters.get("statuses"),
            "all_statuses": bool(filters.get("all_statuses")),
            "assigned_user_ids": filters.get("assigned_user_ids"),
            "request_id": filters.get("request_id"),
            "page": int(filters.get("page") or 1),
        },
        config=config,
    )
    records, total = page["records"], page["total"]
    noun = _request_noun(state)
    if page.get("search"):
        message = f"Searched all {noun}s — {len(records)} match{'es' if len(records) != 1 else ''}"
        if page.get("truncated"):
            message += " (search stopped at the page limit; older requests may be missing)"
    elif (page.get("page") or 1) > 1:
        message = f"Retrieved page {page['page']} — {len(records)} of {total} {noun}s"
    elif len(records) == total:
        message = f"Retrieved {total} {noun}{'s' if total != 1 else ''}"
    else:
        message = f"Retrieved the latest {len(records)} of {total} {noun}s"
    return StepResult(
        message,
        {"contact_requests": records, "contact_requests_total": total, "contact_requests_truncated": bool(page.get("truncated"))},
        data={"retrieved": len(records), "total": total},
    )


@tracked_step("select_contact_request", "Reading the selected Contact Us request", "Unable to read the selected request.")
async def select_contact_request(state: ContactUsAgentState, config: RunnableConfig) -> StepResult:
    """Pick the record whose id is `contact_request_id` from the retrieved requests.

    Raises:
        StepFailed: That id is not among the retrieved requests.
    """
    wanted = state.get("contact_request_id")
    record = next((r for r in state.get("contact_requests", []) if r.get("id") == wanted), None)
    if record is None:
        raise StepFailed("The selected Contact Us request was not found in the retrieved data.")
    name = " ".join(p for p in (record.get("firstName"), record.get("lastName")) if p) or "Request"
    return StepResult(
        f"{name} selected",
        {"contact_request": record},
        data={"name": name, "email": record.get("emailId")},
    )


@tracked_step("prepare_customer_data", "Preparing customer information", "Unable to prepare customer information.")
async def prepare_customer_data(state: ContactUsAgentState, config: RunnableConfig) -> StepResult:
    """Map the selected request to a customer draft and list the required fields it lacks."""
    draft, missing = build_customer_draft(state["contact_request"])
    filled = sum(1 for v in draft.values() if v)
    message = f"Mapped {filled} customer field{'s' if filled != 1 else ''} from the request"
    if missing:
        message += f" — missing required: {', '.join(missing)}"
    return StepResult(
        message,
        {"customer_payload": draft, "missing_required_fields": missing},
        data={"missingFields": missing},
    )


@tracked_step("create_customer", "Creating customer account", "Customer creation failed.")
async def create_customer_node(state: ContactUsAgentState, config: RunnableConfig) -> StepResult:
    """Run-workflow customer creation through the `create_customer` tool.

    Raises:
        StepFailed: The API response has no customer id.
    """
    result = await create_customer.ainvoke({"customer_payload": state["customer_payload"]}, config=config)
    customer_id = result.get("id") if isinstance(result, dict) else None
    if not customer_id:
        raise StepFailed("Customer creation failed. The API response did not include a customer id.")
    return StepResult("Customer created", {"customer_result": result}, data={"customerId": customer_id})


@tracked_step("verify_customer", "Verifying customer", "Customer verification failed.")
async def verify_customer(state: ContactUsAgentState, config: RunnableConfig) -> StepResult:
    """Confirm that the creation response contains a customer id.

    Raises:
        StepFailed: There is no customer id to verify.
    """
    # No customer-details API with documented parameters exists yet, so verification
    # is limited to the creation response itself.
    customer_id = (state.get("customer_result") or {}).get("id")
    if not customer_id:
        raise StepFailed("Customer verification failed. No customer id to verify.")
    return StepResult(f"Customer {customer_id} confirmed in the API response")
