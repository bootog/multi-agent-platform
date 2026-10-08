"""Frontend Order Enquiry graph nodes.

Every node is wrapped by `tracked_step`, which publishes running / completed /
failed / blocked events on the run channel and logs the step. Logs carry ids,
counts and status only — never tokens or customer details."""

import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.frontorder_enquiry.schemas import (
    build_enquiry_context,
    split_by_order_action,
    validate_enquiries,
)
from app.agents.frontorder_enquiry.state import FrontOrderEnquiryAgentState
from app.agents.frontorder_enquiry.tools import get_order_enquiries
from app.agents.runtime import RunEmitter, utc_now
from app.core.logging import get_logger, log_step
from app.integrations.bootog import BootogApiError

AGENT = "FRONTORDER_ENQUIRY_AGENT"
logger = get_logger(__name__)


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


def tracked_step(step_id: str, running_message: str, failure_message: str):
    def decorator(fn: NodeFn):
        @functools.wraps(fn)
        async def node(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> dict[str, Any]:
            emitter: RunEmitter = config["configurable"]["emitter"]
            run_id = state["run_id"]
            await emitter.step(step_id, "running", running_message)
            log_step(logger, AGENT, run_id, step_id, "running")
            started = time.perf_counter()

            def elapsed() -> int:
                return int((time.perf_counter() - started) * 1000)

            try:
                result = await fn(state, config)
            except StepBlocked as exc:
                await emitter.step(step_id, "blocked", str(exc), duration_ms=elapsed())
                log_step(logger, AGENT, run_id, step_id, "blocked", elapsed())
                return {"current_step": step_id, "execution_status": "blocked", "error": str(exc)}
            except StepFailed as exc:
                return await _fail(emitter, run_id, step_id, str(exc), elapsed())
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
    await emitter.step(step_id, "failed", message, duration_ms=duration_ms)
    log_step(logger, AGENT, run_id, step_id, "failed", duration_ms)
    return {"current_step": step_id, "execution_status": "failed", "error": message}


def _enquiries(count: int) -> str:
    return f"{count} order enquir{'y' if count == 1 else 'ies'}"


# --- nodes -----------------------------------------------------------------


@tracked_step("initialize_run", "Initializing Frontend Order Enquiry Agent", "Unable to start the Frontend Order Enquiry Agent.")
async def initialize_run(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    logger.info("[%s] run=%s agent started", AGENT, state["run_id"])
    return StepResult("Agent started — retrieving order enquiries", {"execution_status": "running", "started_at": utc_now()})


@tracked_step("fetch_enquiries", "Fetching order enquiries", "Unable to retrieve order enquiries.")
async def fetch_enquiries(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    run_id = state["run_id"]
    logger.info("[%s] run=%s fetching enquiries", AGENT, run_id)
    page = await get_order_enquiries.ainvoke({}, config=config)
    records, total, truncated = page["records"], page["total"], page["truncated"]
    logger.info("[%s] run=%s enquiries received=%s total=%s truncated=%s", AGENT, run_id, len(records), total, truncated)

    if not records:
        message = "No order enquiries returned"
    elif truncated:
        message = f"Retrieved the latest {len(records)} of {total} order enquiries"
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


@tracked_step("filter_converted", "Filtering enquiries already converted to orders", "Unable to filter order enquiries.")
async def filter_converted(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    run_id = state["run_id"]
    eligible, converted, invalid = split_by_order_action(state.get("valid_enquiries", []))
    logger.info(
        "[%s] run=%s converted_ignored=%s invalid_order_action=%s eligible=%s",
        AGENT, run_id, len(converted), len(invalid), len(eligible),
    )
    if invalid:
        logger.warning("[%s] run=%s enquiries with missing/invalid orderAction set aside: %s", AGENT, run_id, invalid)

    message = f"{len(eligible)} eligible — ignored {len(converted)} already converted to order"
    if invalid:
        message += f", set aside {len(invalid)} with missing/invalid orderAction"
    return StepResult(
        message,
        {"eligible_enquiries": eligible, "converted_enquiry_ids": converted, "invalid_order_action_ids": invalid},
        data={"eligible": len(eligible), "convertedIgnored": len(converted), "invalidOrderAction": len(invalid)},
    )


@tracked_step("prepare_enquiries", "Preparing enquiry context", "Unable to prepare order enquiries.")
async def prepare_enquiries(state: FrontOrderEnquiryAgentState, config: RunnableConfig) -> StepResult:
    run_id = state["run_id"]
    contexts = []
    for record in state.get("eligible_enquiries", []):
        context = build_enquiry_context(record)
        logger.info("[%s] run=%s preparing enquiry=%s", AGENT, run_id, context["enquiryId"])
        contexts.append(context)
    return StepResult(
        f"Prepared {_enquiries(len(contexts))} for processing",
        {"enquiry_contexts": contexts},
        data={"prepared": len(contexts)},
    )
