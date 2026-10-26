"""LLM interpretation of a chat request for the Frontend Order Enquiry Agent.

The LLM only classifies the message into a typed `EnquiryRequest` (what to do, which
kind of period, which provider NAME). Dates are resolved and validated in `dates.py`,
the provider name is looked up in Bootog in `providers.py`, and the graph decides which
API calls to make — the LLM never builds a filter, picks an id or calls an API."""

import time
from datetime import date
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.frontorder_enquiry.monitoring import LlmUsage
from app.agents.frontorder_enquiry.prompts import UNDERSTAND_SYSTEM_PROMPT
from app.llm import get_llm

Action = Literal[
    "list_enquiries", "list_followed_up", "list_not_followed_up", "list_converted", "send_follow_up", "continue", "other"
]
Period = Literal["none", "all", "today", "yesterday", "last_n_days", "date_range"]


class EnquiryRequest(BaseModel):
    action: Action
    period: Period = "none"
    last_n_days: int | None = Field(None, description="Number of days for period=last_n_days.")
    start_date: str | None = Field(None, description="YYYY-MM-DD, inclusive, for period=date_range.")
    end_date: str | None = Field(None, description="YYYY-MM-DD, inclusive, for period=date_range.")
    provider_name: str | None = Field(None, description="Provider/company name as the user wrote it, or null.")
    all_providers: bool = False
    choice_number: int | None = Field(None, description="Number picked from the numbered list in the previous reply, or null.")
    page_change: Literal["none", "next", "previous"] = "none"
    page_number: int | None = Field(None, description="Page the user asked for explicitly, or null.")
    needs_clarification: bool = False
    clarification_question: str | None = None
    request_summary: str = Field("", description="Neutral one-line description of the request (max 12 words).")


# Used when a run starts without a chat message ("Run Agent" button): open enquiries,
# with whatever provider/date scope the conversation already has (none at first).
DEFAULT_REQUEST = EnquiryRequest(action="continue", period="none", request_summary="Order enquiries")


def _llm(config: RunnableConfig) -> BaseChatModel:
    """Tests inject a model through `configurable.llm`; otherwise the shared one."""
    return config.get("configurable", {}).get("llm") or get_llm()


def _usage(raw, duration_ms: int) -> LlmUsage:
    """Token usage reported by the model integration; None fields when it reported none."""
    metadata = getattr(raw, "usage_metadata", None) or {}
    response = getattr(raw, "response_metadata", None) or {}
    return LlmUsage(
        model=response.get("model_name") or response.get("model"),
        duration_ms=duration_ms,
        input_tokens=metadata.get("input_tokens"),
        output_tokens=metadata.get("output_tokens"),
        total_tokens=metadata.get("total_tokens"),
    )


async def understand_request(config: RunnableConfig, message: str, today: date) -> tuple[EnquiryRequest, LlmUsage]:
    # include_raw keeps the model's own message, which carries the token usage.
    structured = _llm(config).with_structured_output(EnquiryRequest, include_raw=True)
    messages = [
        SystemMessage(UNDERSTAND_SYSTEM_PROMPT),
        SystemMessage(f"Today is {today.isoformat()} ({today.strftime('%A')}) in the business timezone."),
        HumanMessage(message),
    ]
    # Callbacks only: never pass the Bootog client or emitter into the model call.
    child = {"callbacks": config.get("callbacks")} if config.get("callbacks") else {}
    started = time.perf_counter()
    output = await structured.ainvoke(messages, config=child)
    usage = _usage(output.get("raw"), int((time.perf_counter() - started) * 1000))
    result = output.get("parsed")
    if result is None:
        raise output.get("parsing_error") or ValueError("The model returned no interpretation.")
    if not isinstance(result, EnquiryRequest):  # defensive: some providers return dicts
        result = EnquiryRequest.model_validate(result)
    return result, usage
