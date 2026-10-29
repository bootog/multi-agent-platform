"""Request and response models of the Frontend Order Enquiry Agent API: runs
(POST /agents/frontorder-enquiry/runs) and chat history (/agents/frontorder-enquiry/history).
Business rules live in domain.py."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class RunStarted(CamelModel):
    run_id: str
    steps: list[dict[str, str]]
    # The conversation this run belongs to. Send it back with the next message so the
    # agent keeps the provider, dates and kind of enquiries ("now only converted ones").
    session_id: str | None = None
    # True when the sessionId sent was no longer live (restart/expiry): a new
    # conversation was started and the previous context is not applied.
    session_restarted: bool = False


class FrontOrderEnquiryRunRequest(CamelModel):
    # The admin's chat message, e.g. "How many enquiries today?". No message = "Run
    # Agent": the provider summary of open enquiries, without a date filter.
    message: str | None = Field(None, max_length=4000)
    # A provider picked in the UI (e.g. from the provider summary). It is verified against
    # Bootog's provider list before use. Without it, a provider named in the message is
    # looked up by name; with neither, the agent answers with the provider summary.
    target_provider_id: str | None = None
    # Conversation id from a previous RunStarted, to continue its context.
    session_id: str | None = Field(None, max_length=64)
    # Page of the current view (provider summary or a provider's enquiries).
    page: int | None = Field(None, ge=1, le=10000)


# --- chat history -----------------------------------------------------------------------


class HistorySessionOut(CamelModel):
    """One stored conversation in the History panel."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class HistoryMessageOut(CamelModel):
    """One stored message; `result` is the reply's run result, so its card shows again."""

    id: int
    role: Literal["user", "assistant"]
    content: str
    result: dict[str, Any] | None = None
    error: bool = False
    created_at: datetime


class HistoryDetailOut(HistorySessionOut):
    """One conversation with all its messages."""

    messages: list[HistoryMessageOut]


class DeleteHistoryRequest(CamelModel):
    session_ids: list[str] = Field(..., min_length=1, max_length=100)


class DeleteHistoryResult(CamelModel):
    deleted_session_ids: list[str]
    deleted_count: int

