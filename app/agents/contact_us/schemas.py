from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app.agents.contact_us.modules import AGENT_KEY_PATTERN


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AgentOptions(CamelModel):
    auto_categorize: bool = True
    draft_replies: bool = True
    auto_assign: bool = False
    flag_needs_clarification: bool = True
    require_approval: bool = True
    default_category: str = Field("Review", max_length=50)
    reply_tone: str = Field("Professional", max_length=50)


class ContactUsRunRequest(CamelModel):
    provider_id: str | None = Field(None, max_length=64)
    provider_name: str | None = Field(None, max_length=200)
    # Set -> customer-creation run for that request; unset -> retrieval run.
    contact_request_id: str | None = Field(None, max_length=64)
    options: AgentOptions = Field(default_factory=AgentOptions)


class RunStarted(CamelModel):
    run_id: str
    mode: str
    steps: list[dict[str, str]]


class ChatTurnRequest(CamelModel):
    # Omit to start a new session. Unknown/expired ids start a new session too.
    session_id: str | None = Field(None, max_length=64)
    # Omit (or leave empty) to start/refresh the session without a message ("Run Agent").
    message: str | None = Field(None, max_length=4000)
    # Contact agent: route segment of its Contact submenu (e.g. "Request-Demo").
    # Omit for the Contact Us agent. A session always stays with the agent it started with.
    agent_key: str | None = Field(None, pattern=AGENT_KEY_PATTERN)
    # The submenu's label, used only for wording (e.g. "Request Demo").
    agent_label: str | None = Field(None, max_length=80)


class ChatTurnStarted(CamelModel):
    run_id: str
    session_id: str
    # True when the given session no longer existed and a new one was started.
    session_restarted: bool = False


# --- Chat history ------------------------------------------------------------
# Response bodies never carry top-level "message"/"status"/"failed" keys: the host's
# AuthInterceptor turns those into toasts.


class ChatSessionRequest(CamelModel):
    agent_key: str | None = Field(None, pattern=AGENT_KEY_PATTERN)


class ChatSessionCreated(CamelModel):
    session_id: str
    created_at: datetime


class ChatHistorySession(CamelModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ChatHistoryMessage(CamelModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    # The `result` of the turn's final run event (cards the UI showed with the reply).
    result: dict[str, Any] | None = None
    error: bool = False
    created_at: datetime


class ChatHistoryDetail(ChatHistorySession):
    messages: list[ChatHistoryMessage]


class DeleteHistoryRequest(CamelModel):
    session_ids: list[str] = Field(..., min_length=1, max_length=100)
    agent_key: str | None = Field(None, pattern=AGENT_KEY_PATTERN)


class DeleteHistoryResult(CamelModel):
    deleted_session_ids: list[str]
    deleted_count: int


class ProviderOut(CamelModel):
    provider_id: str
    provider_name: str


# --- Customer draft -------------------------------------------------------
# Fields of the Bootog "New Customer" form. This is the agent's internal draft,
# NOT an API payload: the customer-creation API and its field names are not
# documented in ContactUS.pdf, so nothing is sent until that mapping is confirmed.

CUSTOMER_FIELDS: list[tuple[str, str, str, bool]] = [
    # (draft field, Contact Us source field, form label, required)
    ("firstName", "firstName", "First Name", True),
    ("lastName", "lastName", "Last Name", True),
    ("email", "emailId", "Email", True),
    ("phoneNumberCode", "phoneNumberCode", "Phone Code", False),
    ("phone", "phone", "Phone Number", False),
    ("address1", "address1", "Street Address 1", True),
    ("address2", "address2", "Street Address 2", False),
    ("city", "city", "City", True),
    ("state", "state", "State", True),
    ("zipcode", "zipcode", "Zipcode", True),
    ("county", "county", "County", True),
    ("country", "country", "Country", True),
]


def build_customer_draft(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map a Contact Us record to the customer draft; return (draft, missing required labels)."""
    draft: dict[str, Any] = {}
    missing: list[str] = []
    for field, source, label, required in CUSTOMER_FIELDS:
        value = record.get(source)
        value = value.strip() if isinstance(value, str) else value
        draft[field] = value or None
        if required and not value:
            missing.append(label)
    return draft, missing


def summarize_contact_request(record: dict[str, Any]) -> dict[str, Any]:
    """The subset of a Contact Us record the UI shows in the request picker."""
    comments = record.get("comments") or ""
    # Web-form submissions append " ~ {json echo of the form}"; keep the message part.
    comments = comments.split(" ~ ", 1)[0].strip()
    phone = " ".join(p for p in (record.get("phoneNumberCode"), record.get("phone")) if p)
    return {
        "id": record.get("id"),
        "firstName": record.get("firstName"),
        "lastName": record.get("lastName"),
        "emailId": record.get("emailId"),
        "phone": phone or None,
        "moduleName": record.get("moduleName"),
        "status": record.get("status"),
        "category": record.get("category"),
        "comments": comments[:160] or None,
        "createdDate": record.get("createdDate"),
        "assignedUserName": record.get("assignedUserName"),
    }
