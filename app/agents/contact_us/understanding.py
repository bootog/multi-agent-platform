"""LLM decision layer of the Contact Us chat agent.

The LLM interprets the user's message into a typed `TurnUnderstanding` and writes the
user-facing reply from structured facts. It never calls APIs: the graph routes on the
interpretation and the existing nodes/tools do the work."""

import json
from datetime import date
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, create_model

from app.agents.contact_us.schemas import CUSTOMER_FIELDS
from app.agents.contact_us.state import ChatMessage
from app.llm import get_llm
from app.prompts.contact_us import RESPOND_SYSTEM_PROMPT, UNDERSTAND_SYSTEM_PROMPT

Intent = Literal[
    "greeting_or_help",
    "list_requests",
    "find_request",
    "show_request_details",
    "create_customer",
    "provide_information",
    "cancel_operation",
    "other",
]

INTENT_LABELS: dict[str, str] = {
    "greeting_or_help": "Conversation",
    "list_requests": "List Contact Us requests",
    "find_request": "Find a Contact Us request",
    "show_request_details": "Show request details",
    "create_customer": "Create a customer",
    "provide_information": "Customer information provided",
    "cancel_operation": "Cancel the current task",
    "other": "General question",
}

# Intents that act on a specific Contact Us request.
REQUEST_INTENTS = {"find_request", "show_request_details", "create_customer"}
WORKFLOW_INTENTS = REQUEST_INTENTS | {"provide_information"}


class RequestReference(BaseModel):
    use_current: bool = Field(False, description="The user refers to the request already being discussed.")
    name: str | None = Field(None, description="Person's name used to identify the request.")
    email: str | None = Field(None, description="Email used to identify the request.")
    request_id: str | None = Field(None, description="Contact Us request id, if given.")
    position: int | None = Field(None, description="1-based position in the last list shown; -1 = last.")


class RequestFilters(BaseModel):
    created_from: str | None = Field(None, description="YYYY-MM-DD, inclusive. Only for an explicit period; null for 'recent'.")
    created_to: str | None = Field(None, description="YYYY-MM-DD, inclusive. Only for an explicit period; null for 'recent'.")
    incomplete_only: bool = Field(False, description="Only requests missing customer information.")
    company_name: str | None = Field(None, description="Company/provider the user wants to work with.")


# One optional string per customer draft field, generated from CUSTOMER_FIELDS so the
# LLM can only ever produce fields the customer schema knows about.
CustomerFieldUpdates = create_model(
    "CustomerFieldUpdates",
    **{key: (str | None, Field(None, description=label)) for key, _, label, _ in CUSTOMER_FIELDS},
)


class ExtraInfo(BaseModel):
    key: str = Field(description="snake_case name, e.g. customer_type")
    value: str


class TurnUnderstanding(BaseModel):
    intent: Intent
    goal: Literal["keep", "create_customer", "none"] = "keep"
    request_summary: str = Field(description="Neutral one-line description of the user's request (max 12 words).")
    request_reference: RequestReference | None = None
    filters: RequestFilters | None = None
    customer_field_updates: CustomerFieldUpdates | None = None  # type: ignore[valid-type]
    extra_information: list[ExtraInfo] = Field(default_factory=list)


def _llm(config: RunnableConfig) -> BaseChatModel:
    """Tests inject a model through `configurable.llm`; otherwise the shared one."""
    return config.get("configurable", {}).get("llm") or get_llm()


def _history(messages: list[ChatMessage], limit: int) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages[-limit:]:
        out.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
    return out


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, indent=1)


async def understand(
    config: RunnableConfig,
    history: list[ChatMessage],
    state_summary: dict[str, Any],
) -> TurnUnderstanding:
    structured = _llm(config).with_structured_output(TurnUnderstanding)
    messages: list[BaseMessage] = [
        SystemMessage(UNDERSTAND_SYSTEM_PROMPT),
        SystemMessage(f"Current agent state (today is {date.today().isoformat()}):\n{_json(state_summary)}"),
        *_history(history, 12),
    ]
    result = await structured.ainvoke(messages, config=_child(config))
    if not isinstance(result, TurnUnderstanding):  # defensive: some providers return dicts
        result = TurnUnderstanding.model_validate(result)
    return result


async def compose_reply(config: RunnableConfig, history: list[ChatMessage], facts: dict[str, Any]) -> str:
    messages: list[BaseMessage] = [
        SystemMessage(RESPOND_SYSTEM_PROMPT),
        *_history(history, 12),
        SystemMessage(f"Facts for this turn:\n{_json(facts)}"),
    ]
    reply = await _llm(config).ainvoke(messages, config=_child(config))
    text = reply.content if isinstance(reply.content, str) else "".join(
        part.get("text", "") for part in reply.content if isinstance(part, dict)
    )
    return text.strip()


def _child(config: RunnableConfig) -> RunnableConfig:
    """Callbacks only: never pass Bootog clients or emitters into the model call."""
    return {"callbacks": config.get("callbacks")} if config.get("callbacks") else {}
