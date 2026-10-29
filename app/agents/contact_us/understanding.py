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

from app.agents.contact_us.conversions import FIELD_SPECS
from app.agents.contact_us.state import ChatMessage
from app.llm import get_llm
from app.prompts.contact_us import RESPOND_SYSTEM_PROMPT, UNDERSTAND_SYSTEM_PROMPT

Intent = Literal[
    "greeting_or_help",
    "list_requests",
    "find_request",
    "show_request_details",
    "create_customer",
    "create_b2b_client",
    "create_partner",
    "convert_request",
    "edit_details",
    "review_details",
    "list_providers",
    "provide_information",
    "confirm_action",
    "retry_operation",
    "cancel_operation",
    "show_history",
    "change_status",
    "other",
]

INTENT_LABELS: dict[str, str] = {
    "greeting_or_help": "Conversation",
    "list_requests": "List Contact Us requests",
    "find_request": "Find a Contact Us request",
    "show_request_details": "Show request details",
    "create_customer": "Create a customer",
    "create_b2b_client": "Create a B2B client",
    "create_partner": "Create a partner",
    "convert_request": "Convert the request",
    "edit_details": "Change details",
    "review_details": "Review the details",
    "list_providers": "Find service providers",
    "provide_information": "Information provided",
    "confirm_action": "Confirmation",
    "retry_operation": "Retry the failed step",
    "cancel_operation": "Cancel the current task",
    "show_history": "Show the request's history",
    "change_status": "Change the request status",
    "other": "General question",
}

# Intents that act on a specific Contact Us request.
CONVERSION_INTENTS = {"create_customer", "create_b2b_client", "create_partner", "convert_request"}
# A request's tracking history, and changing its status (ContactUs/ChangeStatus).
REQUEST_ACTION_INTENTS = {"show_history", "change_status"}
REQUEST_INTENTS = {"find_request", "show_request_details"} | CONVERSION_INTENTS | REQUEST_ACTION_INTENTS
# Intents that move a conversion forward when one is in progress.
_DETAIL_INTENTS = {"provide_information", "confirm_action", "retry_operation", "edit_details", "review_details"}
ADVANCING_INTENTS = CONVERSION_INTENTS | _DETAIL_INTENTS
WORKFLOW_INTENTS = REQUEST_INTENTS | _DETAIL_INTENTS


class RequestReference(BaseModel):
    use_current: bool = Field(False, description="The user refers to the request already being discussed.")
    name: str | None = Field(None, description="Person's name used to identify the request.")
    email: str | None = Field(None, description="Email used to identify the request.")
    request_id: str | None = Field(None, description="Contact Us request id, if given (copy it exactly).")
    position: int | None = Field(None, description="1-based number in the last request list shown; -1 = last.")
    created_date: str | None = Field(None, description="YYYY-MM-DD the request was created, when used to pick one.")
    status: str | None = Field(None, description="Status name used to pick a request, e.g. 'Under Review'.")
    assigned_user_name: str | None = Field(None, description="Assignee name used to pick a request.")


class RequestFilters(BaseModel):
    created_from: str | None = Field(None, description="YYYY-MM-DD, inclusive. Only for an explicit period; null for 'recent'.")
    created_to: str | None = Field(None, description="YYYY-MM-DD, inclusive. Only for an explicit period; null for 'recent'.")
    incomplete_only: bool = Field(False, description="Only requests missing customer information.")
    only_open: bool = Field(False, description="The user explicitly asks for open/pending/active requests only.")
    company_name: str | None = Field(None, description="Company whose requests the user wants to list (session company filter).")
    statuses: list[str] = Field(default_factory=list, description="Request status names, e.g. 'Under Review', 'CRM Completed'.")
    assigned_user_name: str | None = Field(None, description="Person the requests are assigned to.")
    all_statuses: bool = Field(False, description="The user explicitly asks for every status ('all statuses').")
    clear: list[Literal["status", "date", "company", "assignee", "all"]] = Field(
        default_factory=list, description="Active list filters the user removes ('all' = clear every filter)."
    )
    assigned_to_me: bool = Field(False, description="The user asks for the requests assigned to them ('my requests').")
    page: Literal["next", "previous"] | None = Field(None, description="Next / previous page of the same list.")
    page_number: int | None = Field(None, description="A specific page of the same list, when the user names one.")


# One optional string per conversion form field, generated from FIELD_SPECS so the LLM
# can only ever produce fields the conversion forms know about.
CustomerFieldUpdates = create_model(
    "CustomerFieldUpdates",
    **{key: (str | None, Field(None, description=spec.label)) for key, spec in FIELD_SPECS.items()},
)


class ConversionDetails(BaseModel):
    partner_type: Literal["insurance_agency", "real_estate_agency", "contractor"] | None = Field(
        None, description="Partner company type, when the user names one."
    )
    target_provider_name: str | None = Field(
        None, description="Service provider a new B2B client / partner should be created under."
    )
    option_position: int | None = Field(
        None, description="1-based choice from the options the agent's pending question listed."
    )
    option_positions: list[int] = Field(
        default_factory=list, description="Several 1-based choices from that list ('1 and 3')."
    )
    subcategory_names: list[str] = Field(default_factory=list, description="Contractor service categories named.")
    edit_fields: list[str] = Field(
        default_factory=list,
        description="Fields the user wants to change but gave no new value for: field keys, or 'address' for the whole address.",
    )
    review_first: bool = Field(False, description="The user wants to see / check the details before the account is created.")


class StatusChange(BaseModel):
    status: str | None = Field(None, description="The new status name, in the user's words ('Approved', 'cancel it' -> 'Cancelled').")
    comment: str | None = Field(None, description="A comment the user wants recorded with the change, word for word.")
    positions: list[int] = Field(
        default_factory=list, description="1-based numbers of SEVERAL requests in the last list to change ('1 and 3')."
    )


class ProviderQuery(BaseModel):
    name: str | None = Field(None, description="Provider name or part of it.")
    location: str | None = Field(None, description="City/state the providers should be in.")


class ExtraInfo(BaseModel):
    key: str = Field(description="snake_case name, e.g. customer_type")
    value: str


class TurnUnderstanding(BaseModel):
    intent: Intent
    goal: Literal["keep", "create_customer", "create_b2b_client", "create_partner", "convert_request", "none"] = "keep"
    request_summary: str = Field(description="Neutral one-line description of the user's request (max 12 words).")
    request_reference: RequestReference | None = None
    filters: RequestFilters | None = None
    customer_field_updates: CustomerFieldUpdates | None = None  # type: ignore[valid-type]
    conversion: ConversionDetails | None = None
    status_change: StatusChange | None = None
    provider_query: ProviderQuery | None = None
    extra_information: list[ExtraInfo] = Field(default_factory=list)


def _llm(config: RunnableConfig) -> BaseChatModel:
    """Tests inject a model through `configurable.llm`; otherwise the shared one."""
    return config.get("configurable", {}).get("llm") or get_llm()


def _history(messages: list[ChatMessage], limit: int) -> list[BaseMessage]:
    """The last `limit` transcript messages as LangChain messages."""
    out: list[BaseMessage] = []
    for m in messages[-limit:]:
        out.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
    return out


def _json(data: Any) -> str:
    """Readable JSON for a prompt; values JSON can't encode are written with str()."""
    return json.dumps(data, ensure_ascii=False, default=str, indent=1)


async def understand(
    config: RunnableConfig,
    history: list[ChatMessage],
    state_summary: dict[str, Any],
) -> TurnUnderstanding:
    """Ask the LLM what the user's latest message means.

    Args:
        config: The turn's config (an `llm` may be injected for tests).
        history: The conversation so far; the last 12 messages are sent.
        state_summary: What the agent currently knows (see agent_nodes._state_summary).

    Returns:
        The typed interpretation of the message.
    """
    structured =_llm(config).with_structured_output(TurnUnderstanding)
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
    """Ask the LLM to write the reply to the user from this turn's facts.

    Args:
        config: The turn's config (an `llm` may be injected for tests).
        history: The conversation so far; the last 12 messages are sent.
        facts: Structured results of the turn (see agent_nodes.build_facts).

    Returns:
        The reply text, stripped; empty when the model returned no text.
    """
    messages:list[BaseMessage] = [
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
