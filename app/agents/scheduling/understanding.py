"""LLM layer of the Scheduling Agent: interprets a message into a typed
`TurnUnderstanding` and writes replies from structured facts. It never calls Bootog and
never chooses ids — the graph resolves every reference against real API data."""

import json
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.contact_us.state import ChatMessage
from app.agents.scheduling.domain import CALLER_TYPES, RESPONSE_RESULTS
from app.llm import get_llm
from app.prompts.scheduling import RESPOND_SYSTEM_PROMPT, UNDERSTAND_SYSTEM_PROMPT

Intent = Literal[
    "greeting_or_help",
    "order_summary",
    "list_orders",
    "find_order",
    "order_details",
    "find_producers",
    "assign_order",
    "assign_orders",
    "check_availability",
    "schedule_order",
    "add_log",
    "list_logs",
    "list_reference",
    "confirm_action",
    "cancel_action",
    "other",
]

INTENT_LABELS: dict[str, str] = {
    "greeting_or_help": "Conversation",
    "order_summary": "Order status summary",
    "list_orders": "List orders",
    "find_order": "Find an order",
    "order_details": "Order details",
    "find_producers": "Find producers",
    "assign_order": "Assign an order",
    "assign_orders": "Assign several orders",
    "check_availability": "Producer availability",
    "schedule_order": "Schedule an inspection",
    "add_log": "Add a contact log",
    "list_logs": "Contact logs",
    "list_reference": "Reference data",
    "confirm_action": "Confirmation",
    "cancel_action": "Cancel",
    "other": "General question",
}

ResponseResult = Literal[tuple(RESPONSE_RESULTS)]  # type: ignore[valid-type]
CallerType = Literal[tuple(CALLER_TYPES)]  # type: ignore[valid-type]


class OrderReference(BaseModel):
    use_current: bool = Field(False, description="The user means the order already being discussed.")
    order_id: str | None = Field(None, description="Order id (GUID) exactly as given.")
    job_number: str | None = Field(None, description="Job number, e.g. JOB1410260002, exactly as given.")
    policy_number: str | None = Field(None, description="Policy number, e.g. KAR-141026, exactly as given.")
    position: int | None = Field(None, description="1-based number of the order in the last list shown; -1 = last.")
    customer_name: str | None = Field(None, description="Customer / homeowner / insured name identifying the order ('Naveen's order').")
    status_names: list[str] = Field(default_factory=list, description="Status that narrows the order, e.g. NEW for 'Naveen's new order'.")
    search_text: str | None = Field(None, description="Address or other text identifying the order (not a person's name).")


class OrderFilters(BaseModel):
    status_names: list[str] = Field(default_factory=list, description="Order status card names, e.g. NEW, SCHEDULED, Assigned, QA Approved, TPQA rejected, Cancelled.")
    service_type_names: list[str] = Field(default_factory=list, description="Service types, e.g. Wind Mitigation Inspection.")
    program_names: list[str] = Field(default_factory=list, description="Program names.")
    state_codes: list[str] = Field(default_factory=list, description="Two-letter US state codes of the property.")
    created_from: str | None = Field(None, description="YYYY-MM-DD, orders created on/after (e.g. today's orders).")
    created_to: str | None = Field(None, description="YYYY-MM-DD, orders created on/before.")
    scheduled_from: str | None = Field(None, description="YYYY-MM-DD, inspections scheduled on/after.")
    scheduled_to: str | None = Field(None, description="YYYY-MM-DD, inspections scheduled on/before.")
    rush_only: bool = Field(False, description="Only rush orders.")
    search_text: str | None = Field(None, description="Free text searched across all order fields.")
    customer_name: str | None = Field(None, description="Customer / homeowner name: 'all orders for <customer>'.")
    page_action: Literal["next", "previous", "first"] | None = Field(None, description="Paging the last order list.")
    page: int | None = Field(None, description="Explicit page number of the last order list.")


class ProducerReference(BaseModel):
    name: str | None = Field(None, description="Producer/inspector name as given.")
    position: int | None = Field(None, description="1-based number in the last producer list shown.")
    use_assigned: bool = Field(False, description="The producer already assigned to the order.")
    selection: Literal["nearest"] | None = Field(None, description="'nearest' when the user wants the agent to pick the nearest/best eligible producer.")


class SlotChoice(BaseModel):
    date: str | None = Field(None, description="YYYY-MM-DD of the appointment the user picked.")
    start_time: str | None = Field(None, description="Start time as 'hh:mm AM/PM'.")
    position: int | None = Field(None, description="1-based number of the slot in the last slot list shown.")
    from_date: str | None = Field(None, description="YYYY-MM-DD to start looking for availability.")
    days: int | None = Field(None, description="How many days of availability to show.")


class LogDetails(BaseModel):
    channel: Literal["phone_call", "sms", "email", "note"] | None = None
    party: Literal["customer", "producer"] | None = Field(None, description="Who was contacted: the customer or the inspector/producer.")
    message: str | None = Field(None, description="What was said / the notes / the message body.")
    subject: str | None = Field(None, description="Email subject.")
    contact_name: str | None = None
    contact_number: str | None = None
    email_to: str | None = None
    response_result: ResponseResult | None = Field(None, description="Phone call result.")  # type: ignore[valid-type]
    caller_type: CallerType | None = Field(None, description="Who the call was with.")  # type: ignore[valid-type]
    direction: Literal["inbound", "outbound"] | None = None


class OrderGroupAssignment(BaseModel):
    """One part of a multi-order assignment: which orders of the last list go to whom."""

    positions: list[int] = Field(default_factory=list, description="1-based numbers of orders in the last order list (or in the batch being prepared). 'first 4' -> [1,2,3,4]; 'next 3' continues after the previous group; 'last 3' of 10 -> [8,9,10].")
    all_orders: bool = Field(False, description="Every order of the list ('all these orders').")
    remaining: bool = Field(False, description="The orders not covered by the other groups ('the rest').")
    producer_name: str | None = Field(None, description="Producer/inspector name as given.")
    nearest: bool = Field(False, description="The nearest eligible producer for each of these orders.")


class TurnUnderstanding(BaseModel):
    intent: Intent
    summary: str = Field(description="Neutral one-line description of the request (max 12 words).")
    order: OrderReference | None = None
    filters: OrderFilters | None = None
    producer: ProducerReference | None = None
    slot: SlotChoice | None = None
    log: LogDetails | None = None
    reference_kind: Literal["service_types", "programs", "time_zones", "order_statuses", "sub_statuses", "providers"] | None = None
    schedule_comments: str | None = Field(None, description="Comments for the appointment, when given.")
    max_distance_miles: float | None = Field(None, description="Distance limit for producers, e.g. 30 for 'within 30 miles'.")
    reassign: bool = Field(False, description="The user explicitly wants to replace the producer already assigned.")
    assignments: list[OrderGroupAssignment] = Field(default_factory=list, description="assign_orders: the order groups and their producers.")
    retry_failed: bool = Field(False, description="assign_orders: retry the orders that failed in the last multi-order assignment.")


def _llm(config: RunnableConfig) -> BaseChatModel:
    """Tests inject a model through `configurable.llm`; otherwise the shared one."""
    return config.get("configurable", {}).get("llm") or get_llm()


def _history(messages: list[ChatMessage], limit: int) -> list[BaseMessage]:
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in messages[-limit:]]


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, indent=1)


def _child(config: RunnableConfig) -> RunnableConfig:
    """Callbacks only: never pass Bootog clients or emitters into the model call."""
    return {"callbacks": config.get("callbacks")} if config.get("callbacks") else {}


async def understand(config: RunnableConfig, history: list[ChatMessage], context: dict[str, Any]) -> TurnUnderstanding:
    structured = _llm(config).with_structured_output(TurnUnderstanding)
    messages: list[BaseMessage] = [
        SystemMessage(UNDERSTAND_SYSTEM_PROMPT),
        SystemMessage(f"Current agent state:\n{_json(context)}"),
        *_history(history, 12),
    ]
    result = await structured.ainvoke(messages, config=_child(config))
    if not isinstance(result, TurnUnderstanding):
        result = TurnUnderstanding.model_validate(result)
    return result


async def compose_reply(config: RunnableConfig, history: list[ChatMessage], facts: dict[str, Any]) -> str:
    messages: list[BaseMessage] = [
        SystemMessage(RESPOND_SYSTEM_PROMPT),
        *_history(history, 8),
        SystemMessage(f"Facts for this turn:\n{_json(facts)}"),
    ]
    reply = await _llm(config).ainvoke(messages, config=_child(config))
    content = reply.content
    text = content if isinstance(content, str) else "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return text.strip()
