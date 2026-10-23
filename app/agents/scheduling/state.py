"""LangGraph state of the Scheduling Agent chat (one checkpointed thread per session).

Per-turn fields are reset by `start_turn`; the rest is the conversation's working memory.
Everything kept here comes from real Bootog responses (summaries only — no raw payloads,
tokens or credentials)."""

from typing import Annotated, Any, TypedDict

from app.agents.contact_us.state import ChatMessage, WorkflowStatus, append_history


class SchedulingChatState(TypedDict, total=False):
    run_id: str
    session_id: str
    conversation_history: Annotated[list[ChatMessage], append_history]

    # --- set per turn -------------------------------------------------------------
    current_user_message: str | None
    action: dict[str, Any] | None  # structured input from a chat card (button)
    utc_offset_minutes: int  # the user's clock: minutes to add to UTC
    intent: str | None
    understanding: dict[str, Any] | None
    turn_steps: list[str]
    shown: list[str]  # result cards produced this turn ("summary", "orders", "order", ...)
    notices: list[str]  # facts for the reply (e.g. an unknown status name)
    question: dict[str, Any] | None  # what the agent asks this turn ({type, text, ...})
    answered_question: dict[str, Any] | None  # the question this message answers (asked last turn)
    action_result: dict[str, Any] | None  # outcome of a confirmed write this turn
    error: str | None
    execution_status: str | None
    agent_response: str | None
    workflow_status: WorkflowStatus
    awaiting_user_input: bool

    # --- kept across turns ----------------------------------------------------------
    summary: dict[str, Any] | None  # last order dashboard (status groups + counts)
    orders: dict[str, Any] | None  # last order listing {items, total, page, pageSize, description, context}
    orders_query: dict[str, Any] | None  # how the last listing was queried (for "next page")
    selected_order: dict[str, Any] | None  # the order being worked on (domain.order_summary)
    producers: dict[str, Any] | None  # {orderId, items}: eligible producers for that order
    ranking: dict[str, Any] | None  # Assign: producers ranked for that order (domain.rank_producers)
    availability: dict[str, Any] | None  # {orderId, producer, durationMinutes, days}
    log_draft: dict[str, Any] | None  # Add Log details collected so far {orderId, ...}
    logs: dict[str, Any] | None  # {orderId, items, total}
    reference: dict[str, Any] | None  # {kind, title, items, total}
    pending_action: dict[str, Any] | None  # write awaiting confirmation {id, type, ...}
    pending_question: dict[str, Any] | None  # the question the user's next message may answer
    # Multi-order assignment: the orders taken from the last list and the producer chosen for
    # each position ({orders: [{id, jobNumber, policyNumber, customer}], mapping, listingKey}),
    # the table shown for it, and the outcome of the last confirmed batch (for "retry").
    batch_draft: dict[str, Any] | None
    batch_view: dict[str, Any] | None
    last_batch: dict[str, Any] | None
