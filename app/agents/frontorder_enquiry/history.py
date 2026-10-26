"""Chat history of the Frontend Order Enquiry Agent, stored in PostgreSQL.

It uses the platform's history store (app/history/store.py) with this agent's own tables,
which have the same columns as the Contact agents' tables:
`frontorder_enquiry_sessions` (one row per conversation) and
`frontorder_enquiry_messages` (one row per message). Rows have
agent_id = "frontorder-enquiry", and every query is limited to the signed-in user.

- save_user_message / save_agent_reply: every chat message and reply is stored.
- restore_conversation: a conversation opened from the History panel (or continued after
  a backend restart) is registered again, and the provider, dates and numbered list of
  its last reply are put back, so replying "1" or "next page" keeps working.

History is optional: when PostgreSQL is not reachable the chat still works, it is only
not saved."""

from typing import Any

from app.agents.frontorder_enquiry.sessions import AGENT_KEY, sessions
from app.core.logging import get_logger
from app.history import HistoryUnavailable, frontorder_history_store

logger = get_logger(__name__)


async def save_user_message(session_id: str, owner_id: str | None, message: str | None) -> None:
    """Store the admin's message. The first message of a conversation becomes its title."""
    if owner_id and message:
        await frontorder_history_store.record_message(session_id, owner_id, AGENT_KEY, "user", message)


async def save_agent_reply(
    session_id: str, owner_id: str | None, reply: str, result: dict[str, Any], failed: bool
) -> None:
    """Store the agent's reply together with its result, so a reopened conversation can
    show the same cards and continue from the same provider and dates."""
    if owner_id and reply:
        await frontorder_history_store.record_message(
            session_id, owner_id, AGENT_KEY, "assistant", reply, result=result, is_error=failed
        )


def scope_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """The conversation scope (provider, dates, kind of enquiries, page) a reply was about."""
    scope = result.get("scope") or {}
    return {
        "action": scope.get("action"),
        "date_from": scope.get("dateFrom"),
        "date_to": scope.get("dateTo"),
        "provider_id": scope.get("providerId"),
        "provider_name": scope.get("providerName"),
        "page": scope.get("page") or 1,
    }


def offered_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The numbered provider list a reply showed, so the user can still answer with a number."""
    # "Several providers match ..." question: its rows are already numbered.
    candidates = result.get("providerCandidates") or []
    if candidates:
        return candidates

    # Provider summary: rows are numbered across pages (page 2 starts at 11 with 10 per page).
    summary = result.get("providerSummary") or []
    pagination = result.get("pagination") or {}
    first = ((pagination.get("page") or 1) - 1) * (pagination.get("pageSize") or len(summary)) + 1
    offered = []
    for number, row in enumerate(summary, start=first):
        offered.append({"number": number, "providerId": row.get("providerId"), "providerName": row.get("providerName")})
    return offered


async def restore_conversation(session_id: str, owner_id: str, graph) -> bool:
    """Continue a stored conversation that is no longer live. Returns False when it is not
    this user's conversation with this agent (or history is unavailable)."""
    try:
        found = await frontorder_history_store.get_session(session_id, owner_id, AGENT_KEY)
    except HistoryUnavailable:
        return False
    if found is None:
        return False
    _, messages = found
    await sessions.restore(session_id, owner_id, AGENT_KEY)

    # The newest successful reply tells us what the conversation was looking at.
    last_reply = None
    for message in reversed(messages):
        if message.role == "assistant" and not message.is_error and message.result:
            last_reply = message
            break
    if last_reply is None:
        return True

    config = {"configurable": {"thread_id": session_id}}
    try:
        if not (await graph.aget_state(config)).values:
            await graph.aupdate_state(
                config,
                {"scope": scope_from_result(last_reply.result), "offered_providers": offered_from_result(last_reply.result)},
                as_node="respond",
            )
    except Exception:  # the conversation still continues, only without its earlier context
        logger.warning("[FRONTORDER_ENQUIRY_AGENT] session=%s context not restored from history", session_id)
    return True
