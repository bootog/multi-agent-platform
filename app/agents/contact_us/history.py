"""Chat history for the Contact agents, kept beside — not inside — the LangGraph session.

- Every turn is recorded: the user's message before the turn runs, the agent's reply
  (the final `run` event the UI also receives) after it.
- A conversation reopened from history whose live session is gone (backend restart,
  idle expiry, cleared) is re-registered under the same session_id, and its recorded
  messages seed the new LangGraph thread so the agent keeps the conversational context.
  Structured workflow state (selected request, collected fields) is not persisted."""

import asyncio
from typing import Any

from app.agents.contact_us.agent_graph import CONTACT_US_CHAT_GRAPH, sessions
from app.agents.contact_us.state import _HISTORY_LIMIT
from app.agents.runtime import RunChannel
from app.core.logging import get_logger
from app.history import HistoryUnavailable, history_store, session_uuid

logger = get_logger(__name__)


def canonical_session_id(session_id: str | None) -> str | None:
    """Session ids in the chat API's form; other strings pass through unchanged."""
    sid = session_uuid(session_id)
    return sid.hex if sid else session_id


async def restore_session(session_id: str, owner: str, agent: str, graph=None) -> bool:
    """Re-registers a conversation from history. False if it isn't this caller's
    conversation with this agent (or history is unavailable)."""
    try:
        found = await history_store.get_session(session_id, owner, agent)
    except HistoryUnavailable:
        return False
    if found is None:
        return False
    _, messages = found
    await sessions.restore(session_id, owner, agent)

    graph = graph or CONTACT_US_CHAT_GRAPH
    config = {"configurable": {"thread_id": session_id}}
    transcript = [{"role": m.role, "content": m.content} for m in messages if not m.is_error][-_HISTORY_LIMIT:]
    try:
        if transcript and not (await graph.aget_state(config)).values:
            await graph.aupdate_state(
                config, {"conversation_history": transcript, "workflow_status": "completed"}, as_node="respond"
            )
    except Exception:
        logger.warning("[CONTACT_US_AGENT] session=%s history not loaded into the agent", session_id)
    return True


def record_turn(
    turn: "asyncio.Future[Any]",
    channel: RunChannel,
    session_id: str,
    owner: str,
    agent: str,
    message: str | None,
) -> asyncio.Task:
    return asyncio.create_task(_record_turn(turn, channel, session_id, owner, agent, message))


async def _record_turn(
    turn: "asyncio.Future[Any]",
    channel: RunChannel,
    session_id: str,
    owner: str,
    agent: str,
    message: str | None,
) -> None:
    if message:
        await history_store.record_message(session_id, owner, agent, "user", message)
    try:
        await turn
    except Exception:
        pass  # the turn reports its own failure on the channel
    run = next((e for e in reversed(channel.events) if e.get("type") == "run"), None)
    if run is None:
        return
    result = run.get("result") if isinstance(run.get("result"), dict) else None
    reply = (result or {}).get("reply") or run.get("message")
    if reply:
        await history_store.record_message(
            session_id, owner, agent, "assistant", reply, result=result, is_error=run.get("status") == "failed"
        )
