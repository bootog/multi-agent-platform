"""Contact agent chat history: new conversations, the history list, one conversation's
messages, and deleting selected conversations. Stored in the AGENT_DB_NAME database;
always scoped to the caller and to one Contact agent."""

from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Query, status

from app.agents.contact_us.agent_graph import owner_fingerprint, sessions
from app.agents.contact_us.history import canonical_session_id
from app.agents.contact_us.modules import AGENT_KEY_PATTERN, canonical_agent_key
from app.agents.contact_us.schemas import (
    ChatHistoryDetail,
    ChatHistoryMessage,
    ChatHistorySession,
    ChatSessionCreated,
    ChatSessionRequest,
    DeleteHistoryRequest,
    DeleteHistoryResult,
)
from app.api.routes.contact_us import _auth
from app.history import HistoryUnavailable, history_store, session_uuid

router = APIRouter(prefix="/agents", tags=["agents"])

_UNAVAILABLE = "Chat history is unavailable right now."


def _owner(authorization: str | None) -> str:
    return owner_fingerprint(_auth(authorization, None, None, None).bearer_token)


@router.post(
    "/contact-us/session",
    response_model=ChatSessionCreated,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_chat_session(request: ChatSessionRequest, authorization: str | None = Header(None)) -> ChatSessionCreated:
    """New Chat: a new conversation with its own session_id. Works without history too."""
    owner, agent = _owner(authorization), canonical_agent_key(request.agent_key)
    session_id = await sessions.create(owner, agent)
    created_at = await history_store.create_session(session_id, owner, agent)
    return ChatSessionCreated(session_id=session_id, created_at=created_at or datetime.now(timezone.utc))


@router.get("/contact-us/history", response_model=list[ChatHistorySession], response_model_by_alias=True)
async def list_chat_history(
    agent_key: str | None = Query(None, alias="agentKey", pattern=AGENT_KEY_PATTERN),
    limit: int = Query(100, ge=1, le=200),
    authorization: str | None = Header(None),
) -> list[ChatHistorySession]:
    try:
        found = await history_store.list_sessions(_owner(authorization), canonical_agent_key(agent_key), limit)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from exc
    return [ChatHistorySession(**vars(s)) for s in found]


@router.get("/contact-us/history/{session_id}", response_model=ChatHistoryDetail, response_model_by_alias=True)
async def get_chat_history(
    session_id: str,
    agent_key: str | None = Query(None, alias="agentKey", pattern=AGENT_KEY_PATTERN),
    authorization: str | None = Header(None),
) -> ChatHistoryDetail:
    try:
        found = await history_store.get_session(session_id, _owner(authorization), canonical_agent_key(agent_key))
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from exc
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    session, messages = found
    return ChatHistoryDetail(
        **vars(session),
        messages=[
            ChatHistoryMessage(
                id=m.id, role=m.role, content=m.content, result=m.result, error=m.is_error, created_at=m.created_at
            )
            for m in messages
        ],
    )


@router.delete("/contact-us/history", response_model=DeleteHistoryResult, response_model_by_alias=True)
async def delete_chat_history(request: DeleteHistoryRequest, authorization: str | None = Header(None)) -> DeleteHistoryResult:
    """Permanently deletes the selected conversations (by session_id) and their messages."""
    if any(session_uuid(sid) is None for sid in request.session_ids):
        raise HTTPException(422, detail="Invalid conversation id.")
    owner, agent = _owner(authorization), canonical_agent_key(request.agent_key)
    try:
        deleted = await history_store.delete_sessions(request.session_ids, owner, agent)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from exc
    # A deleted conversation can't be continued either: drop its live session too
    # (unless a turn is running in it; its reply then has nowhere to be recorded).
    for session_id in deleted:
        live = canonical_session_id(session_id)
        if sessions.owner_of(live) == owner and sessions.begin(live):
            sessions.end(live)
            await sessions.delete(live)
    return DeleteHistoryResult(deleted_session_ids=deleted, deleted_count=len(deleted))
