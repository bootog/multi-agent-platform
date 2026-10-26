"""Frontend Order Enquiry Agent chat history: the list of the caller's conversations, one
conversation's messages, and deleting conversations. Stored in the AGENT_DB_NAME
database, tables frontorder_enquiry_sessions / frontorder_enquiry_messages; always
limited to the signed-in user."""

from fastapi import APIRouter, Header, HTTPException, Query, status

from app.agents.frontorder_enquiry.history import AGENT_KEY, owner_fingerprint, sessions
from app.agents.frontorder_enquiry.schemas import (
    DeleteHistoryRequest,
    DeleteHistoryResult,
    HistoryDetailOut,
    HistoryMessageOut,
    HistorySessionOut,
)
from app.history import HistoryUnavailable, frontorder_history_store, session_uuid
from app.integrations.bootog import auth_from_headers

router = APIRouter(prefix="/agents", tags=["agents"])

_HISTORY_UNAVAILABLE = "Chat history is unavailable right now."


def _owner(authorization: str | None) -> str:
    """The signed-in user's opaque id; history is always limited to that user."""
    token = auth_from_headers(authorization, None, None, None).bearer_token
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Sign in again: no Bootog access token was sent.")
    return owner_fingerprint(token)


@router.get("/frontorder-enquiry/history", response_model=list[HistorySessionOut], response_model_by_alias=True)
async def list_frontorder_enquiry_history(
    limit: int = Query(100, ge=1, le=200),
    authorization: str | None = Header(None),
) -> list[HistorySessionOut]:
    """The signed-in user's previous conversations with this agent, newest first."""
    try:
        found = await frontorder_history_store.list_sessions(_owner(authorization), AGENT_KEY, limit)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_HISTORY_UNAVAILABLE) from exc
    return [HistorySessionOut(**vars(s)) for s in found]


@router.get(
    "/frontorder-enquiry/history/{session_id}", response_model=HistoryDetailOut, response_model_by_alias=True
)
async def get_frontorder_enquiry_history(session_id: str, authorization: str | None = Header(None)) -> HistoryDetailOut:
    """Every message of one conversation, oldest first, with the result cards of each reply."""
    try:
        found = await frontorder_history_store.get_session(session_id, _owner(authorization), AGENT_KEY)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_HISTORY_UNAVAILABLE) from exc
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    session, messages = found
    return HistoryDetailOut(
        **vars(session),
        messages=[
            HistoryMessageOut(
                id=m.id, role=m.role, content=m.content, result=m.result, error=m.is_error, created_at=m.created_at
            )
            for m in messages
        ],
    )


@router.delete("/frontorder-enquiry/history", response_model=DeleteHistoryResult, response_model_by_alias=True)
async def delete_frontorder_enquiry_history(
    request: DeleteHistoryRequest, authorization: str | None = Header(None)
) -> DeleteHistoryResult:
    """Permanently deletes the selected conversations and their messages."""
    if any(session_uuid(sid) is None for sid in request.session_ids):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Invalid conversation id.")
    owner = _owner(authorization)
    try:
        deleted = await frontorder_history_store.delete_sessions(request.session_ids, owner, AGENT_KEY)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_HISTORY_UNAVAILABLE) from exc
    # A deleted conversation can't be continued either: drop its live session too
    # (unless the agent is still answering in it).
    for session_id in deleted:
        if sessions.owner_of(session_id) == owner and sessions.begin(session_id):
            sessions.end(session_id)
            await sessions.delete(session_id)
    return DeleteHistoryResult(deleted_session_ids=deleted, deleted_count=len(deleted))
