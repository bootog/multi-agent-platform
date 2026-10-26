"""Frontend Order Enquiry Agent endpoints.

Run events are streamed by the shared `GET /agents/runs/{run_id}/events` endpoint
(the run registry is shared by all agents), so only the start endpoint and the chat
history endpoints live here."""

import asyncio

from fastapi import APIRouter, Header, HTTPException, Query, status

from app.agents.contact_us.schemas import (
    ChatHistoryDetail,
    ChatHistoryMessage,
    ChatHistorySession,
    DeleteHistoryRequest,
    DeleteHistoryResult,
)
from app.agents.frontorder_enquiry import plan, run_frontorder_enquiry_agent
from app.agents.frontorder_enquiry.api_client import is_guid
from app.agents.frontorder_enquiry.graph import FRONTORDER_ENQUIRY_GRAPH
from app.agents.frontorder_enquiry.history import restore_conversation
from app.agents.frontorder_enquiry.schemas import FrontOrderEnquiryRunRequest, RunStarted
from app.agents.frontorder_enquiry.sessions import AGENT_KEY, owner_fingerprint, sessions
from app.agents.runtime import registry
from app.api.routes.contact_us import _auth
from app.history import HistoryUnavailable, frontorder_history_store, session_uuid
from app.integrations.bootog import BootogClient

router = APIRouter(prefix="/agents", tags=["agents"])

_HISTORY_UNAVAILABLE = "Chat history is unavailable right now."


@router.post(
    "/frontorder-enquiry/runs",
    response_model=RunStarted,
    response_model_by_alias=True,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_frontorder_enquiry_run(
    request: FrontOrderEnquiryRunRequest | None = None,
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None),
    x_role_id: str | None = Header(None),
    x_device_id: str | None = Header(None),
) -> RunStarted:
    """Start a run (one chat message). Body (all optional):
    {"message": "...", "sessionId": "<from the previous RunStarted>", "targetProviderId": "<guid>", "page": 2}.
    No body = "Run Agent": the provider summary of open enquiries, all dates."""
    auth = _auth(authorization, x_tenant_id, x_role_id, x_device_id)
    # Enquiries are read with the signed-in admin's own Bootog token, so Bootog applies
    # that admin's permissions. Without one, BootogClient would fall back to the
    # optional BOOTOG_API_TOKEN service token and an anonymous caller would read
    # enquiries with the service account's rights — so the request is refused instead.
    if not auth.bearer_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Sign in again: no Bootog access token was sent.")

    request = request or FrontOrderEnquiryRunRequest()
    provider_id = (request.target_provider_id or "").strip() or None
    if provider_id and not is_guid(provider_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="targetProviderId must be a GUID.")
    message = (request.message or "").strip() or None

    # Conversation: continue the caller's own session, or start a new one.
    owner = owner_fingerprint(auth.bearer_token)
    requested = session_uuid(request.session_id)
    session_id = requested.hex if requested else None
    if session_id and sessions.owner_of(session_id) is None:
        # Not live any more (backend restart, 4 h idle, or opened from the History panel):
        # continue it from the stored chat history when it is this caller's conversation.
        if not await restore_conversation(session_id, owner, FRONTORDER_ENQUIRY_GRAPH):
            session_id = None
    restarted = bool(request.session_id) and session_id is None
    if session_id is None:
        session_id = await sessions.create(owner, AGENT_KEY)
    elif sessions.owner_of(session_id) != owner or sessions.agent_of(session_id) != AGENT_KEY:
        # 403, not 401: the host's AuthInterceptor opens a login popup on any 401.
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="This conversation belongs to another user.")
    if not sessions.begin(session_id):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="The agent is still working on your previous message.")

    channel = registry.create(AGENT_KEY)
    client = BootogClient(auth)
    registry.track(
        asyncio.create_task(
            run_frontorder_enquiry_agent(
                channel,
                client,
                message,
                target_provider_id=provider_id,
                session_id=session_id,
                page=request.page,
                owner_id=owner,
            )
        )
    )
    return RunStarted(run_id=channel.run_id, steps=plan(), session_id=session_id, session_restarted=restarted)


# --- chat history (PostgreSQL tables frontorder_enquiry_sessions / _messages) ---


def _owner(authorization: str | None) -> str:
    """The signed-in user's opaque id; history is always limited to that user."""
    token = _auth(authorization, None, None, None).bearer_token
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Sign in again: no Bootog access token was sent.")
    return owner_fingerprint(token)


@router.get("/frontorder-enquiry/history", response_model=list[ChatHistorySession], response_model_by_alias=True)
async def list_frontorder_enquiry_history(
    limit: int = Query(100, ge=1, le=200),
    authorization: str | None = Header(None),
) -> list[ChatHistorySession]:
    """The signed-in user's previous conversations with this agent, newest first."""
    try:
        found = await frontorder_history_store.list_sessions(_owner(authorization), AGENT_KEY, limit)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_HISTORY_UNAVAILABLE) from exc
    return [ChatHistorySession(**vars(s)) for s in found]


@router.get(
    "/frontorder-enquiry/history/{session_id}", response_model=ChatHistoryDetail, response_model_by_alias=True
)
async def get_frontorder_enquiry_history(session_id: str, authorization: str | None = Header(None)) -> ChatHistoryDetail:
    """Every message of one conversation, oldest first, with the result cards of each reply."""
    try:
        found = await frontorder_history_store.get_session(session_id, _owner(authorization), AGENT_KEY)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_HISTORY_UNAVAILABLE) from exc
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
