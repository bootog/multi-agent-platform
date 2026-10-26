"""Scheduling Agent endpoints: chat turns plus chat history (same contracts as the Contact
agents' chat, under /agents/scheduling). Turn activity and the final reply stream on the
shared GET /agents/runs/{runId}/events."""

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import Field

from app.agents.contact_us.agent_graph import owner_fingerprint
from app.agents.contact_us.history import canonical_session_id, record_turn, restore_session
from app.agents.contact_us.schemas import (
    CamelModel,
    ChatHistoryDetail,
    ChatHistoryMessage,
    ChatHistorySession,
    ChatSessionCreated,
    ChatTurnStarted,
    DeleteHistoryResult,
)
from app.agents.runtime import registry
from app.agents.scheduling import AGENT_KEY, SCHEDULING_GRAPH, sessions
from app.agents.scheduling.graph import start_turn_task
from app.api.routes.contact_us import _auth
from app.history import HistoryUnavailable, history_store, session_uuid
from app.integrations.bootog import BootogClient

router = APIRouter(prefix="/agents/scheduling", tags=["agents"])

_UNAVAILABLE = "Chat history is unavailable right now."
_ID = r"^[0-9a-fA-F-]{32,36}$"


class ChatAction(CamelModel):
    """A chat card's button. Ids are only ever ones the agent showed; the backend
    re-validates every one of them against Bootog before acting."""

    type: Literal["confirm", "cancel", "select_order", "assign", "reassign", "producers", "availability", "schedule",
                  "list_status", "page", "add_log", "logs", "summary"]
    action_id: str | None = Field(None, max_length=64, pattern=r"^[0-9a-f]{32}$")
    order_id: str | None = Field(None, pattern=_ID)
    producer_id: str | None = Field(None, pattern=_ID)
    date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    start: str | None = Field(None, max_length=10, pattern=r"^\d{1,2}:\d{2}\s?[AaPp][Mm]$")
    status_name: str | None = Field(None, max_length=60)
    page: int | None = Field(None, ge=1, le=10000)
    channel: Literal["phone_call", "sms", "email", "note"] | None = None
    max_distance_miles: float | None = Field(None, gt=0, le=500)


class SchedulingChatRequest(CamelModel):
    session_id: str | None = Field(None, max_length=64)
    message: str | None = Field(None, max_length=4000)
    action: ChatAction | None = None
    # The browser's clock: minutes to add to UTC (e.g. -240 for EDT, 330 for IST).
    utc_offset_minutes: int = Field(0, ge=-14 * 60, le=14 * 60)


class DeleteRequest(CamelModel):
    session_ids: list[str] = Field(..., min_length=1, max_length=100)


def _owner(authorization: str | None) -> str:
    return owner_fingerprint(_auth(authorization, None, None, None).bearer_token)


@router.post("/chat", response_model=ChatTurnStarted, response_model_by_alias=True, status_code=status.HTTP_202_ACCEPTED)
async def send_scheduling_chat_message(
    
    request: SchedulingChatRequest,   
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None),
    x_role_id: str | None = Header(None),
    x_device_id: str | None = Header(None),
) -> ChatTurnStarted:
    auth = _auth(authorization, x_tenant_id, x_role_id, x_device_id)
    owner = owner_fingerprint(auth.bearer_token)
    message = (request.message or "").strip() or None
    if not message and request.action is None:
        raise HTTPException(422, detail="Send a message or an action.")

    session_id, restarted = canonical_session_id(request.session_id), False
    if session_id and sessions.owner_of(session_id) is None:
        if not await restore_session(session_id, owner, AGENT_KEY, graph=SCHEDULING_GRAPH, registry=sessions):
            session_id, restarted = None, True
    if session_id is None:
        session_id = await sessions.create(owner, AGENT_KEY)
    elif sessions.owner_of(session_id) != owner:
        # 403, not 401: the host's AuthInterceptor opens a login popup on any 401.
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="This conversation belongs to another user.")
    elif sessions.agent_of(session_id) != AGENT_KEY:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="This conversation belongs to another agent.")
    if not sessions.begin(session_id):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="The agent is still working on your previous message.")

    action: dict[str, Any] | None = request.action.model_dump(by_alias=True, exclude_none=True) if request.action else None
    channel = registry.create(AGENT_KEY)
    turn = start_turn_task(channel, session_id, message, BootogClient(auth), action, request.utc_offset_minutes)
    registry.track(turn)
    registry.track(record_turn(turn, channel, session_id, owner, AGENT_KEY, message))
    return ChatTurnStarted(run_id=channel.run_id, session_id=session_id, session_restarted=restarted)


@router.delete("/chat/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_scheduling_chat(session_id: str, authorization: str | None = Header(None)) -> Response:
    owner = sessions.owner_of(session_id)
    if owner is not None:
        if owner != _owner(authorization):
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="This conversation belongs to another user.")
        await sessions.delete(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/session", response_model=ChatSessionCreated, response_model_by_alias=True, status_code=status.HTTP_201_CREATED)
async def create_scheduling_session(authorization: str | None = Header(None)) -> ChatSessionCreated:
    owner = _owner(authorization)
    session_id = await sessions.create(owner, AGENT_KEY)
    created_at = await history_store.create_session(session_id, owner, AGENT_KEY)
    return ChatSessionCreated(session_id=session_id, created_at=created_at or datetime.now(timezone.utc))


@router.get("/history", response_model=list[ChatHistorySession], response_model_by_alias=True)
async def list_scheduling_history(
    limit: int = Query(100, ge=1, le=200), authorization: str | None = Header(None)
) -> list[ChatHistorySession]:
    try:
        found = await history_store.list_sessions(_owner(authorization), AGENT_KEY, limit)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from exc
    return [ChatHistorySession(**vars(s)) for s in found]


@router.get("/history/{session_id}", response_model=ChatHistoryDetail, response_model_by_alias=True)
async def get_scheduling_history(session_id: str, authorization: str | None = Header(None)) -> ChatHistoryDetail:
    try:
        found = await history_store.get_session(session_id, _owner(authorization), AGENT_KEY)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from exc
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    session, messages = found
    return ChatHistoryDetail(
        **vars(session),
        messages=[
            ChatHistoryMessage(id=m.id, role=m.role, content=m.content, result=m.result, error=m.is_error, created_at=m.created_at)
            for m in messages
        ],
    )


@router.delete("/history", response_model=DeleteHistoryResult, response_model_by_alias=True)
async def delete_scheduling_history(request: DeleteRequest, authorization: str | None = Header(None)) -> DeleteHistoryResult:
    if any(session_uuid(sid) is None for sid in request.session_ids):
        raise HTTPException(422, detail="Invalid conversation id.")
    owner = _owner(authorization)
    try:
        deleted = await history_store.delete_sessions(request.session_ids, owner, AGENT_KEY)
    except HistoryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from exc
    for session_id in deleted:
        live = canonical_session_id(session_id)
        if sessions.owner_of(live) == owner and sessions.begin(live):
            sessions.end(live)
            await sessions.delete(live)
    return DeleteHistoryResult(deleted_session_ids=deleted, deleted_count=len(deleted))
