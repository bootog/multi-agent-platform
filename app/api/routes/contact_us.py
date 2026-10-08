import asyncio
import json

from fastapi import APIRouter, Header, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from app.agents.contact_us import plan_for, run_contact_us_agent
from app.agents.contact_us.agent_graph import owner_fingerprint, sessions, start_turn_task
from app.agents.contact_us.history import canonical_session_id, record_turn, restore_session
from app.agents.contact_us.modules import canonical_agent_key
from app.agents.contact_us.schemas import ChatTurnRequest, ChatTurnStarted, ContactUsRunRequest, ProviderOut, RunStarted
from app.agents.runtime import registry
from app.integrations.bootog import BootogApiError, BootogAuth, BootogClient, ProviderApi

router = APIRouter(prefix="/agents", tags=["agents"])


def _auth(authorization: str | None, tenant_id: str | None, role_id: str | None, device_id: str | None) -> BootogAuth:
    """Forward the caller's Bootog identity (same headers the Angular host sends)."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip() or None
    return BootogAuth(bearer_token=token, tenant_id=tenant_id, role_id=role_id, device_id=device_id)


@router.get("/contact-us/providers", response_model=list[ProviderOut], response_model_by_alias=True)
async def list_providers(
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None),
    x_role_id: str | None = Header(None),
    x_device_id: str | None = Header(None),
) -> list[ProviderOut]:
    async with BootogClient(_auth(authorization, x_tenant_id, x_role_id, x_device_id)) as client:
        try:
            providers = await ProviderApi(client).list_for_session()
        except BootogApiError as exc:
            # Upstream 401 -> 403 here: the host's AuthInterceptor opens a login popup on any 401.
            code = status.HTTP_403_FORBIDDEN if exc.status_code in (401, 403) else status.HTTP_502_BAD_GATEWAY
            raise HTTPException(code, detail=f"Unable to load companies. {exc.message}") from exc
    return [ProviderOut(provider_id=p.provider_id, provider_name=p.provider_name) for p in providers]


@router.post(
    "/contact-us/runs",
    response_model=RunStarted,
    response_model_by_alias=True,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_contact_us_run(
    request: ContactUsRunRequest,
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None),
    x_role_id: str | None = Header(None),
    x_device_id: str | None = Header(None),
) -> RunStarted:
    channel = registry.create("contact-us")
    client = BootogClient(_auth(authorization, x_tenant_id, x_role_id, x_device_id))
    registry.track(asyncio.create_task(run_contact_us_agent(channel, request, client)))
    mode = "create_customer" if request.contact_request_id else "retrieve"
    return RunStarted(run_id=channel.run_id, mode=mode, steps=plan_for(mode))


@router.post(
    "/contact-us/chat",
    response_model=ChatTurnStarted,
    response_model_by_alias=True,
    status_code=status.HTTP_202_ACCEPTED,
)
async def send_contact_us_chat_message(
    request: ChatTurnRequest,
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None),
    x_role_id: str | None = Header(None),
    x_device_id: str | None = Header(None),
) -> ChatTurnStarted:
    """One chat turn. Follow its activity and final reply on GET /runs/{runId}/events."""
    auth = _auth(authorization, x_tenant_id, x_role_id, x_device_id)
    owner = owner_fingerprint(auth.bearer_token)
    agent = canonical_agent_key(request.agent_key)
    session_id, restarted = canonical_session_id(request.session_id), False
    if session_id and sessions.owner_of(session_id) is None:
        # Not live (restart/expiry/reopened from history): continue it from chat history
        # under the same id when it is this caller's conversation with this agent.
        if not await restore_session(session_id, owner, agent):
            session_id, restarted = None, True
    if session_id is None:
        session_id = await sessions.create(owner, agent)
    elif sessions.owner_of(session_id) != owner:
        # 403, not 401: the host's AuthInterceptor opens a login popup on any 401.
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="This conversation belongs to another user.")
    elif sessions.agent_of(session_id) != agent:
        # Never let one agent's messages land in another agent's conversation.
        raise HTTPException(status.HTTP_409_CONFLICT, detail="This conversation belongs to another agent.")
    if not sessions.begin(session_id):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="The agent is still working on your previous message.")

    message = (request.message or "").strip() or None
    channel = registry.create("contact-us")
    turn = start_turn_task(channel, session_id, message, BootogClient(auth), request.agent_key, request.agent_label)
    registry.track(turn)
    registry.track(record_turn(turn, channel, session_id, owner, agent, message))
    return ChatTurnStarted(run_id=channel.run_id, session_id=session_id, session_restarted=restarted)


@router.delete("/contact-us/chat/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_contact_us_chat(session_id: str, authorization: str | None = Header(None)) -> Response:
    owner = sessions.owner_of(session_id)
    if owner is not None:
        if owner != owner_fingerprint(_auth(authorization, None, None, None).bearer_token):
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="This conversation belongs to another user.")
        await sessions.delete(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str) -> StreamingResponse:
    channel = registry.get(run_id)
    if channel is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Run not found.")

    async def sse():
        async for event in channel.stream():
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
