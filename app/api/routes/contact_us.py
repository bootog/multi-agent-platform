import asyncio
import json

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from app.agents.contact_us import plan_for, run_contact_us_agent
from app.agents.contact_us.schemas import ContactUsRunRequest, ProviderOut, RunStarted
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
