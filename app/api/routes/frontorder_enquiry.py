"""Frontend Order Enquiry Agent endpoints.

Run events are streamed by the shared `GET /agents/runs/{run_id}/events` endpoint
(the run registry is shared by all agents), so only the start endpoint lives here."""

import asyncio

from fastapi import APIRouter, Header, status

from app.agents.frontorder_enquiry import plan, run_frontorder_enquiry_agent
from app.agents.frontorder_enquiry.schemas import RunStarted
from app.agents.runtime import registry
from app.integrations.bootog import BootogAuth, BootogClient

router = APIRouter(prefix="/agents", tags=["agents"])


def _auth(authorization: str | None, tenant_id: str | None, role_id: str | None, device_id: str | None) -> BootogAuth:
    """Forward the caller's Bootog identity (same headers the Angular host sends)."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip() or None
    return BootogAuth(bearer_token=token, tenant_id=tenant_id, role_id=role_id, device_id=device_id)


@router.post(
    "/frontorder-enquiry/runs",
    response_model=RunStarted,
    response_model_by_alias=True,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_frontorder_enquiry_run(
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None),
    x_role_id: str | None = Header(None),
    x_device_id: str | None = Header(None),
) -> RunStarted:
    channel = registry.create("frontorder-enquiry")
    client = BootogClient(_auth(authorization, x_tenant_id, x_role_id, x_device_id))
    registry.track(asyncio.create_task(run_frontorder_enquiry_agent(channel, client)))
    return RunStarted(run_id=channel.run_id, steps=plan())
