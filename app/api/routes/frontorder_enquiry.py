"""Frontend Order Enquiry Agent endpoints.

Run events are streamed by the shared `GET /agents/runs/{run_id}/events` endpoint
(the run registry is shared by all agents), so only the start endpoint lives here.
Chat history endpoints: frontorder_enquiry_history.py."""

import asyncio

from fastapi import APIRouter, Header, HTTPException, status

from app.agents.frontorder_enquiry import plan, run_frontorder_enquiry_agent
from app.agents.frontorder_enquiry.graph import FRONTORDER_ENQUIRY_GRAPH
from app.agents.frontorder_enquiry.history import AGENT_KEY, owner_fingerprint, restore_conversation, sessions
from app.agents.frontorder_enquiry.schemas import FrontOrderEnquiryRunRequest, RunStarted
from app.agents.runtime import registry
from app.history import session_uuid
from app.integrations.bootog import BootogClient, auth_from_headers
from app.integrations.bootog.order_enquiry import is_guid

router = APIRouter(prefix="/agents", tags=["agents"])


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
    auth = auth_from_headers(authorization, x_tenant_id, x_role_id, x_device_id)
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
