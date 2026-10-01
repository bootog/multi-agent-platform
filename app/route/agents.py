from fastapi import APIRouter

from app.service.agents.ContactUsAgents.contact_us_agent import greet

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/contact-us/hello")
def contact_us_hello() -> dict:
    # Keep the key "reply" (not "message"): the host's AuthInterceptor
    # shows a success toast for any response body containing "message".
    return {"agent": "contact-us", "reply": greet()}
