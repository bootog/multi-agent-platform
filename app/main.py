from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.contact_us import router as contact_us_router
from app.api.routes.frontorder_enquiry import router as frontorder_enquiry_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.route.agents import router as agents_router

configure_logging()

app = FastAPI(title="Multi-Agent Platform")

# The Angular host calls this API with credentials, so origins must be listed
# explicitly ("*" is not allowed with credentials). Override with ALLOWED_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents_router, prefix="/api/v1")
app.include_router(contact_us_router, prefix="/api/v1")
app.include_router(frontorder_enquiry_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
