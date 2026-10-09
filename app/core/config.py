import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

# Local development reads `.env`; real environment variables always win.
load_dotenv(override=False)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_or_none(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    # Bootog API (the documented dev base URL from ContactUS.pdf is the default).
    bootog_api_base_url: str
    # Optional service token for server-side runs. When a caller forwards its own
    # Bootog bearer token, that one is used instead. Never logged.
    bootog_api_token: str | None = field(repr=False)
    bootog_timeout_seconds: float
    # How many Contact Us requests the agent reads per run (latest first).
    contact_us_page_size: int
    allowed_origins: list[str]
    # Extra attempts for safe GETs (timeouts, transport errors, 502/503/504). POST/PUT never retry.
    bootog_get_retries: int = 2
    bootog_retry_backoff_seconds: float = 0.3
    # Upper bound on pages read when a lookup must page through a Bootog list
    # (requests, providers, subcategories). Hitting it is reported, never hidden.
    bootog_max_pages: int = 10
    # Sent as X-Client-Type on every Bootog call when set. NOT needed: writes pass the
    # gateway's anti-forgery check with the caller's own XSRF token (integrations/bootog/
    # client.py). Only "Mobile" is accepted (the gateway then skips that check) and only
    # with the Bootog team's authorisation; anything else is ignored with a warning.
    bootog_client_type: str | None = None

    # OpenAI. The chat model drives LangGraph reasoning (understanding, extraction,
    # replies). OPENAI_REALTIME_MODEL is reserved for realtime/voice features and is
    # deliberately NOT used for text reasoning.
    openai_api_key: str | None = field(default=None, repr=False)
    openai_chat_model: str = "gpt-4.1-mini"
    # None -> the model's default (reasoning models reject a temperature).
    openai_temperature: float | None = 0.2
    openai_timeout_seconds: float = 30.0
    openai_realtime_model: str | None = None

    # PostgreSQL server of the agent chat-history database (AGENT_DB_NAME). This is a
    # dedicated database owned by the agent platform; it is created on startup if missing.
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str | None = None
    db_password: str | None = field(default=None, repr=False)
    agent_db_name: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings(
        bootog_api_base_url=os.getenv("BOOTOG_API_BASE_URL", "https://api.dev.bootog.com/api/v1").rstrip("/"),
        bootog_api_token=os.getenv("BOOTOG_API_TOKEN") or None,
        bootog_timeout_seconds=float(os.getenv("BOOTOG_TIMEOUT_SECONDS", "20")),
        contact_us_page_size=int(os.getenv("CONTACT_US_PAGE_SIZE", "50")),
        allowed_origins=_csv(os.getenv("ALLOWED_ORIGINS", "http://localhost:4201")),
        bootog_get_retries=int(os.getenv("BOOTOG_GET_RETRIES", "2")),
        bootog_retry_backoff_seconds=float(os.getenv("BOOTOG_RETRY_BACKOFF_SECONDS", "0.3")),
        bootog_max_pages=int(os.getenv("BOOTOG_MAX_PAGES", "10")),
        bootog_client_type=(os.getenv("BOOTOG_CLIENT_TYPE") or "").strip() or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL") or "gpt-4.1-mini",
        openai_temperature=_float_or_none(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
        openai_realtime_model=os.getenv("OPENAI_REALTIME_MODEL") or None,
        db_host=os.getenv("DB_HOST") or "localhost",
        db_port=int(os.getenv("DB_PORT") or "5432"),
        db_user=os.getenv("DB_USER") or None,
        db_password=os.getenv("DB_PASSWORD") or None,
        agent_db_name=(os.getenv("AGENT_DB_NAME") or "").strip() or None,
    )
