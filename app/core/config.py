import os
from dataclasses import dataclass, field
from functools import lru_cache
from zoneinfo import ZoneInfo

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
    # Scheduling Agent: orders per page in chat, days of appointment slots offered, and
    # the producer search radius (miles) around the property.
    scheduling_page_size: int = 10
    scheduling_slot_days: int = 5
    scheduling_producer_radius_miles: int = 30
    # Multi-order "nearest eligible" search: when nothing eligible is found, the radius grows
    # by this step until a producer is found or the maximum is reached.
    scheduling_radius_step_miles: int = 30
    scheduling_max_radius_miles: int = 120
    # Most orders one multi-order assignment may cover (each order is validated and written
    # on its own, so this bounds the API calls one confirmation can trigger).
    scheduling_max_batch_orders: int = 50
    # Most days of appointment slots one request may ask for, and open slots shown per day.
    scheduling_max_slot_days: int = 14
    scheduling_slots_per_day: int = 8
    # Orders read per GetOrderDetails call when the chat lists every matching order.
    scheduling_order_fetch_page_size: int = 50
    # Latest chat messages sent to the LLM to understand a message / to write the reply.
    scheduling_understand_history_messages: int = 12
    scheduling_reply_history_messages: int = 8
    # Frontend Order Enquiry Agent: enquiries per page of one provider's list; page cap when
    # every enquiry must be checked (follow-up requests); business timezone for "today" and
    # dates in chat (Bootog stores UTC); longest date range one request may ask for;
    # providers per page of the provider summary; most providers offered for an unclear name.
    frontorder_enquiry_page_size: int = 50
    frontorder_enquiry_max_pages: int = 5
    frontorder_enquiry_timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo("Asia/Kolkata"))
    frontorder_enquiry_max_range_days: int = 366
    frontorder_enquiry_summary_page_size: int = 10
    frontorder_enquiry_provider_match_limit: int = 10

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
        scheduling_page_size=int(os.getenv("SCHEDULING_PAGE_SIZE", "10")),
        scheduling_slot_days=int(os.getenv("SCHEDULING_SLOT_DAYS", "5")),
        scheduling_producer_radius_miles=int(os.getenv("SCHEDULING_PRODUCER_RADIUS_MILES", "30")),
        scheduling_radius_step_miles=int(os.getenv("SCHEDULING_RADIUS_STEP_MILES", "30")),
        scheduling_max_radius_miles=int(os.getenv("SCHEDULING_MAX_RADIUS_MILES", "120")),
        scheduling_max_batch_orders=int(os.getenv("SCHEDULING_MAX_BATCH_ORDERS", "50")),
        scheduling_max_slot_days=int(os.getenv("SCHEDULING_MAX_SLOT_DAYS", "14")),
        scheduling_slots_per_day=int(os.getenv("SCHEDULING_SLOTS_PER_DAY", "8")),
        scheduling_order_fetch_page_size=int(os.getenv("SCHEDULING_ORDER_FETCH_PAGE_SIZE", "50")),
        scheduling_understand_history_messages=int(os.getenv("SCHEDULING_UNDERSTAND_HISTORY_MESSAGES", "12")),
        scheduling_reply_history_messages=int(os.getenv("SCHEDULING_REPLY_HISTORY_MESSAGES", "8")),
        frontorder_enquiry_page_size=max(1, int(os.getenv("FRONTORDER_ENQUIRY_PAGE_SIZE", "50"))),
        frontorder_enquiry_max_pages=max(1, int(os.getenv("FRONTORDER_ENQUIRY_MAX_PAGES", "5"))),
        frontorder_enquiry_timezone=ZoneInfo(os.getenv("FRONTORDER_ENQUIRY_TIMEZONE") or "Asia/Kolkata"),
        frontorder_enquiry_max_range_days=max(1, int(os.getenv("FRONTORDER_ENQUIRY_MAX_RANGE_DAYS", "366"))),
        frontorder_enquiry_summary_page_size=max(1, int(os.getenv("FRONTORDER_ENQUIRY_SUMMARY_PAGE_SIZE", "10"))),
        frontorder_enquiry_provider_match_limit=max(2, int(os.getenv("FRONTORDER_ENQUIRY_PROVIDER_MATCH_LIMIT", "10"))),
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
