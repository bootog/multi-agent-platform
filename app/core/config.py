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

    # OpenAI. The chat model drives LangGraph reasoning (understanding, extraction,
    # replies). OPENAI_REALTIME_MODEL is reserved for realtime/voice features and is
    # deliberately NOT used for text reasoning.
    openai_api_key: str | None = field(default=None, repr=False)
    openai_chat_model: str = "gpt-4.1-mini"
    # None -> the model's default (reasoning models reject a temperature).
    openai_temperature: float | None = 0.2
    openai_timeout_seconds: float = 30.0
    openai_realtime_model: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings(
        bootog_api_base_url=os.getenv("BOOTOG_API_BASE_URL", "https://api.dev.bootog.com/api/v1").rstrip("/"),
        bootog_api_token=os.getenv("BOOTOG_API_TOKEN") or None,
        bootog_timeout_seconds=float(os.getenv("BOOTOG_TIMEOUT_SECONDS", "20")),
        contact_us_page_size=int(os.getenv("CONTACT_US_PAGE_SIZE", "50")),
        allowed_origins=_csv(os.getenv("ALLOWED_ORIGINS", "http://localhost:4201")),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL") or "gpt-4.1-mini",
        openai_temperature=_float_or_none(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
        openai_realtime_model=os.getenv("OPENAI_REALTIME_MODEL") or None,
    )
