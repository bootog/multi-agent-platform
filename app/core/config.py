import os
from dataclasses import dataclass
from functools import lru_cache


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # Bootog API (the documented dev base URL from ContactUS.pdf is the default).
    bootog_api_base_url: str
    # Optional service token for server-side runs. When a caller forwards its own
    # Bootog bearer token, that one is used instead. Never logged.
    bootog_api_token: str | None
    bootog_timeout_seconds: float
    # How many Contact Us requests the agent reads per run (latest first).
    contact_us_page_size: int
    allowed_origins: list[str]


@lru_cache
def get_settings() -> Settings:
    return Settings(
        bootog_api_base_url=os.getenv("BOOTOG_API_BASE_URL", "https://api.dev.bootog.com/api/v1").rstrip("/"),
        bootog_api_token=os.getenv("BOOTOG_API_TOKEN") or None,
        bootog_timeout_seconds=float(os.getenv("BOOTOG_TIMEOUT_SECONDS", "20")),
        contact_us_page_size=int(os.getenv("CONTACT_US_PAGE_SIZE", "50")),
        allowed_origins=_csv(os.getenv("ALLOWED_ORIGINS", "http://localhost:4201")),
    )
