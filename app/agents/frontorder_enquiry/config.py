"""Frontend Order Enquiry Agent settings.

Bootog base URL, token and timeout come from the shared `app.core.config` settings
(via BootogClient). Only the settings specific to this agent live here. No provider
or business data is configured here: providers and counts always come from Bootog."""

import os
from dataclasses import dataclass
from functools import lru_cache
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class FrontOrderEnquirySettings:
    # Enquiries per page of one provider's enquiry list (newest first).
    page_size: int
    # Upper bound on pages read when a request must check every enquiry (follow-up
    # requests read the Contact Log of each one), so a run never walks an unbounded list.
    max_pages: int
    # Business timezone: "today" and calendar dates in chat requests mean days in this
    # zone. Bootog stores createdDateTime in UTC, so day bounds are converted.
    timezone: ZoneInfo
    # Longest date range one request may ask for (inclusive days).
    max_range_days: int
    # Providers per page of the provider summary ("how many enquiries today?").
    summary_page_size: int
    # Most providers offered when a provider name matches several.
    provider_match_limit: int


@lru_cache
def get_frontorder_enquiry_settings() -> FrontOrderEnquirySettings:
    return FrontOrderEnquirySettings(
        page_size=max(1, int(os.getenv("FRONTORDER_ENQUIRY_PAGE_SIZE", "50"))),
        max_pages=max(1, int(os.getenv("FRONTORDER_ENQUIRY_MAX_PAGES", "5"))),
        timezone=ZoneInfo(os.getenv("FRONTORDER_ENQUIRY_TIMEZONE") or "Asia/Kolkata"),
        max_range_days=max(1, int(os.getenv("FRONTORDER_ENQUIRY_MAX_RANGE_DAYS", "366"))),
        summary_page_size=max(1, int(os.getenv("FRONTORDER_ENQUIRY_SUMMARY_PAGE_SIZE", "10"))),
        provider_match_limit=max(2, int(os.getenv("FRONTORDER_ENQUIRY_PROVIDER_MATCH_LIMIT", "10"))),
    )
