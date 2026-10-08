"""Frontend Order Enquiry Agent settings.

Bootog base URL, token and timeout come from the shared `app.core.config` settings
(via BootogClient). Only the settings specific to this agent live here."""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class FrontOrderEnquirySettings:
    # Enquiries requested per OrderEnquery page (newest first).
    page_size: int
    # Upper bound on pages read per run, so one run never walks an unbounded list.
    max_pages: int


@lru_cache
def get_frontorder_enquiry_settings() -> FrontOrderEnquirySettings:
    return FrontOrderEnquirySettings(
        page_size=max(1, int(os.getenv("FRONTORDER_ENQUIRY_PAGE_SIZE", "50"))),
        max_pages=max(1, int(os.getenv("FRONTORDER_ENQUIRY_MAX_PAGES", "5"))),
    )
