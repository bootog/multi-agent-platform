"""Provider name -> provider id, using Bootog's provider list at request time.

The name the user typed is only a lookup key. The id sent to the enquiry API always
comes from Bootog's own provider records (Provider/GetProviders), never from the LLM
and never from a list kept in this code. Nothing is cached: each lookup is one or two
small filtered requests, so a renamed or new provider is found on the next message.

Bootog matches names with a literal "contains" (ILIKE '%text%'), so punctuation that
differs between what was typed and what is stored ("Inc." vs "Inc", a missing comma)
would miss. The lookup therefore:
1. searches for the name without punctuation;
2. if that finds nothing, searches for its first word and compares the results
   locally, ignoring case, spacing and punctuation.

Rules on the results:
- exactly one provider whose normalized name equals the text -> resolved;
- otherwise exactly one candidate -> resolved;
- several candidates -> ambiguous: the user is asked to choose; nothing is picked;
- none -> not found: no enquiry request is made for an unknown provider."""

import re
from dataclasses import dataclass, field
from typing import Literal

from app.agents.frontorder_enquiry.api_client import OrderEnquiryApi, ProviderRef

MatchStatus = Literal["resolved", "ambiguous", "not_found"]
# Rows read by the broader first-word search (step 2), compared locally.
BROAD_SEARCH_LIMIT = 50


@dataclass(frozen=True)
class ProviderMatch:
    status: MatchStatus
    provider: ProviderRef | None = None
    candidates: list[ProviderRef] = field(default_factory=list)
    # Number of providers matching on the server (may be more than `candidates`).
    total_matches: int = 0


def normalize_name(name: str) -> str:
    """"Inspection Depot, Inc." and "inspection depot inc" compare equal."""
    return " ".join(re.sub(r"[^\w\s&]", " ", name.casefold()).split())


def _search_text(name: str) -> str:
    return " ".join(re.sub(r"[^\w\s&]", " ", name).split())


async def resolve_provider_name(api: OrderEnquiryApi, name: str, limit: int) -> ProviderMatch:
    wanted = normalize_name(name)
    if not wanted:
        return ProviderMatch("not_found")

    found = await api.search_providers(_search_text(name), limit)
    providers, total = found.providers, found.total
    if not providers:
        first_word = wanted.split()[0]
        if len(first_word) < 3 or first_word == wanted:
            return ProviderMatch("not_found")
        broad = await api.search_providers(first_word, BROAD_SEARCH_LIMIT)
        providers = [p for p in broad.providers if wanted in normalize_name(p.provider_name)]
        total = len(providers)
        if not providers:
            return ProviderMatch("not_found")

    exact = [p for p in providers if normalize_name(p.provider_name) == wanted]
    if len(exact) == 1:
        return ProviderMatch("resolved", provider=exact[0], total_matches=total)
    if len(exact) > 1:
        # Two providers with the same name: only the user can tell which one.
        return ProviderMatch("ambiguous", candidates=exact, total_matches=len(exact))
    if total == 1 and len(providers) == 1:
        return ProviderMatch("resolved", provider=providers[0], total_matches=1)
    candidates = sorted(providers, key=lambda p: p.provider_name.casefold())[:limit]
    return ProviderMatch("ambiguous", candidates=candidates, total_matches=total)
