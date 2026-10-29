"""LangChain tools over the Bootog APIs this agent reads.

This is the seam between the workflow and Bootog: nodes only call these tools, and the
tools only call `OrderEnquiryApi` (app/integrations/bootog/order_enquiry.py). To move to MCP later, re-implement
these tool bodies (or swap them for MCP tools with the same names/arguments); nodes and
graph stay as is.

As in the Contact Us agent, the run's Bootog client is injected through
RunnableConfig (`configurable.bootog_client`), so auth never appears in tool
arguments. The run monitor (`configurable.monitor`) is optional. Nodes call these
deterministically today; the same tools can be bound to an LLM later.

Only read APIs are wired. Contact Log update, Change Status and Convert to Order
(all PUT on the .NET side) are deliberately absent until the follow-up workflow that
needs them is confirmed — see docs/frontend-order-inquiry-architecture.md."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.core.config import get_settings
from app.integrations.bootog.order_enquiry import (
    ORDER_ACTION_CONVERTED_TO_ORDER,
    OrderEnquiryApi,
    ProviderRef,
    created_between_filter,
    order_action_filter,
)


def _api(config: RunnableConfig) -> OrderEnquiryApi:
    configurable = config["configurable"]
    return OrderEnquiryApi(configurable["bootog_client"], configurable.get("monitor"))


def enquiry_filter(created_from: str | None, created_before: str | None, converted: bool | None) -> str | None:
    """The OrderEnquery filter for a creation-date window and an orderAction scope.
    converted: True = only orderAction 6, False = everything else, None = all."""
    clauses: list[str] = []
    if created_from and created_before:
        clauses.append(created_between_filter(datetime.fromisoformat(created_from), datetime.fromisoformat(created_before)))
    if converted is not None:
        clauses.append(order_action_filter(ORDER_ACTION_CONVERTED_TO_ORDER, include=converted))
    return ",".join(clauses) or None


def _provider(p) -> dict[str, str | None]:
    return {"providerId": p.provider_id, "providerName": p.provider_name, "location": p.location}


# --- provider name -> provider id -------------------------------------------------------
# The name the user typed is only a lookup key. The id sent to the enquiry API always
# comes from Bootog's own provider records (Provider/GetProviders), never from the LLM
# and never from a list kept in this code. Nothing is cached: each lookup is one or two
# small filtered requests, so a renamed or new provider is found on the next message.
#
# Bootog matches names with a literal "contains" (ILIKE '%text%'), so punctuation that
# differs between what was typed and what is stored ("Inc." vs "Inc", a missing comma)
# would miss. The lookup therefore:
# 1. searches for the name without punctuation;
# 2. if that finds nothing, searches for its first word and compares the results
#    locally, ignoring case, spacing and punctuation.
#
# Rules on the results:
# - exactly one provider whose normalized name equals the text -> resolved;
# - otherwise exactly one candidate -> resolved;
# - several candidates -> ambiguous: the user is asked to choose; nothing is picked;
# - none -> not found: no enquiry request is made for an unknown provider.

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


@tool
async def resolve_provider_by_name(config: RunnableConfig, provider_name: str) -> dict[str, Any]:
    """Look a provider name up in Bootog's provider list.
    Returns {"status": "resolved"|"ambiguous"|"not_found", "provider": {...}|None,
    "candidates": [...], "totalMatches": n}. Never returns a guessed id."""
    settings = get_settings()
    match = await resolve_provider_name(_api(config), provider_name, settings.frontorder_enquiry_provider_match_limit)
    return {
        "status": match.status,
        "provider": _provider(match.provider) if match.provider else None,
        "candidates": [_provider(p) for p in match.candidates],
        "totalMatches": match.total_matches,
    }


@tool
async def verify_provider(config: RunnableConfig, provider_id: str) -> dict[str, str] | None:
    """The provider with this id from Bootog's provider list, or None when it doesn't exist."""
    provider = await _api(config).get_provider(provider_id)
    return _provider(provider) if provider else None


@tool
async def get_provider_enquiry_summary(
    config: RunnableConfig,
    page: int,
    created_from: str | None = None,
    created_before: str | None = None,
    converted: bool | None = None,
) -> dict[str, Any]:
    """One page of providers ranked by how many matching enquiries they have
    (GET OrderEnquery/GetAllOrderEnquery). Counts are the server's own totals."""
    settings = get_settings()
    summary = await _api(config).get_provider_summary(
        page=page,
        page_size=settings.frontorder_enquiry_summary_page_size,
        filter_expr=enquiry_filter(created_from, created_before, converted),
    )
    return {
        "totalProviders": summary.total_providers,
        "pageSize": settings.frontorder_enquiry_summary_page_size,
        "providers": [
            {"providerId": p.provider_id, "providerName": p.provider_name, "count": p.count} for p in summary.providers
        ],
    }


@tool
async def get_order_enquiries(
    config: RunnableConfig,
    target_provider_id: str,
    created_from: str | None = None,
    created_before: str | None = None,
    converted: bool | None = None,
    page: int | None = None,
) -> dict[str, Any]:
    """One provider's enquiries, newest first (GET OrderEnquery).

    created_from / created_before: ISO datetimes with a timezone; enquiries created in
    [created_from, created_before). Both or neither.
    converted: True = only enquiries converted to orders, False = only enquiries not
    converted, None = all. Scoping on the server keeps the returned `total` exact.
    page: one page of `FRONTORDER_ENQUIRY_PAGE_SIZE` records; None = every page up to
    the page cap (for checks that must see every record)."""
    settings = get_settings()
    api = _api(config)
    filter_expr = enquiry_filter(created_from, created_before, converted)
    if page is None:
        result, truncated = await api.list_all_enquiries(
            page_size=settings.frontorder_enquiry_page_size,
            max_pages=settings.frontorder_enquiry_max_pages,
            filter_expr=filter_expr,
            target_provider_id=target_provider_id,
        )
        return {"total": result.total, "records": result.records, "truncated": truncated, "page": None}
    result = await api.list_enquiries(
        page=page,
        page_size=settings.frontorder_enquiry_page_size,
        filter_expr=filter_expr,
        target_provider_id=target_provider_id,
    )
    has_more = (page - 1) * settings.frontorder_enquiry_page_size + len(result.records) < result.total
    return {"total": result.total, "records": result.records, "truncated": has_more, "page": page}


@tool
async def get_enquiry_contact_history(config: RunnableConfig, enquiry_id: str) -> list[dict[str, Any]]:
    """Contact Log / tracking history rows of one order enquiry, newest first."""
    return await _api(config).list_tracking_history(enquiry_id)
