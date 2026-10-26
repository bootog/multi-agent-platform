"""Bootog API calls used by the Frontend Order Enquiry Agent. Nothing else in the agent
builds a URL or a filter string.

Every contract below was verified in paperless-micro-services (file paths in
docs/frontend-order-inquiry-architecture.md, section "API reference").

1. Provider summary — GET OrderEnquery/GetAllOrderEnquery
   ?filter=...&page=P&pageSize=N&orderBy=createdDateTime desc[&targetProviderId=<guid>]
   -> {"count": <number of PROVIDER GROUPS matching the filter>,
       "data": [{"providerId", "providerName", "totalCount", "orderEnquery": [...]}]}
   - The server groups the matching enquiries by provider (SQL GROUP BY) and sorts the
     groups by count, highest first. `page`/`pageSize` page through PROVIDERS, not
     enquiries; pageSize <= 0 means every provider.
   - `totalCount` is the exact number of matching enquiries of that provider.
   - `orderEnquery` holds at most 10 of the provider's newest enquiries (fixed on the
     server), so it is never used as a record list here.
   - The endpoint is [AllowAnonymous] and applies no provider scoping (see the
     architecture doc, "Known API limitations").

2. Enquiry records — GET OrderEnquery?filter=...&page=P&pageSize=N&orderBy=...&targetProviderId=<guid>
   -> {"count": <total matching enquiries>, "data": [OrderEnqueryReadModel, ...]}
   - Real record paging. Bootog scopes it: a Providers.Manage user gets the provider in
     `targetProviderId`; any other user always gets their own provider.

3. Contact Log history — GET OrderEnqueryTrackingHistory?filter=orderEnqueryId=<guid>&page=1&pageSize=0

4. Provider lookup — GET Provider/GetProviders?filter=providerName=*<text>/i&page=1&pageSize=N
   [&targetProviderId=<guid>] -> {"count": <total matches>, "data": [{"id", "providerName", ...}]}
   (the "Providers - List" response). `=*` with `/i` is a case-insensitive "contains".

Filter grammar (Imatch.Domain GridifyFilterParser): `,` = AND, `(a|b)` = OR. Values are
SQL parameters on the server, but the grammar itself has no escaping, so only GUIDs
and cleaned search text are ever put into a filter. `createdDateTime` is stored in UTC.

`count` is not the same thing in every response: providers for (1), enquiries for (2)."""

import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.integrations.bootog.client import BootogApiError, BootogClient

logger = get_logger(__name__)

ORDER_ENQUIRY_PATH = "OrderEnquery"
PROVIDER_SUMMARY_PATH = "OrderEnquery/GetAllOrderEnquery"
TRACKING_HISTORY_PATH = "OrderEnqueryTrackingHistory"
PROVIDER_LIST_PATH = "Provider/GetProviders"
DEFAULT_ORDER_BY = "createdDateTime desc"

# OrderAction.ConvertedToOrder (Assignments.Shared/Enum/OrderAction.cs; the Angular
# OrderAction enum labels it "Converted To Order"). The client sets it when it sends
# PUT OrderEnquery/UpdateOrderEnquery to convert an enquiry. It is a business
# classification: whether such enquiries are wanted depends on the request.
ORDER_ACTION_CONVERTED_TO_ORDER = 6

# Ids are interpolated into the filter grammar, which has no escaping: only GUIDs pass.
_GUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
# Characters kept in a provider-name search: letters, digits, spaces and & ' . -
# Everything else (`,` `|` `(` `)` operators, `/`, LIKE wildcards % and _) is dropped.
_NAME_UNSAFE = re.compile(r"[^\w\s&'.-]|_")
_SEARCH_MAX = 100


class OrderEnquiryResponseError(BootogApiError):
    """A Bootog response did not have the expected shape."""

    def __init__(self, what: str = "order enquiry") -> None:
        super().__init__(f"The Bootog service returned an unexpected {what} response.")


@dataclass(frozen=True)
class OrderEnquiryPage:
    total: int
    records: list[Any]


@dataclass(frozen=True)
class ProviderEnquiryCount:
    # None for the group of enquiries that have no provider.
    provider_id: str | None
    provider_name: str | None
    count: int


@dataclass(frozen=True)
class ProviderSummaryPage:
    # Number of providers with at least one matching enquiry (all pages).
    total_providers: int
    providers: list[ProviderEnquiryCount]


@dataclass(frozen=True)
class ProviderRef:
    provider_id: str
    provider_name: str
    # "City, ST" from the provider record: tells apart providers with the same name.
    location: str | None = None


@dataclass(frozen=True)
class ProviderLookupPage:
    # Number of providers matching the search on the server (may exceed len(providers)).
    total: int
    providers: list[ProviderRef]


def is_guid(value: Any) -> bool:
    return isinstance(value, str) and bool(_GUID.match(value))


def clean_search_text(text: str | None) -> str:
    """Provider-name text that is safe inside a filter value."""
    cleaned = _NAME_UNSAFE.sub(" ", text or "")
    return " ".join(cleaned.split())[:_SEARCH_MAX]


def created_between_filter(start_utc: datetime, end_utc: datetime) -> str:
    """Filter clause for enquiries created in [start_utc, end_utc).

    The end bound is exclusive (`<` the next day's start), so nothing created in the last
    second of the final day is missed."""
    return f"createdDateTime>={_api_datetime(start_utc)},createdDateTime<{_api_datetime(end_utc)}"


def order_action_filter(order_action: int, include: bool) -> str:
    """Filter clause for enquiries whose orderAction is (include=True) or is not
    (include=False) `order_action`. Combine clauses with `,` (AND)."""
    return f"orderAction{'=' if include else '!='}{int(order_action)}"


def _api_datetime(value: datetime) -> str:
    # The server parses filter values with DateTimeOffset.TryParse; an explicit `Z` keeps
    # the instant unambiguous whatever timezone the server runs in.
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class OrderEnquiryApi:
    """Read-only Bootog calls for the Frontend Order Enquiry Agent. `monitor` (optional,
    the agent's RunMonitor) records every call through its `api_call(...)` method:
    operation, HTTP status, duration, attempts, page and result size — never headers."""

    def __init__(self, client: BootogClient, monitor: Any = None):
        self._client = client
        self._monitor = monitor

    async def _get(
        self,
        operation: str,
        path: str,
        params: dict[str, Any],
        count_results=None,
    ) -> Any:
        correlation_id = uuid.uuid4().hex[:12]
        stats: dict[str, Any] = {}
        started = time.perf_counter()
        error: BootogApiError | None = None
        body: Any = None
        try:
            body = await self._client.get(path, params=params, stats=stats)
            return body
        except BootogApiError as exc:
            error = exc
            raise
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            result_count = None
            if error is None and count_results is not None:
                try:
                    result_count = count_results(body)
                except Exception:
                    result_count = None
            status = error.status_code if error else stats.get("status_code")
            logger.info(
                "bootog op=%s cid=%s status=%s duration=%sms attempts=%s page=%s pageSize=%s results=%s",
                operation, correlation_id, status, duration_ms, stats.get("attempts"),
                params.get("page"), params.get("pageSize"), result_count,
            )
            if self._monitor is not None:
                self._monitor.api_call(
                    operation,
                    correlation_id=correlation_id,
                    status_code=status,
                    duration_ms=duration_ms,
                    attempts=stats.get("attempts"),
                    page=params.get("page"),
                    page_size=params.get("pageSize"),
                    result_count=result_count,
                    timed_out=bool(error and error.timed_out),
                    error=error.message if error else None,
                )

    # --- provider summary (GetAllOrderEnquery) ----------------------------------

    async def get_provider_summary(
        self,
        page: int,
        page_size: int,
        filter_expr: str | None = None,
        target_provider_id: str | None = None,
    ) -> ProviderSummaryPage:
        """One page of providers ranked by their number of matching enquiries."""
        if target_provider_id is not None and not is_guid(target_provider_id):
            raise ValueError("targetProviderId must be a GUID")
        body = await self._get(
            "get_provider_enquiry_summary",
            PROVIDER_SUMMARY_PATH,
            {
                "filter": filter_expr,
                "page": page,
                "pageSize": page_size,
                "orderBy": DEFAULT_ORDER_BY,
                "targetProviderId": target_provider_id,
            },
            count_results=lambda b: len(b.get("data") or []) if isinstance(b, dict) else None,
        )
        return parse_provider_summary(body)

    # --- enquiry records (OrderEnquery) -----------------------------------------

    async def list_enquiries(
        self,
        page: int = 1,
        page_size: int = 12,
        order_by: str = DEFAULT_ORDER_BY,
        filter_expr: str | None = None,
        target_provider_id: str | None = None,
    ) -> OrderEnquiryPage:
        if target_provider_id is not None and not is_guid(target_provider_id):
            raise ValueError("targetProviderId must be a GUID")
        body = await self._get(
            "get_provider_enquiries",
            ORDER_ENQUIRY_PATH,
            {
                "filter": filter_expr,
                "page": page,
                "pageSize": page_size,
                "orderBy": order_by,
                "targetProviderId": target_provider_id,
            },
            count_results=_data_length,
        )
        return parse_enquiry_page(body)

    async def list_all_enquiries(
        self,
        page_size: int,
        max_pages: int,
        order_by: str = DEFAULT_ORDER_BY,
        filter_expr: str | None = None,
        target_provider_id: str | None = None,
    ) -> tuple[OrderEnquiryPage, bool]:
        """Read pages until `total` is reached, a page comes back short, or `max_pages`
        is hit. Returns (all records read, truncated). Any page failure raises — a
        partial list is never returned as if it were complete."""
        records: list[Any] = []
        total = 0
        for page in range(1, max_pages + 1):
            result = await self.list_enquiries(
                page=page,
                page_size=page_size,
                order_by=order_by,
                filter_expr=filter_expr,
                target_provider_id=target_provider_id,
            )
            total = max(total, result.total)
            records.extend(result.records)
            if len(result.records) < page_size or len(records) >= total:
                return OrderEnquiryPage(total=max(total, len(records)), records=records), False
        truncated = len(records) < total
        if truncated:
            logger.warning("OrderEnquery read capped at %s pages: %s of %s records", max_pages, len(records), total)
        return OrderEnquiryPage(total=max(total, len(records)), records=records), truncated

    async def list_tracking_history(self, enquiry_id: str) -> list[dict[str, Any]]:
        """Every Contact Log / history row of one enquiry (newest first, server order)."""
        if not is_guid(enquiry_id):
            raise ValueError("enquiry id must be a GUID")
        body = await self._get(
            "get_enquiry_contact_history",
            TRACKING_HISTORY_PATH,
            {"filter": f"orderEnqueryId={enquiry_id}", "page": 1, "pageSize": 0},
            count_results=_data_length,
        )
        page = parse_enquiry_page(body)
        return [row for row in page.records if isinstance(row, dict)]

    # --- providers (Provider/GetProviders) --------------------------------------

    async def search_providers(self, name: str, limit: int) -> ProviderLookupPage:
        """Providers whose name contains `name` (case-insensitive), first `limit` rows."""
        text = clean_search_text(name)
        if not text:
            return ProviderLookupPage(total=0, providers=[])
        body = await self._get(
            "resolve_provider_by_name",
            PROVIDER_LIST_PATH,
            {"filter": f"providerName=*{text}/i", "page": 1, "pageSize": limit},
            count_results=_data_length,
        )
        return parse_provider_list(body)

    async def get_provider(self, provider_id: str) -> ProviderRef | None:
        """The provider with this id, or None when Bootog has no such provider."""
        if not is_guid(provider_id):
            raise ValueError("provider id must be a GUID")
        body = await self._get(
            "verify_provider",
            PROVIDER_LIST_PATH,
            {"targetProviderId": provider_id, "page": 1, "pageSize": 1},
            count_results=_data_length,
        )
        found = parse_provider_list(body).providers
        return next((p for p in found if p.provider_id.lower() == provider_id.lower()), None)


# --- response parsing ---------------------------------------------------------


def _data_length(body: Any) -> int | None:
    if isinstance(body, list):
        return len(body)
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        return len(body["data"])
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_enquiry_page(body: Any) -> OrderEnquiryPage:
    if isinstance(body, list):
        return OrderEnquiryPage(total=len(body), records=body)
    if not isinstance(body, dict):
        logger.warning("OrderEnquery unexpected response type: %s", type(body).__name__)
        raise OrderEnquiryResponseError()

    records = body.get("data")
    if records is None and "data" in body:
        records = []  # explicit null -> no records
    if not isinstance(records, list):
        logger.warning("OrderEnquery unexpected response keys: %s", sorted(body)[:10])
        raise OrderEnquiryResponseError()

    total = _int_or_none(body.get("count")) or 0
    return OrderEnquiryPage(total=max(total, len(records)), records=records)


def parse_provider_summary(body: Any) -> ProviderSummaryPage:
    """GetAllOrderEnquery -> provider counts. A group without a readable `totalCount` makes
    the whole response unusable: a missing count is never reported as 0."""
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        logger.warning("GetAllOrderEnquery unexpected response: %s", type(body).__name__)
        raise OrderEnquiryResponseError("provider summary")
    providers: list[ProviderEnquiryCount] = []
    for group in body["data"]:
        count = _int_or_none(group.get("totalCount")) if isinstance(group, dict) else None
        if count is None or count < 0:
            logger.warning("GetAllOrderEnquery group without a readable totalCount")
            raise OrderEnquiryResponseError("provider summary")
        provider_id = group.get("providerId")
        name = " ".join(str(group.get("providerName") or "").split()) or None
        providers.append(ProviderEnquiryCount(provider_id if is_guid(provider_id) else None, name, count))
    total = _int_or_none(body.get("count"))
    if total is None:
        logger.warning("GetAllOrderEnquery response without a readable count")
        raise OrderEnquiryResponseError("provider summary")
    # The server already sorts by count (highest first); sorting again keeps the order
    # right even if that changes, with the name as a stable tie-breaker (enquiries
    # without a provider last).
    providers.sort(key=lambda p: (-p.count, p.provider_id is None, (p.provider_name or "").casefold()))
    return ProviderSummaryPage(total_providers=max(total, len(providers)), providers=providers)


def parse_provider_list(body: Any) -> ProviderLookupPage:
    """Provider/GetProviders -> (id, name) pairs. Rows without an id or a name are skipped:
    they cannot be offered as a choice."""
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        logger.warning("GetProviders unexpected response: %s", type(body).__name__)
        raise OrderEnquiryResponseError("provider list")
    providers: list[ProviderRef] = []
    for row in body["data"]:
        if not isinstance(row, dict):
            continue
        provider_id = row.get("id")
        name = " ".join(str(row.get("providerName") or "").split())
        if is_guid(provider_id) and name:
            place = [" ".join(str(row.get(k) or "").split()) for k in ("city", "state")]
            providers.append(ProviderRef(provider_id, name, ", ".join(p for p in place if p) or None))
    total = _int_or_none(body.get("count"))
    return ProviderLookupPage(total=max(total or 0, len(providers)), providers=providers)
