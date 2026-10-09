"""Provider (company) lookups and provider-vendor links.

ContactUS.pdf, call 1: GET /ProviderUserMapping/GetProviderUserMappingBySession
-> {"count": n, "data": [{"providerId": ..., "providerName": ..., ...}]}
This is the list the Contact Us screen uses for its company selector
(ProviderUserService.getProvidersBySession in the Angular libs).

GET /Provider/SearchProviders?providerType=1&providerName=<prefix>&page=1&pageSize=10
-> [ProviderSearchModel, ...]  (a plain array, no count). The service matches
`ProviderName ILIKE '<providerName>%'` (prefix only) and has no location parameter;
a short page means the end of the list. This is the "Select Service Provider" list
of the B2B / Partner screens (app-provider-select).

POST /Provider/SaveProviderVendor {"vendorId", "targetProviderId"} -> Result. Inserts
the (provider, vendor) link only when it doesn't exist yet, so repeating it is safe.
`vendorId` is the vendor's PROVIDER id (AddVendorUser's `value`), not a user id.
`targetProviderId` must always be sent: when omitted, managers get Guid.Empty."""

from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.integrations.bootog.client import BootogClient, check_result

logger = get_logger(__name__)

SERVICE_PROVIDER_TYPE = 1  # ProviderType.ServiceProvider


@dataclass(frozen=True)
class Provider:
    provider_id: str
    provider_name: str


@dataclass(frozen=True)
class ProviderSearchResult:
    provider_id: str
    provider_name: str
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    website: str | None = None


@dataclass(frozen=True)
class ProviderSearchPage:
    providers: list[ProviderSearchResult]
    # True when the read stopped at the page cap with more pages possibly left.
    truncated: bool = False


class ProviderApi:
    def __init__(self, client: BootogClient):
        self._client = client

    async def list_for_session(self) -> list[Provider]:
        body = await self._client.get("ProviderUserMapping/GetProviderUserMappingBySession")
        rows = body.get("data", []) if isinstance(body, dict) else body or []
        providers: dict[str, Provider] = {}
        for row in rows:
            provider_id = row.get("providerId")
            if provider_id and provider_id not in providers:
                providers[provider_id] = Provider(provider_id, (row.get("providerName") or "").strip())
        return sorted(providers.values(), key=lambda p: p.provider_name.lower())

    async def search_providers(
        self,
        provider_name: str | None = None,
        provider_type: int = SERVICE_PROVIDER_TYPE,
        page: int = 1,
        page_size: int = 10,
    ) -> list[ProviderSearchResult]:
        providers, _ = await self._search_page(provider_name, provider_type, page, page_size)
        return providers

    async def search_all_providers(
        self,
        provider_name: str | None,
        page_size: int,
        max_pages: int,
        provider_type: int = SERVICE_PROVIDER_TYPE,
    ) -> ProviderSearchPage:
        """Every page of SearchProviders (bounded). The endpoint returns no count, so
        the read ends on a short page; hitting `max_pages` is reported as truncated."""
        found: dict[str, ProviderSearchResult] = {}
        for page in range(1, max(1, max_pages) + 1):
            providers, raw_rows = await self._search_page(provider_name, provider_type, page, page_size)
            for p in providers:
                found.setdefault(p.provider_id, p)
            if raw_rows < page_size:
                return ProviderSearchPage(list(found.values()))
        logger.warning("SearchProviders read capped at %s pages", max_pages)
        return ProviderSearchPage(list(found.values()), truncated=True)

    async def _search_page(
        self, provider_name: str | None, provider_type: int, page: int, page_size: int
    ) -> tuple[list[ProviderSearchResult], int]:
        """One page: (usable providers, number of raw rows Bootog returned)."""
        body = await self._client.get(
            "Provider/SearchProviders",
            params={"providerType": provider_type, "providerName": provider_name, "page": page, "pageSize": page_size},
        )
        rows = body.get("data") if isinstance(body, dict) else body
        rows = rows if isinstance(rows, list) else []
        out: list[ProviderSearchResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            provider_id = row.get("providerId")
            name = " ".join(str(row.get("providerName") or "").split())
            if not isinstance(provider_id, str) or not provider_id or not name:
                continue  # a blank name can't be offered as a choice
            out.append(
                ProviderSearchResult(
                    provider_id=provider_id,
                    provider_name=name,
                    address=_text(row.get("providerAddress")),
                    phone=_text(row.get("providerPhone")),
                    email=_text(row.get("providerEmailId")),
                    website=_text(row.get("providerWebsite")),
                )
            )
        return out, len(rows)

    async def save_provider_vendor(self, vendor_id: str, target_provider_id: str) -> dict[str, Any]:
        if not vendor_id or not target_provider_id:
            raise ValueError("vendorId and targetProviderId are required")
        body = await self._client.post(
            "Provider/SaveProviderVendor", json={"vendorId": vendor_id, "targetProviderId": target_provider_id}
        )
        return check_result(body, "Linking the vendor to the provider")


def _text(value: Any) -> str | None:
    text = " ".join(str(value).split()) if value is not None else ""
    return text.strip(" ,") or None
