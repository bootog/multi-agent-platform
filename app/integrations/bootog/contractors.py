"""Contractor service categories (Partner conversion, "Contractor" company type).

GET  /ContractorSubCategory?orderBy=name asc[&targetProviderId=]&page=&pageSize=
     -> {"count", "data": [{id, name, contractorCategoryId, contractorCategory, isActive, ...}]}
     Global rows plus the resolved provider's own rows. pageSize 0 would return all.
POST /ProviderContractorSubCategory/BulkInsert
     [{"contractorSubCategoryId": <id>, "targetProviderId": <provider>}, ...]
     Exactly what the partner screen (agency-popup.component.ts saveContractorSubCategory)
     sends after a Contractor is created: the selected service provider (or the user's own
     tenant) as targetProviderId, NOT the new contractor's id."""

from dataclasses import dataclass
from typing import Any

from app.integrations.bootog.client import BootogApiError, BootogClient


@dataclass(frozen=True)
class SubCategory:
    id: str
    name: str
    category: str | None


@dataclass(frozen=True)
class SubCategoryPage:
    items: list[SubCategory]
    total: int
    truncated: bool = False


class ContractorApi:
    def __init__(self, client: BootogClient):
        self._client = client

    async def subcategories(self, target_provider_id: str | None, page_size: int, max_pages: int) -> SubCategoryPage:
        items: dict[str, SubCategory] = {}
        total = 0
        raw = 0
        for page in range(1, max(1, max_pages) + 1):
            body = await self._client.get(
                "ContractorSubCategory",
                params={"orderBy": "name asc", "targetProviderId": target_provider_id, "page": page, "pageSize": page_size},
            )
            rows = body.get("data") if isinstance(body, dict) else None
            rows = rows if isinstance(rows, list) else []
            try:
                total = max(total, int((body or {}).get("count") or 0))
            except (TypeError, ValueError, AttributeError):
                pass
            raw += len(rows)
            for row in rows:
                if not isinstance(row, dict) or row.get("isActive") is False:
                    continue
                sub_id, name = row.get("id"), " ".join(str(row.get("name") or "").split())
                if isinstance(sub_id, str) and sub_id and name:  # blank names are not choices
                    category = " ".join(str(row.get("contractorCategory") or "").split()) or None
                    items.setdefault(sub_id, SubCategory(sub_id, name, category))
            if len(rows) < page_size or raw >= total:
                return SubCategoryPage(list(items.values()), max(total, raw))
        return SubCategoryPage(list(items.values()), max(total, raw), truncated=raw < total)

    async def save_provider_subcategories(self, subcategory_ids: list[str], target_provider_id: str) -> Any:
        body = await self._client.post(
            "ProviderContractorSubCategory/BulkInsert",
            json=[{"contractorSubCategoryId": i, "targetProviderId": target_provider_id} for i in subcategory_ids],
        )
        if isinstance(body, dict) and body.get("failed") is True:
            raise BootogApiError(f"Saving the service categories failed. {body.get('message') or ''}".strip())
        return body
