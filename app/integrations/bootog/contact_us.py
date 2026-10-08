"""Contact Us requests.

ContactUS.pdf, call 5:
GET /ContactUs?filter=(status!=8,status!=11,status!=12),(domainType!=14),(moduleName=Contact-Us|moduleName=contact)
              &page=1&pageSize=12&orderBy=createdDate desc
-> {"count": 113, "data": [ContactUs record, ...]}

The company selector on the Contact Us screen narrows this same call with the
`targetProviderId` query parameter, its search box with `ContactName`, and its date
range with `(createdDate>=YYYY-MM-DD,createdDate<=YYYY-MM-DD)` appended to the filter
(see contact-us.component.ts filterString() / getData())."""

from dataclasses import dataclass
from typing import Any

from app.integrations.bootog.client import BootogClient

# Default "open Contact Us requests" filter, exactly as documented: open statuses and
# domains, then the module clause (Contact-Us by default; other Contact modules such as
# Request Demo use their own clause on the same endpoint).
OPEN_STATUS_FILTER = "(status!=8,status!=11,status!=12),(domainType!=14)"
CONTACT_US_MODULE_FILTER = "(moduleName=Contact-Us|moduleName=contact)"
OPEN_CONTACT_US_FILTER = f"{OPEN_STATUS_FILTER},{CONTACT_US_MODULE_FILTER}"


@dataclass(frozen=True)
class ContactUsPage:
    total: int
    records: list[dict[str, Any]]


class ContactUsApi:
    def __init__(self, client: BootogClient):
        self._client = client

    async def list_requests(
        self,
        provider_id: str | None = None,
        page: int = 1,
        page_size: int = 12,
        order_by: str = "createdDate desc",
        filter_expr: str = OPEN_CONTACT_US_FILTER,
        contact_name: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        module_filter: str | None = None,
    ) -> ContactUsPage:
        if module_filter:
            filter_expr = f"{OPEN_STATUS_FILTER},{module_filter}"
        filter_expr += _created_date_filter(created_from, created_to)
        body = await self._client.get(
            "ContactUs",
            params={
                "filter": filter_expr,
                "page": page,
                "pageSize": page_size,
                "orderBy": order_by,
                "targetProviderId": provider_id,
                "ContactName": contact_name,
            },
        )
        records = body.get("data") if isinstance(body, dict) else None
        records = records if isinstance(records, list) else []
        total = body.get("count") if isinstance(body, dict) else None
        return ContactUsPage(total=max(int(total or 0), len(records)), records=records)


def _created_date_filter(created_from: str | None, created_to: str | None) -> str:
    """Date-range clause in the host app's syntax. Dates must be YYYY-MM-DD."""
    parts = []
    if created_from:
        parts.append(f"createdDate>={created_from}")
    if created_to:
        parts.append(f"createdDate<={created_to}")
    return f",(({','.join(parts)}))" if parts else ""
