"""Contact Us requests.

ContactUS.pdf, call 5:
GET /ContactUs?filter=(status!=8,status!=11,status!=12),(domainType!=14),(moduleName=Contact-Us|moduleName=contact)
              &page=1&pageSize=12&orderBy=createdDate desc
-> {"count": 113, "data": [ContactUs record, ...]}

The company selector on the Contact Us screen narrows this same call with the
`targetProviderId` query parameter (see contact-us.component.ts getData())."""

from dataclasses import dataclass
from typing import Any

from app.integrations.bootog.client import BootogClient

# Default "open Contact Us requests" filter, exactly as documented.
OPEN_CONTACT_US_FILTER = "(status!=8,status!=11,status!=12),(domainType!=14),(moduleName=Contact-Us|moduleName=contact)"


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
    ) -> ContactUsPage:
        body = await self._client.get(
            "ContactUs",
            params={
                "filter": filter_expr,
                "page": page,
                "pageSize": page_size,
                "orderBy": order_by,
                "targetProviderId": provider_id,
            },
        )
        records = body.get("data") if isinstance(body, dict) else None
        records = records if isinstance(records, list) else []
        total = body.get("count") if isinstance(body, dict) else None
        return ContactUsPage(total=max(int(total or 0), len(records)), records=records)
