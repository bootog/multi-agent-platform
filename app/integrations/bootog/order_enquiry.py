"""Frontend Order Enquiries.

GET /OrderEnquery?page=1&pageSize=12&orderBy=createdDateTime desc
-> enquiry records (the endpoint name keeps Bootog's own "Enquery" spelling).

The envelope is expected to match the other Bootog list endpoints
({"count": n, "data": [...]}); a bare JSON list is also accepted. Anything else is
reported as an unexpected response instead of being treated as "no enquiries".
Individual record fields are not assumed to be present — callers validate them."""

from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.integrations.bootog.client import BootogApiError, BootogClient

logger = get_logger(__name__)

ORDER_ENQUIRY_PATH = "OrderEnquery"
DEFAULT_ORDER_BY = "createdDateTime desc"


class OrderEnquiryResponseError(BootogApiError):
    """The OrderEnquery response did not have the expected shape."""

    def __init__(self) -> None:
        super().__init__("The Bootog service returned an unexpected order enquiry response.")


@dataclass(frozen=True)
class OrderEnquiryPage:
    total: int
    records: list[Any]


class OrderEnquiryApi:
    def __init__(self, client: BootogClient):
        self._client = client

    async def list_enquiries(
        self,
        page: int = 1,
        page_size: int = 12,
        order_by: str = DEFAULT_ORDER_BY,
    ) -> OrderEnquiryPage:
        body = await self._client.get(
            ORDER_ENQUIRY_PATH,
            params={"page": page, "pageSize": page_size, "orderBy": order_by},
        )
        return parse_enquiry_page(body)

    async def list_all_enquiries(
        self,
        page_size: int,
        max_pages: int,
        order_by: str = DEFAULT_ORDER_BY,
    ) -> tuple[OrderEnquiryPage, bool]:
        """Read pages until `total` is reached, a page comes back short, or `max_pages`
        is hit. Returns (all records read, truncated). Any page failure raises — a
        partial list is never returned as if it were complete."""
        records: list[Any] = []
        total = 0
        for page in range(1, max_pages + 1):
            result = await self.list_enquiries(page=page, page_size=page_size, order_by=order_by)
            total = max(total, result.total)
            records.extend(result.records)
            if len(result.records) < page_size or len(records) >= total:
                return OrderEnquiryPage(total=max(total, len(records)), records=records), False
        truncated = len(records) < total
        if truncated:
            logger.warning("OrderEnquery read capped at %s pages: %s of %s records", max_pages, len(records), total)
        return OrderEnquiryPage(total=max(total, len(records)), records=records), truncated


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

    try:
        total = int(body.get("count") or 0)
    except (TypeError, ValueError):
        total = 0
    return OrderEnquiryPage(total=max(total, len(records)), records=records)
