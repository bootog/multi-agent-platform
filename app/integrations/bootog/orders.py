"""Orders module (Assignment > Orders) read APIs, as the host's Orders page calls them.

    GET Order/GetOrderHeaders?assignmentCategory=1          status cards (name, count, statusId, reviewStatus)
    GET Order/GetOrderDetails?page&pageSize&orderBy&...     order grid ({count, data})
    GET Order/GetOrderDetailById?OrderId=<id>               one order
    GET OrderStatus                                         order / service statuses ({count, data})
    GET OrderSubStatus?filter=isActive=true&type=2          service sub-statuses ({count, data})
    GET ServiceType/GetServiceTypeNames                     [{value, text, categoryName, ...}]
    GET ServiceType?filter=(id=<id>)                        one service type with its requirements ({count, data})
    GET Program/ProgramDetailById?programId=<id>            a program incl. programSkillSets / programDocumentTypes
    GET SkillSet/GetSkillSetNames?targetProviderId=<id>     valid skill set ids
    GET DocumentType/GetDocumentTypeNames?targetProviderId=<id>  valid credential (document type) ids
    GET PostalCode/GetStatesByCountryCode?countryCode=<cc>  [{stateCode, stateName}]
    GET Program/GetProgramNames                             [{value, text, aliasName, ...}]
    GET UserDetail/GetAllTomeZones                          ["Eastern Standard Time", ...] (Bootog's spelling)
    GET Provider/GetProviderNames?providerType=1            [{value, text, ...}]
    GET Order/GetTPQAProviderNames                          [{value, text, ...}]

Grid query parameters follow libs/services/order/order-utilles.service.ts buildQuery():
list values are sent as repeated keys (Angular HttpParams.append), exactly like the host.
Response shapes are validated; anything unexpected is reported, never read as "empty"."""

from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.integrations.bootog.client import BootogApiError, BootogClient

logger = get_logger(__name__)

# AssignmentCategory (libs/models/enums/assignment-category.model.ts): the Orders page.
ASSIGNMENT_CATEGORY_ORDER = 1
DEFAULT_ORDER_BY = "createdDateTime desc"

# Query parameters GET Order/GetOrderDetails accepts (names as buildQuery() sends them).
ORDER_QUERY_PARAMS = frozenset({
    "orderServiceStatusIds", "orderStatusIds", "orderSubStatusIds", "orderReviewStatuses",
    "orderTpQaReviewStatuses", "notInOrderStatusName", "mainServiceTypeIds", "programIds",
    "propertyStates", "counties", "isRushOrder", "isBatchedOrder", "ScheduleFromDate", "ScheduleToDate",
    "statusFromDate", "statusToDate", "statusId", "allField", "policyNumber", "jobNumber",
    "customerName", "customerPhone", "customerEmail", "propertyAddress1", "propertyCity",
    "propertyZipCode", "targetUserIds", "targetProviderId",
})


class OrdersResponseError(BootogApiError):
    def __init__(self, what: str) -> None:
        super().__init__(f"The Bootog service returned an unexpected {what} response.")


@dataclass(frozen=True)
class OrderPage:
    total: int
    records: list[dict[str, Any]]


class OrdersApi:
    def __init__(self, client: BootogClient):
        self._client = client

    async def order_headers(self, assignment_category: int = ASSIGNMENT_CATEGORY_ORDER) -> list[dict[str, Any]]:
        body = await self._client.get("Order/GetOrderHeaders", params={"assignmentCategory": assignment_category})
        return _list_of_dicts(body, "order status")

    async def list_orders(
        self,
        page: int = 1,
        page_size: int = 10,
        order_by: str = DEFAULT_ORDER_BY,
        filters: dict[str, Any] | None = None,
        assignment_category: int = ASSIGNMENT_CATEGORY_ORDER,
    ) -> OrderPage:
        params: dict[str, Any] = {
            "page": page, "pageSize": page_size, "assignmentCategory": assignment_category, "orderBy": order_by,
        }
        for key, value in (filters or {}).items():
            if key not in ORDER_QUERY_PARAMS:
                raise ValueError(f"unsupported order filter: {key}")
            if value in (None, "", []):
                continue
            params[key] = _param(value)
        body = await self._client.get("Order/GetOrderDetails", params=params)
        return _page(body, "order list")

    async def get_order(self, order_id: str) -> dict[str, Any]:
        body = await self._client.get("Order/GetOrderDetailById", params={"OrderId": order_id})
        if not isinstance(body, dict) or not body.get("id"):
            raise OrdersResponseError("order")
        return body

    async def service_type(self, service_type_id: str) -> dict[str, Any] | None:
        """The service type's setup incl. required skill sets and credential types."""
        records = _page(await self._client.get("ServiceType", params={"filter": f"(id={service_type_id})"}), "service type").records
        return next((r for r in records if str(r.get("id") or service_type_id) == service_type_id), records[0] if records else None)

    async def program_detail(self, program_id: str) -> dict[str, Any] | None:
        """GET Program/ProgramDetailById: programSkillSets, programDocumentTypes, ..."""
        body = await self._client.get("Program/ProgramDetailById", params={"programId": program_id})
        if body is None:
            return None
        if not isinstance(body, dict):
            raise OrdersResponseError("program")
        return body

    async def skill_set_names(self, target_provider_id: str | None) -> list[dict[str, Any]]:
        """GET SkillSet/GetSkillSetNames: [{value, text, ...}] — the valid skill set ids."""
        body = await self._client.get("SkillSet/GetSkillSetNames", params={"targetProviderId": target_provider_id})
        return _list_of_dicts(body, "skill set")

    async def document_type_names(self, target_provider_id: str | None) -> list[dict[str, Any]]:
        """GET DocumentType/GetDocumentTypeNames: [{value, text, categoryName, ...}] — valid credential ids."""
        body = await self._client.get("DocumentType/GetDocumentTypeNames", params={"targetProviderId": target_provider_id})
        return _list_of_dicts(body, "document type")

    async def states(self, country_code: str) -> list[dict[str, Any]]:
        """GET PostalCode/GetStatesByCountryCode: [{stateCode, stateName, ...}]."""
        body = await self._client.get("PostalCode/GetStatesByCountryCode", params={"countryCode": country_code})
        return _list_of_dicts(body, "state")

    async def order_statuses(self) -> list[dict[str, Any]]:
        return _page(await self._client.get("OrderStatus"), "order status").records

    async def order_sub_statuses(self, status_type: int = 2) -> list[dict[str, Any]]:
        body = await self._client.get("OrderSubStatus", params={"filter": "isActive=true", "type": status_type})
        return _page(body, "order sub-status").records

    async def service_types(self) -> list[dict[str, Any]]:
        return _list_of_dicts(await self._client.get("ServiceType/GetServiceTypeNames"), "service type")

    async def programs(self) -> list[dict[str, Any]]:
        return _list_of_dicts(await self._client.get("Program/GetProgramNames"), "program")

    async def time_zones(self) -> list[str]:
        body = await self._client.get("UserDetail/GetAllTomeZones")
        if not isinstance(body, list):
            raise OrdersResponseError("time zone")
        return [str(z) for z in body if isinstance(z, str) and z.strip()]

    async def providers(self, provider_type: int = 1) -> list[dict[str, Any]]:
        body = await self._client.get("Provider/GetProviderNames", params={"providerType": provider_type})
        return _list_of_dicts(body, "provider")

    async def tpqa_providers(self) -> list[dict[str, Any]]:
        return _list_of_dicts(await self._client.get("Order/GetTPQAProviderNames"), "TPQA provider")


def _param(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return [_param(v) for v in value]
    return value


def _list_of_dicts(body: Any, what: str) -> list[dict[str, Any]]:
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        body = body["data"]
    if not isinstance(body, list):
        logger.warning("bootog %s: unexpected response type %s", what, type(body).__name__)
        raise OrdersResponseError(what)
    return [item for item in body if isinstance(item, dict)]


def _page(body: Any, what: str) -> OrderPage:
    if isinstance(body, list):
        records = [r for r in body if isinstance(r, dict)]
        return OrderPage(total=len(records), records=records)
    if not isinstance(body, dict) or not isinstance(body.get("data", []), list):
        logger.warning("bootog %s: unexpected response", what)
        raise OrdersResponseError(what)
    records = [r for r in body.get("data") or [] if isinstance(r, dict)]
    try:
        total = int(body.get("count") or 0)
    except (TypeError, ValueError):
        total = 0
    return OrderPage(total=max(total, len(records)), records=records)
