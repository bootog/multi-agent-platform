"""Assign / Schedule / Add Log APIs, as the Orders page's "Assign/Schedule" and
"Add Log" actions call them.

Assign / Schedule (libs/component/assignment-scheduling):
    GET  UserDetail/GetOrderUsers?...                  producer candidates, grouped
         (nearByInspectors, coverageAreaInspectors, otherInspectors, matchedInspectors,
          scheduledInspectors); query as assignment-scheduling.service getProducerFilter()
    GET  UserWorkingHour/GetUserWorkingHourList?page=1&targetUserId=<id>   working hours
    GET  UserBlockDay?targetUserId=<id>                                     blocked days
    GET  Order/GetOrderServiceDetails?startDate&toDate&from=Personal-Info&targetUserId
         the producer's booked appointments in a date range
    GET  Review/GetAverageRating?reviewForUserId=<id>
    GET  Order/GetProducerServiceCount?targetUserId=<id>  {assigned, scheduled, submitted, rejected}
    POST Order/AssignOrSchedule                       assign (no date) or schedule (date + times)

Add Log (libs/popup/contact-log-popup):
    GET  ClientContactLog?filter=orderId=<id>&orderBy=createdDateTime desc
    POST ClientContactLog                              one contact log entry
    GET  Template/GetTemplates?filter=templateType=<n>

Writes are sent once (never retried by the client) and checked against Bootog's Result
envelope: an HTTP 200 with `failed: true` is a failure."""

import json
from typing import Any

from app.core.logging import get_logger
from app.integrations.bootog.client import BootogApiError, BootogClient, check_result
from app.integrations.bootog.orders import OrderPage, _list_of_dicts, _page

logger = get_logger(__name__)

PRODUCER_GROUPS: dict[str, str] = {
    "matchedInspectors": "Matched",
    "scheduledInspectors": "Scheduled nearby",
    "nearByInspectors": "Close by",
    "coverageAreaInspectors": "Covering county",
    "otherInspectors": "Other",
}
CALENDAR_FROM = "Personal-Info"  # the value the scheduling calendar sends


class SchedulingResponseError(BootogApiError):
    def __init__(self, what: str) -> None:
        super().__init__(f"The Bootog service returned an unexpected {what} response.")


class SchedulingApi:
    def __init__(self, client: BootogClient):
        self._client = client

    async def order_users(
        self,
        *,
        latitude: float | None,
        longitude: float | None,
        state_code: str | None,
        state_name: str | None,
        county: str | None,
        target_provider_id: str | None,
        max_distance_miles: int,
        program_id: str | None = None,
        skillset_ids: list[str] | None = None,
        credential_ids: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Producer groups for an order. `skillset_ids` / `credential_ids` are the matching
        filters the Assign/Schedule screen pre-fills from the service type (or program);
        BOOTOG's "matchedInspectors" are the provider's producers that pass them."""
        params: dict[str, Any] = {
            "nearByInspectors": "true",
            "coverageAreaUsers": "true",
            "otherUsers": "true",
            "nearByScheduledUsers": "true",
            "showMatchedUsers": "true",
            "latitude": latitude,
            "longitude": longitude,
            "fromDistance": 0,
            "toDistance": max_distance_miles,
            "targetProviderId": target_provider_id,
            "programId": program_id,
            "skillsetIds": skillset_ids or None,
            "credentialIds": credential_ids or None,
        }
        if state_code and county:
            # The county as the Assign/Schedule screen sends it (1.assignschedule.txt, api 6).
            params["counties"] = json.dumps(
                {"text": county, "value": county, "state": state_name or state_code, "stateCode": state_code, "selected": False},
                separators=(",", ":"),
            )
        body = await self._client.get("UserDetail/GetOrderUsers", params=params)
        if not isinstance(body, dict):
            raise SchedulingResponseError("producer search")
        groups: dict[str, list[dict[str, Any]]] = {}
        for key in PRODUCER_GROUPS:
            items = body.get(key)
            if items is None:
                continue
            if not isinstance(items, list):
                raise SchedulingResponseError("producer search")
            groups[key] = [u for u in items if isinstance(u, dict) and u.get("id")]
        return groups

    async def working_hours(self, user_id: str) -> list[dict[str, Any]]:
        body = await self._client.get("UserWorkingHour/GetUserWorkingHourList", params={"page": 1, "targetUserId": user_id})
        return _page(body, "working hours").records

    async def block_days(self, user_id: str) -> list[dict[str, Any]]:
        return _page(await self._client.get("UserBlockDay", params={"targetUserId": user_id}), "blocked days").records

    async def booked_appointments(self, user_id: str, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
        body = await self._client.get(
            "Order/GetOrderServiceDetails",
            params={"startDate": start_iso, "toDate": end_iso, "from": CALENDAR_FROM, "targetUserId": user_id},
        )
        if body is None:
            return []
        return _list_of_dicts(body, "producer calendar")

    async def producer_service_count(self, user_id: str) -> dict[str, Any]:
        """{userId, assigned, scheduled, submitted, rejected} — the producer's current workload."""
        body = await self._client.get("Order/GetProducerServiceCount", params={"targetUserId": user_id})
        if not isinstance(body, dict):
            raise SchedulingResponseError("producer workload")
        return body

    async def average_rating(self, user_id: str) -> dict[str, Any]:
        body = await self._client.get("Review/GetAverageRating", params={"reviewForUserId": user_id})
        if not isinstance(body, dict):
            raise SchedulingResponseError("rating")
        return body

    async def assign_or_schedule(self, payload: dict[str, Any]) -> dict[str, Any]:
        return check_result(await self._client.post("Order/AssignOrSchedule", json=payload), "Assign/Schedule")

    async def contact_logs(self, order_id: str, page: int = 1, page_size: int = 20) -> OrderPage:
        body = await self._client.get(
            "ClientContactLog",
            params={"filter": f"orderId={order_id}", "orderBy": "createdDateTime desc", "page": page, "pageSize": page_size},
        )
        return _page(body, "contact log")

    async def create_contact_log(self, payload: dict[str, Any]) -> dict[str, Any]:
        return check_result(await self._client.post("ClientContactLog", json=payload), "Add Log")

    async def templates(self, template_type: int) -> list[dict[str, Any]]:
        body = await self._client.get("Template/GetTemplates", params={"filter": f"templateType={template_type}"})
        return _page(body, "template").records
