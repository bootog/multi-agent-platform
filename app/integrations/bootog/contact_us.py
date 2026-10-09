"""Contact Us requests.

ContactUS.pdf, call 5:
GET /ContactUs?filter=(status!=8,status!=11,status!=12),(domainType!=14),(moduleName=Contact-Us|moduleName=contact)
              &page=1&pageSize=12&orderBy=createdDate desc
-> {"count": 113, "data": [ContactUs record, ...]}

The company selector on the Contact Us screen narrows this same call with the
`targetProviderId` query parameter and its date range with
`(createdDate>=YYYY-MM-DD,createdDate<=YYYY-MM-DD)` appended to the filter (see
contact-us.component.ts filterString() / getData()). The same screen's status filter
REPLACES the default open-status clause with `(status=X|status=Y)`, and its assignee
filter appends `(assignedUserId=A|assignedUserId=B)`.

Searching: the screen's `ContactName` query parameter is NOT read by the ContactUs
service (it is not on TGridifyQuery), so it is silently ignored. Name/email search uses
the Gridify filter grammar instead: `FirstName=*text/i` = case-insensitive contains.
Comma = AND; `|` = OR only between conditions on the SAME field, so a name is searched
as two queries (FirstName, then LastName). The grammar has no escaping, so only values
without its operator characters are ever interpolated.

Other calls used by the agent:
GET /ContactUs/GetContactUsAssignedUser?filter=(moduleName=...)  -> [{assignedUserId, assignedUserName}]
PUT /ContactUs/bulk-update-file {"updateValues": {...}, "ids": [...]} -> {"message": ...}"""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from app.core.logging import get_logger
from app.integrations.bootog.client import BootogApiError, BootogClient

logger = get_logger(__name__)

# Default "open Contact Us requests" filter, exactly as documented: open statuses and
# domains, then the module clause (Contact-Us by default; other Contact modules such as
# Request Demo use their own clause on the same endpoint).
OPEN_STATUS_CLAUSE = "(status!=8,status!=11,status!=12)"
DOMAIN_CLAUSE = "(domainType!=14)"
OPEN_STATUS_FILTER = f"{OPEN_STATUS_CLAUSE},{DOMAIN_CLAUSE}"
CONTACT_US_MODULE_FILTER = "(moduleName=Contact-Us|moduleName=contact)"
OPEN_CONTACT_US_FILTER = f"{OPEN_STATUS_FILTER},{CONTACT_US_MODULE_FILTER}"

# ContactUsStatus.CRMCompleted (Providers.Shared/Enum/ContactUsStatus.cs and the host's
# contact-us-status.model.ts agree): the status every conversion screen sets.
CRM_COMPLETED_STATUS = 11

# Gridify operator/grouping characters plus whitespace; the grammar has no escaping.
_FILTER_UNSAFE = re.compile(r"[,|()=<>^$!*/\\\s]")
_SAFE_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


@dataclass(frozen=True)
class ContactUsPage:
    total: int
    records: list[dict[str, Any]]
    # True when a multi-page read stopped at the page cap before reaching `total`.
    truncated: bool = False


@dataclass(frozen=True)
class AssignedUser:
    user_id: str
    user_name: str


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
        created_from: str | None = None,
        created_to: str | None = None,
        module_filter: str | None = None,
        statuses: list[int] | None = None,
        all_statuses: bool = False,
        assigned_user_ids: list[str] | None = None,
        name_field_filter: tuple[str, str] | None = None,
        email_contains: str | None = None,
        request_id: str | None = None,
    ) -> ContactUsPage:
        """Status scope: `statuses` -> exactly those; `all_statuses` -> no status clause
        (archived requests included); neither -> the documented open-status clause."""
        if module_filter or statuses or all_statuses:
            if statuses:
                status_clause = _status_clause(statuses) + ","
            else:
                status_clause = "" if all_statuses else OPEN_STATUS_CLAUSE + ","
            filter_expr = f"{status_clause}{DOMAIN_CLAUSE},{module_filter or CONTACT_US_MODULE_FILTER}"
        filter_expr += _created_date_filter(created_from, created_to)
        filter_expr += _assigned_user_filter(assigned_user_ids)
        if request_id:
            if not _SAFE_ID.match(request_id):
                raise ValueError("unsupported request id")
            filter_expr += f",id={request_id}"  # the Contact Us screen's own id filter
        if name_field_filter:
            field, text = name_field_filter
            if field not in ("FirstName", "LastName") or not filter_safe(text):
                raise ValueError("unsupported name filter")
            filter_expr += f",{field}=*{text}/i"
        if email_contains:
            if not filter_safe(email_contains):
                raise ValueError("unsupported email filter")
            filter_expr += f",EmailId=*{email_contains}/i"
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
        records = [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []
        total = body.get("count") if isinstance(body, dict) else None
        try:
            total = int(total or 0)
        except (TypeError, ValueError):
            total = 0
        return ContactUsPage(total=max(total, len(records)), records=records)

    async def list_all_requests(self, page_size: int, max_pages: int, **filters: Any) -> ContactUsPage:
        """Reads pages until `total` is reached, a page comes back short, or `max_pages`
        is hit (then `truncated` is set). A failing page raises — a partial list is
        never returned as if it were complete."""
        records: list[dict[str, Any]] = []
        seen: set[Any] = set()
        total = 0
        for page in range(1, max(1, max_pages) + 1):
            result = await self.list_requests(page=page, page_size=page_size, **filters)
            total = max(total, result.total)
            for record in result.records:
                if record.get("id") not in seen:
                    seen.add(record.get("id"))
                    records.append(record)
            if len(result.records) < page_size or len(records) >= total:
                return ContactUsPage(total=max(total, len(records)), records=records)
        truncated = len(records) < total
        if truncated:
            logger.warning("ContactUs read capped at %s pages: %s of %s records", max_pages, len(records), total)
        return ContactUsPage(total=max(total, len(records)), records=records, truncated=truncated)

    async def search_requests(
        self,
        page_size: int,
        max_pages: int,
        name: str | None = None,
        email: str | None = None,
        **filters: Any,
    ) -> ContactUsPage:
        """Server-side search by name and/or email across all pages (bounded).

        Name: the first name token is searched once against FirstName and once against
        LastName (OR across fields is not supported) and the results are merged; the
        caller verifies every record against the full name. Email: case-insensitive
        contains; the caller verifies equality."""
        token = next(iter(name_tokens(name)), None)
        email_value = (email or "").strip() or None
        if email_value and not filter_safe(email_value):
            email_value = None  # not expressible in the grammar; the caller still verifies
        if token:
            queries = [{"name_field_filter": (f, token), "email_contains": email_value} for f in ("FirstName", "LastName")]
        else:
            queries = [{"email_contains": email_value}]

        records: dict[Any, dict[str, Any]] = {}
        truncated = False
        for query in queries:
            page = await self.list_all_requests(page_size, max_pages, **filters, **query)
            truncated = truncated or page.truncated
            for record in page.records:
                records.setdefault(record.get("id"), record)
        ordered = sorted(records.values(), key=lambda r: str(r.get("createdDate") or ""), reverse=True)
        return ContactUsPage(total=len(ordered), records=ordered, truncated=truncated)

    async def assigned_users(self, module_filter: str | None = None) -> list[AssignedUser]:
        # The service de-duplicates assignees AFTER paging, so read every row (pageSize=0 = all).
        body = await self._client.get(
            "ContactUs/GetContactUsAssignedUser",
            params={"filter": module_filter or CONTACT_US_MODULE_FILTER, "page": 1, "pageSize": 0},
        )
        rows = body.get("data") if isinstance(body, dict) else body
        users: dict[str, AssignedUser] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            user_id, name = row.get("assignedUserId"), (row.get("assignedUserName") or "").strip()
            if isinstance(user_id, str) and user_id and name and user_id not in users:
                users[user_id] = AssignedUser(user_id, name)
        return sorted(users.values(), key=lambda u: u.user_name.casefold())

    async def update_requests(self, ids: list[str], values: dict[str, Any]) -> dict[str, Any]:
        """PUT bulk-update-file. Keys are ContactUsInfo column names (case-sensitive).
        Bootog answers 200 {"message": "Update completed successfully."}; it has no
        tenant scoping and writes no tracking history."""
        if not ids or not all(isinstance(i, str) and _SAFE_ID.match(i) for i in ids):
            raise ValueError("valid Contact Us ids are required")
        body = await self._client.put("ContactUs/bulk-update-file", json={"updateValues": values, "ids": ids})
        if isinstance(body, dict) and (body.get("failed") is True or body.get("error")):
            raise BootogApiError(f"The request update failed. {body.get('message') or ''}".strip())
        return body if isinstance(body, dict) else {"message": str(body or "")}


def filter_safe(value: str) -> bool:
    return bool(value) and len(value) <= 100 and not _FILTER_UNSAFE.search(value)


def name_tokens(name: str | None) -> list[str]:
    """Alphanumeric name parts, lower-cased (always filter-safe)."""
    return [t for t in re.findall(r"[^\W_]+", (name or "").casefold()) if len(t) > 1]


def _created_date_filter(created_from: str | None, created_to: str | None) -> str:
    """Date-range clause in the host app's syntax. Dates must be YYYY-MM-DD; both are
    inclusive days. The service converts a bare date to midnight, so `createdDate<=D`
    would drop everything created during day D: the end is sent as `< D + 1 day`."""
    parts = []
    if created_from:
        parts.append(f"createdDate>={date.fromisoformat(created_from).isoformat()}")
    if created_to:
        parts.append(f"createdDate<{(date.fromisoformat(created_to) + timedelta(days=1)).isoformat()}")
    return f",(({','.join(parts)}))" if parts else ""


def _status_clause(statuses: list[int]) -> str:
    return "(" + "|".join(f"status={int(s)}" for s in statuses) + ")"


def _assigned_user_filter(user_ids: list[str] | None) -> str:
    ids = [u for u in user_ids or [] if _SAFE_ID.match(u)]
    return f",(assignedUserId={'|assignedUserId='.join(ids)})" if ids else ""
