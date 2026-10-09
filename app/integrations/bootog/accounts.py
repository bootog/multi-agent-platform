"""User accounts and roles used when converting a Contact Us request.

GET  /UserDetail/GetUserRolesByEmailId?emailId=   -> [{userId, userName, userEmail, providers:[{providerId,
     providerName, roles:[{roleId, roleName}]}]}]. Exact, case-sensitive email match; may be 204/null.
GET  /Roles/GetRoles?roleNames=X&isdefault=true&targetProviderId=0000…  -> {"count", "data": [{id, name}]}.
     The empty provider id makes the service filter by name only (any other value ORs in
     that provider's roles).
POST /UserDetail/AddVendorUser  [FromForm] UserProviderModel -> Result {value, failed, message}.
     `value` is the PROVIDER id the account was attached to (new company for
     IsSkipTenant=true without a TargetProviderId), not the user id. The server sends the
     password in the "account created" email (Auth0 forces a change on first login). It is
     NOT idempotent and has no rollback: a repeat with the same role fails with
     "User Email Id Already Exists!"."""

import re
from dataclasses import dataclass, field
from typing import Any

from app.integrations.bootog.client import BootogClient, BootogResultError, check_result

EMPTY_GUID = "00000000-0000-0000-0000-000000000000"
_GUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


@dataclass(frozen=True)
class AccountRole:
    provider_id: str | None
    provider_name: str | None
    role_name: str


@dataclass(frozen=True)
class ExistingAccount:
    user_id: str | None
    user_name: str | None
    roles: list[AccountRole] = field(default_factory=list)

    def has_role(self, role_name: str) -> bool:
        return any(r.role_name.casefold() == role_name.casefold() for r in self.roles)


class RoleNotFoundError(BootogResultError):
    pass


class AccountApi:
    def __init__(self, client: BootogClient):
        self._client = client

    async def accounts_by_email(self, email: str) -> list[ExistingAccount]:
        body = await self._client.get("UserDetail/GetUserRolesByEmailId", params={"emailId": email})
        rows = body.get("data") if isinstance(body, dict) else body
        accounts: list[ExistingAccount] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            roles = [
                AccountRole(p.get("providerId"), p.get("providerName"), str(r.get("roleName")))
                for p in row.get("providers") or []
                if isinstance(p, dict)
                for r in p.get("roles") or []
                if isinstance(r, dict) and r.get("roleName")
            ]
            accounts.append(ExistingAccount(row.get("userId"), row.get("userName"), roles))
        return accounts

    async def role_id(self, role_name: str) -> str:
        body = await self._client.get(
            "Roles/GetRoles", params={"roleNames": role_name, "isdefault": "true", "targetProviderId": EMPTY_GUID}
        )
        rows = body.get("data") if isinstance(body, dict) else None
        rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        match = next((r for r in rows if r.get("name") == role_name), None) or next(
            (r for r in rows if str(r.get("name") or "").casefold() == role_name.casefold()), None
        )
        role_id = (match or {}).get("id")
        if not isinstance(role_id, str) or not _GUID.match(role_id) or role_id == EMPTY_GUID:
            raise RoleNotFoundError(f"The '{role_name}' role could not be found.")
        return role_id

    async def add_vendor_user(self, fields: dict[str, Any]) -> str | None:
        """Creates the account; returns the provider id from `value` (None when Bootog
        reports success without a usable id). Raises on HTTP or application failure.
        `fields` holds the password: it is sent once and never logged or returned."""
        body = await self._client.post_form("UserDetail/AddVendorUser", fields)
        result = check_result(body, "Creating the account")
        value = result.get("value")
        return value if isinstance(value, str) and _GUID.match(value) and value != EMPTY_GUID else None


def is_guid(value: Any) -> bool:
    return isinstance(value, str) and bool(_GUID.match(value)) and value != EMPTY_GUID
