"""User accounts and roles used when converting a Contact Us request.

GET  /UserDetail/GetUserRolesByEmailId?emailId=   -> [{userId, userName, userEmail, providers:[{providerId,
     providerName, roles:[{roleId, roleName}]}]}]. Exact, case-sensitive email match. [] = no
     account; null (204) = the service's query threw — never read as "no account".
GET  /Roles/GetRoles?roleNames=X&isdefault=true&targetProviderId=0000…  -> {"count", "data": [{id, name}]}.
     The empty provider id makes the service filter by name only (any other value ORs in
     that provider's roles).
POST /UserDetail/AddVendorUser  [FromForm] UserProviderModel -> Result {value, failed, message}.
     `value` is the PROVIDER id the account was attached to (new company for
     IsSkipTenant=true without a TargetProviderId), not the user id. The server sends the
     password in the "account created" email (Auth0 forces a change on first login). It is
     NOT idempotent and has no rollback: a repeat with the same role fails with
     "User Email Id Already Exists!" — matched on UserRoles (UserId, RoleId) WITHOUT
     IsDelete, so a removed role blocks it too. Every failure after the UserRoles insert
     leaves the login and role behind (RemoveUserAccountAsync only acts on a null model,
     i.e. never), whatever message comes back. Its catch-all returns any .NET exception as
     `failed: true` + the exception message (e.g. "Object reference not set to an instance of
     an object."), AFTER the Auth0 login and UserRoles row were written — a partial account."""

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.integrations.bootog.client import BootogApiError, BootogClient, BootogResultError, check_result

logger = get_logger(__name__)

EMPTY_GUID = "00000000-0000-0000-0000-000000000000"
_GUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


@dataclass(frozen=True)
class AccountRole:
    provider_id: str | None
    provider_name: str | None
    role_name: str
    # UserRoles."IsDelete": a removed role is history, not an active role.
    deleted: bool = False
    # UserRoles."RoleId": AddVendorUser's duplicate check matches on it (not on the name,
    # which other providers' copies of a role share).
    role_id: str | None = None

    @property
    def active(self) -> bool:
        return not self.deleted


@dataclass(frozen=True)
class ExistingAccount:
    user_id: str | None
    user_name: str | None
    roles: list[AccountRole] = field(default_factory=list)

    def has_role(self, role_name: str) -> bool:
        """An ACTIVE role with that name (deleted roles don't count)."""
        return any(r.active and r.role_name.casefold() == role_name.casefold() for r in self.roles)


class RoleNotFoundError(BootogResultError):
    pass


class AccountLookupError(BootogApiError):
    """The existing-account check gave no usable answer: duplicates can't be ruled out."""


class AccountCreationIncomplete(BootogResultError):
    """AddVendorUser crashed part-way (its catch-all turned a .NET exception into
    `failed: true`). The Auth0 login and the UserRoles row are written BEFORE the steps
    that can throw, and nothing rolls them back — so part of the account may exist."""

    @property
    def outcome_unknown(self) -> bool:
        return True


# .NET exception texts that AddVendorUserAsync's `catch (Exception ex) => Result.Fail(ex.Message)`
# returns. They describe a server fault, not a problem with the submitted details.
_SERVER_FAULT = re.compile(
    r"Object reference not set to an instance of an object|Nullable object must have a value|"
    r"Sequence contains no (matching )?elements?|Index was outside the bounds|Value cannot be null|"
    r"Input string was not in a correct format|Response status code does not indicate success|"
    r"\bat [A-Z][\w.]+\.[\w<>]+\(",  # a stack-trace frame
    re.IGNORECASE,
)
# AddVendorUserAsync's answer when the email already has an account with the role.
_EMAIL_EXISTS = re.compile(r"\bemail\b.*\balready exists\b", re.IGNORECASE)


class AccountApi:
    def __init__(self, client: BootogClient):
        self._client = client

    async def accounts_by_email(self, email: str) -> list[ExistingAccount]:
        body = await self._client.get("UserDetail/GetUserRolesByEmailId", params={"emailId": email})
        # The service answers [] when the email has no account, and null (-> 204 / empty
        # body) when its query throws. "No answer" must never read as "no account".
        if body is None:
            raise AccountLookupError("Bootog could not check whether this email already has an account.")
        rows = body.get("data") if isinstance(body, dict) else body
        if not isinstance(rows, list):
            raise AccountLookupError("Bootog's account check returned an unexpected answer.")
        # Anything that doesn't have the documented shape makes the answer unusable: a
        # silently skipped entry could be exactly the account that makes this a duplicate.
        malformed = AccountLookupError("Bootog's account check returned data in an unexpected format.")
        accounts: list[ExistingAccount] = []
        for row in rows:
            if not isinstance(row, dict):
                raise malformed
            providers = row.get("providers")
            if providers is None:
                providers = []  # an account without any role (LEFT JOIN)
            if not isinstance(providers, list):
                raise malformed
            roles: list[AccountRole] = []
            for provider in providers:
                if provider is None:
                    continue
                if not isinstance(provider, dict) or not isinstance(provider.get("roles") or [], list):
                    raise malformed
                for role in provider.get("roles") or []:
                    if role is None:
                        continue
                    if not isinstance(role, dict):
                        raise malformed
                    if role.get("roleName"):
                        roles.append(AccountRole(
                            provider.get("providerId"), provider.get("providerName"), str(role["roleName"]),
                            deleted=role.get("isDelete") is True,
                            role_id=role.get("roleId") if is_guid(role.get("roleId")) else None,
                        ))
            accounts.append(ExistingAccount(row.get("userId"), row.get("userName"), roles))
        return accounts

    async def role_id(self, role_name: str, provider_id: str = EMPTY_GUID) -> str:
        """The id of BOOTOG's default `role_name` role for `provider_id` (the BOOTOG platform
        provider, the empty GUID, unless given), read from GET Roles/GetRoles.

        A role is used only when every check holds: exact name, that provider, isDefault
        and isActive true, a real GUID id. With roleNames the service filters by name only
        (isdefault is ignored), so other providers' copies of the role come back too and are
        skipped. No match, several matches, or a malformed answer raise: an account is never
        created with a role that could not be verified. API errors / timeouts raise as well."""
        body = await self._client.get(
            "Roles/GetRoles", params={"roleNames": role_name, "isdefault": "true", "targetProviderId": provider_id}
        )
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise RoleNotFoundError(f"BOOTOG's role lookup returned an unexpected answer, so the '{role_name}' role could not be verified.")
        candidates = [r for r in rows if isinstance(r, dict) and str(r.get("name") or "").casefold() == role_name.casefold()]
        valid = [
            r for r in candidates
            if r.get("name") == role_name
            and str(r.get("providerId") or EMPTY_GUID).casefold() == provider_id.casefold()
            and r.get("isDefault") is True
            and r.get("isActive") is True
            and is_guid(r.get("id"))
        ]
        if len(valid) == 1:
            return valid[0]["id"]
        if len(valid) > 1:
            raise RoleNotFoundError(f"Several default '{role_name}' roles were returned, so the role could not be verified.")
        if not candidates:
            raise RoleNotFoundError(f"The '{role_name}' role could not be found.")
        reasons = []
        for r in candidates:
            if str(r.get("providerId") or EMPTY_GUID).casefold() != provider_id.casefold():
                reasons.append("belongs to another provider")
            elif r.get("isDefault") is not True:
                reasons.append("is not the default role")
            elif r.get("isActive") is not True:
                reasons.append("is not active")
            else:
                reasons.append("has no valid id")
        raise RoleNotFoundError(
            f"The default '{role_name}' role could not be verified (the role returned {', '.join(dict.fromkeys(reasons))})."
        )

    async def add_vendor_user(
        self, fields: dict[str, Any], files: dict[str, tuple[str, bytes, str]] | None = None
    ) -> str | None:
        """Creates the account; returns the provider id from `value` (None when Bootog
        reports success without a usable id). Raises on HTTP or application failure.
        `fields` holds the password: it is sent once and never logged or returned.
        `files`: the LogoFile / HeadshotFile parts, when images are attached."""
        body = await self._client.post_form("UserDetail/AddVendorUser", fields, files)
        try:
            result = check_result(body, "Creating the account")
        except BootogResultError as exc:
            if _SERVER_FAULT.search(exc.message):
                # Bootog's own error text (no request data) for the backend team.
                logger.warning("bootog POST UserDetail/AddVendorUser -> server fault: %s", exc.message[:200])
                raise AccountCreationIncomplete(
                    "Bootog hit an internal error while creating the account (not a problem with the "
                    "details sent). Part of the account may already exist.",
                    code="SERVER_FAULT",
                ) from exc
            if _EMAIL_EXISTS.search(exc.message):
                # A definitive business rejection of this email; nothing was created.
                raise BootogResultError(exc.message, code="EMAIL_EXISTS") from exc
            raise
        value = result.get("value")
        return value if isinstance(value, str) and _GUID.match(value) and value != EMPTY_GUID else None


def is_guid(value: Any) -> bool:
    return isinstance(value, str) and bool(_GUID.match(value)) and value != EMPTY_GUID
