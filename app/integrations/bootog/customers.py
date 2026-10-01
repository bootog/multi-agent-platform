"""Customer operations.

ContactUS.pdf does NOT document a customer-creation endpoint or its payload.
The only customer-related entry is `UserDetail/GetCustomerDetails`, listed without
parameters or a response, so it is not wired either.

`CustomerApi.create` therefore refuses to run until the real endpoint and payload
are confirmed. To enable it: implement the call here (path + exact field names from
the documented payload) and map `CustomerDraft` to that payload in `to_payload`.
Nothing else in the agent needs to change."""

from typing import Any

from app.integrations.bootog.client import BootogClient

CUSTOMER_API_NOT_CONFIGURED = "Customer creation API not yet configured"


class CustomerApiNotConfiguredError(Exception):
    def __init__(self) -> None:
        super().__init__(CUSTOMER_API_NOT_CONFIGURED)


class CustomerApi:
    def __init__(self, client: BootogClient):
        self._client = client

    @property
    def create_configured(self) -> bool:
        return False

    async def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise CustomerApiNotConfiguredError()
