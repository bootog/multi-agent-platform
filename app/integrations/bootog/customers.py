"""Customer creation for the one-shot run workflow (POST /contact-us/runs).

Customers ARE created by the chat agent: it collects missing details, checks for an
existing account, resolves the role and asks for confirmation before calling
UserDetail/AddVendorUser (see app/agents/contact_us/conversions.py). The run workflow
has no way to ask for details or a confirmation, so its create step stays blocked and
sends nothing."""

from typing import Any

from app.integrations.bootog.client import BootogClient

CUSTOMER_API_NOT_CONFIGURED = (
    "Customer creation is done in the Contact Us chat agent, which collects missing details "
    "and asks for confirmation first"
)


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
