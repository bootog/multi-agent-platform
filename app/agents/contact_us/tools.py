"""LangChain tools over the documented Bootog APIs (API = tool).

The Bootog client for the current run is injected through RunnableConfig
(`configurable.bootog_client`), so tools carry the caller's auth without it ever
appearing in tool arguments. Nodes call them deterministically today; the same
tools can be bound to an LLM in later phases."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.core.config import get_settings
from app.integrations.bootog import BootogClient, ContactUsApi, CustomerApi, ProviderApi


def _client(config: RunnableConfig) -> BootogClient:
    return config["configurable"]["bootog_client"]


@tool
async def get_provider_list(config: RunnableConfig) -> list[dict[str, str]]:
    """List the companies (providers) available to the current Bootog session."""
    providers = await ProviderApi(_client(config)).list_for_session()
    return [{"providerId": p.provider_id, "providerName": p.provider_name} for p in providers]


@tool
async def get_contact_us_requests(config: RunnableConfig, provider_id: str | None = None) -> dict[str, Any]:
    """Retrieve open Contact Us requests, newest first, optionally narrowed to one provider."""
    page = await ContactUsApi(_client(config)).list_requests(
        provider_id=provider_id,
        page_size=get_settings().contact_us_page_size,
    )
    return {"total": page.total, "records": page.records}


@tool
async def create_customer(config: RunnableConfig, customer_payload: dict[str, Any]) -> dict[str, Any]:
    """Create a Bootog customer. Raises until the customer-creation API is documented and configured."""
    return await CustomerApi(_client(config)).create(customer_payload)


CONTACT_US_TOOLS = [get_provider_list, get_contact_us_requests, create_customer]
