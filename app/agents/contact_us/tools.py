"""LangChain tools over the documented Bootog APIs (API = tool).

The Bootog client for the current run is injected through RunnableConfig
(`configurable.bootog_client`), so tools carry the caller's auth without it ever
appearing in tool arguments. Nodes call them deterministically today; the same
tools can be bound to an LLM in later phases. No tool takes or returns a password."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.core.config import get_settings
from app.integrations.bootog import (
    AccountApi,
    BootogClient,
    ContactUsApi,
    ContractorApi,
    CustomerApi,
    ProviderApi,
)
from app.integrations.bootog.contact_us import CRM_COMPLETED_STATUS


def _client(config: RunnableConfig) -> BootogClient:
    return config["configurable"]["bootog_client"]


@tool
async def get_provider_list(config: RunnableConfig) -> list[dict[str, str]]:
    """List the companies (providers) available to the current Bootog session."""
    providers = await ProviderApi(_client(config)).list_for_session()
    return [{"providerId": p.provider_id, "providerName": p.provider_name} for p in providers]


@tool
async def get_contact_us_requests(
    config: RunnableConfig,
    provider_id: str | None = None,
    contact_name: str | None = None,
    email: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    module_filter: str | None = None,
    statuses: list[int] | None = None,
    all_statuses: bool = False,
    assigned_user_ids: list[str] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve Contact Us requests, newest first. Without a name/email: the latest page
    (open requests unless `statuses` or `all_statuses` is given). With a name and/or
    email: a server-side search across every page (bounded; `truncated` says if the bound
    was hit). `request_id`: that one request, whatever its status. `module_filter` scopes
    the records to one Contact module (default: Contact-Us)."""
    settings = get_settings()
    api = ContactUsApi(_client(config))
    filters = {
        "provider_id": provider_id,
        "created_from": created_from,
        "created_to": created_to,
        "module_filter": module_filter,
        "statuses": statuses,
        "all_statuses": all_statuses,
        "assigned_user_ids": assigned_user_ids,
    }
    if request_id:
        page = await api.list_requests(page_size=5, request_id=request_id, **{**filters, "all_statuses": True})
        return {"total": page.total, "records": page.records, "truncated": False, "search": True}
    if contact_name or email:
        page = await api.search_requests(
            settings.contact_us_page_size, settings.bootog_max_pages, name=contact_name, email=email, **filters
        )
        return {"total": page.total, "records": page.records, "truncated": page.truncated, "search": True}
    page = await api.list_requests(page_size=settings.contact_us_page_size, **filters)
    return {"total": page.total, "records": page.records, "truncated": False, "search": False}


@tool
async def get_contact_us_assigned_users(config: RunnableConfig, module_filter: str | None = None) -> list[dict[str, str]]:
    """List the users Contact Us requests are assigned to (for "assigned to X" filters)."""
    users = await ContactUsApi(_client(config)).assigned_users(module_filter)
    return [{"id": u.user_id, "name": u.user_name} for u in users]


@tool
async def mark_contact_request_converted(config: RunnableConfig, request_id: str) -> dict[str, Any]:
    """Set a Contact Us request to CRM Completed (status 11) after a successful conversion."""
    return await ContactUsApi(_client(config)).update_requests([request_id], {"Status": CRM_COMPLETED_STATUS})


@tool
async def search_service_providers(
    config: RunnableConfig, provider_name: str | None = None, all_pages: bool = True
) -> dict[str, Any]:
    """Search BOOTOG service providers by name prefix (every page when `all_pages`, bounded)."""
    settings = get_settings()
    api = ProviderApi(_client(config))
    if all_pages:
        page = await api.search_all_providers(provider_name, page_size=50, max_pages=settings.bootog_max_pages)
        providers, truncated = page.providers, page.truncated
    else:
        providers, truncated = await api.search_providers(provider_name, page_size=10), False
    return {
        "providers": [
            {"id": p.provider_id, "name": p.provider_name, "address": p.address, "phone": p.phone, "email": p.email}
            for p in providers
        ],
        "truncated": truncated,
    }


@tool
async def link_vendor_to_provider(config: RunnableConfig, vendor_id: str, target_provider_id: str) -> dict[str, Any]:
    """Link a vendor company to a service provider (idempotent on Bootog's side)."""
    return await ProviderApi(_client(config)).save_provider_vendor(vendor_id, target_provider_id)


@tool
async def get_accounts_by_email(config: RunnableConfig, email: str) -> list[dict[str, Any]]:
    """Existing BOOTOG accounts for an email, with their roles per provider."""
    accounts = await AccountApi(_client(config)).accounts_by_email(email)
    return [
        {
            "userId": a.user_id,
            "userName": a.user_name,
            "roles": [{"providerId": r.provider_id, "providerName": r.provider_name, "roleName": r.role_name} for r in a.roles],
        }
        for a in accounts
    ]


@tool
async def get_default_role_id(config: RunnableConfig, role_name: str) -> str:
    """Resolve the id of a default BOOTOG role by name (e.g. Customer, InsuranceCarrier)."""
    return await AccountApi(_client(config)).role_id(role_name)


@tool
async def get_contractor_subcategories(config: RunnableConfig, target_provider_id: str | None = None) -> dict[str, Any]:
    """Contractor service sub-categories available for a provider (blank names dropped)."""
    page = await ContractorApi(_client(config)).subcategories(
        target_provider_id, page_size=200, max_pages=get_settings().bootog_max_pages
    )
    return {
        "items": [{"id": s.id, "name": s.name, "category": s.category} for s in page.items],
        "total": page.total,
        "truncated": page.truncated,
    }


@tool
async def save_contractor_subcategories(
    config: RunnableConfig, subcategory_ids: list[str], target_provider_id: str
) -> Any:
    """Save the selected contractor sub-categories the way the partner screen does."""
    return await ContractorApi(_client(config)).save_provider_subcategories(subcategory_ids, target_provider_id)


@tool
async def create_customer(config: RunnableConfig, customer_payload: dict[str, Any]) -> dict[str, Any]:
    """Run-workflow customer creation. Always blocked: creation happens in the chat agent."""
    return await CustomerApi(_client(config)).create(customer_payload)


CONTACT_US_TOOLS = [
    get_provider_list,
    get_contact_us_requests,
    get_contact_us_assigned_users,
    mark_contact_request_converted,
    search_service_providers,
    link_vendor_to_provider,
    get_accounts_by_email,
    get_default_role_id,
    get_contractor_subcategories,
    save_contractor_subcategories,
    create_customer,
]
