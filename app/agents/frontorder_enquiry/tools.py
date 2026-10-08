"""LangChain tools over the Bootog Order Enquiry APIs (API = tool).

As in the Contact Us agent, the run's Bootog client is injected through
RunnableConfig (`configurable.bootog_client`), so auth never appears in tool
arguments. Nodes call these deterministically today; the same tools can be bound to
an LLM later.

Only the confirmed read API is wired. Contact Log, AI Generate, Change Status and
Convert to Order are deliberately absent until their payloads and rules are confirmed."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agents.frontorder_enquiry.config import get_frontorder_enquiry_settings
from app.integrations.bootog import BootogClient
from app.integrations.bootog.order_enquiry import OrderEnquiryApi


def _client(config: RunnableConfig) -> BootogClient:
    return config["configurable"]["bootog_client"]


@tool
async def get_order_enquiries(config: RunnableConfig) -> dict[str, Any]:
    """Retrieve Frontend Order Enquiries, newest first."""
    settings = get_frontorder_enquiry_settings()
    page, truncated = await OrderEnquiryApi(_client(config)).list_all_enquiries(
        page_size=settings.page_size,
        max_pages=settings.max_pages,
    )
    return {"total": page.total, "records": page.records, "truncated": truncated}


FRONTORDER_ENQUIRY_TOOLS = [get_order_enquiries]
