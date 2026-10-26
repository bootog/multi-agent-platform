"""Central LLM factory. Agents call `get_llm()`; nothing else constructs OpenAI clients.

The chat model (OPENAI_CHAT_MODEL, default in app/core/config.py) powers LangGraph
text reasoning. One client per process is created lazily and reused by every node
and request, so the HTTP connection pool is shared."""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings


class LlmNotConfiguredError(Exception):
    """OPENAI_API_KEY is missing. The message is safe to show to users."""

    def __init__(self) -> None:
        super().__init__("The AI service is not configured.")


def build_llm(settings: Settings) -> BaseChatModel:
    if not settings.openai_api_key:
        raise LlmNotConfiguredError()
    kwargs = {}
    if settings.openai_temperature is not None:
        kwargs["temperature"] = settings.openai_temperature
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=2,
        **kwargs,
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """The shared chat model used by LangGraph nodes."""
    return build_llm(get_settings())
