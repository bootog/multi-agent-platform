"""Provider (company) lookups.

ContactUS.pdf, call 1: GET /ProviderUserMapping/GetProviderUserMappingBySession
-> {"count": n, "data": [{"providerId": ..., "providerName": ..., ...}]}
This is the list the Contact Us screen uses for its company selector
(ProviderUserService.getProvidersBySession in the Angular libs)."""

from dataclasses import dataclass

from app.integrations.bootog.client import BootogClient


@dataclass(frozen=True)
class Provider:
    provider_id: str
    provider_name: str


class ProviderApi:
    def __init__(self, client: BootogClient):
        self._client = client

    async def list_for_session(self) -> list[Provider]:
        body = await self._client.get("ProviderUserMapping/GetProviderUserMappingBySession")
        rows = body.get("data", []) if isinstance(body, dict) else body or []
        providers: dict[str, Provider] = {}
        for row in rows:
            provider_id = row.get("providerId")
            if provider_id and provider_id not in providers:
                providers[provider_id] = Provider(provider_id, (row.get("providerName") or "").strip())
        return sorted(providers.values(), key=lambda p: p.provider_name.lower())
