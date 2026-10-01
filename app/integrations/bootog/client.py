"""Single HTTP client for the Bootog API. Domain modules (contact_us, providers,
customers) build on this; agents and nodes never call httpx directly."""

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BootogAuth:
    """Caller identity forwarded to Bootog, mirroring what the Angular host's
    AuthInterceptor sends (bearer token, device id, tenant/role context)."""

    bearer_token: str | None = None
    tenant_id: str | None = None
    role_id: str | None = None
    device_id: str | None = None

    def headers(self, fallback_token: str | None) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        token = self.bearer_token or fallback_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if self.tenant_id:
            headers["X-Tenant-Id"] = self.tenant_id
        if self.role_id:
            headers["X-Role-Id"] = self.role_id
        if self.device_id:
            headers["X-Device-Id"] = self.device_id
        return headers


class BootogApiError(Exception):
    """A Bootog call failed. `message` is safe to show to users: no URLs or tokens, only
    Bootog's own error text (which the host app already shows to users)."""

    def __init__(self, message: str, status_code: int | None = None, timed_out: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.timed_out = timed_out


class BootogClient:
    def __init__(
        self,
        auth: BootogAuth | None = None,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._settings = settings or get_settings()
        self._http = httpx.AsyncClient(
            base_url=self._settings.bootog_api_base_url + "/",
            headers=(auth or BootogAuth()).headers(self._settings.bootog_api_token),
            timeout=self._settings.bootog_timeout_seconds,
            transport=transport,
        )

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, json: Any = None) -> Any:
        return await self._request("POST", path, json=json)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        clean_params = {k: v for k, v in (kwargs.pop("params", None) or {}).items() if v not in (None, "")}
        try:
            response = await self._http.request(method, path.lstrip("/"), params=clean_params or None, **kwargs)
        except httpx.TimeoutException as exc:
            logger.warning("bootog %s %s timed out", method, path)
            raise BootogApiError("The Bootog service did not respond in time.", timed_out=True) from exc
        except httpx.HTTPError as exc:
            logger.warning("bootog %s %s transport error: %s", method, path, type(exc).__name__)
            raise BootogApiError("The Bootog service could not be reached.") from exc

        if response.is_error:
            reason = _error_reason(response)
            logger.warning("bootog %s %s -> HTTP %s%s", method, path, response.status_code, f": {reason}" if reason else "")
            if response.status_code in (401, 403):
                base = "Not authorized to access this Bootog data."
            else:
                base = f"The Bootog service returned an error (HTTP {response.status_code})."
            raise BootogApiError(f"{base} {reason}" if reason else base, response.status_code)

        try:
            return response.json()
        except ValueError as exc:
            raise BootogApiError("The Bootog service returned an unreadable response.", response.status_code) from exc

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "BootogClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _error_reason(response: httpx.Response, limit: int = 200) -> str | None:
    """Bootog's error text, in the shapes the host app reads: {"error": ...},
    {"message": ...} or ASP.NET validation {"title": ..., "errors": {...}}."""
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    parts: list[str] = []
    for key in ("error", "message", "title", "detail"):
        value = body.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    errors = body.get("errors")
    if isinstance(errors, dict):
        for field, msgs in list(errors.items())[:3]:
            text = "; ".join(m for m in msgs if isinstance(m, str)) if isinstance(msgs, list) else str(msgs)
            parts.append(f"{field}: {text}")
    reason = " ".join(parts)
    return reason[:limit] or None
