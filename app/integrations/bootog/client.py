"""Single HTTP client for the Bootog API. Domain modules (contact_us, providers,
customers) build on this; agents and nodes never call httpx directly.

Retries: only GET (safe, idempotent) is retried, on timeouts, transport errors and
502/503/504. POST/PUT are sent exactly once — a blind retry could create a second
account or repeat an update.

Anti-forgery (gateway CsrfValidationMiddleware, applied to authenticated POSTs): the
gateway issues two cookies on authenticated requests — `XSRF-TOKEN` (cookie token) and
`XSRF-REQUEST-TOKEN` (request token, bound to the signed-in user) — and validates a POST
only when the cookie comes back together with the request token in `X-XSRF-TOKEN`. The
host's AuthInterceptor does exactly this; so does this client: httpx keeps the cookies
for the life of the client (one client per caller and turn), and every write carries
the request token, fetched from GET /csrf-init first when no request has issued one
yet. No check is skipped: the token is the caller's own, for the caller's own token."""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

XSRF_HEADER = "X-XSRF-TOKEN"
XSRF_REQUEST_COOKIE = "XSRF-REQUEST-TOKEN"
CSRF_INIT_PATH = "csrf-init"
CSRF_REJECTED = "CSRF_INVALID"


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
    Bootog's own error text (which the host app already shows to users). `code` is
    Bootog's machine-readable error code when it sent one (e.g. CSRF_INVALID)."""

    def __init__(self, message: str, status_code: int | None = None, timed_out: bool = False, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.timed_out = timed_out
        self.code = code

    @property
    def outcome_unknown(self) -> bool:
        """The request may have been processed (no answer, or a server error)."""
        return self.timed_out or self.status_code is None or self.status_code >= 500


class BootogResultError(BootogApiError):
    """HTTP succeeded but Bootog's Result envelope says it didn't (`failed: true`).
    A definitive answer: the request was processed and rejected."""

    @property
    def outcome_unknown(self) -> bool:
        return False


def check_result(body: Any, action: str) -> dict[str, Any]:
    """Validates Bootog's `Result` envelope ({value, failed, message, failures, exception}).
    An HTTP 200 with `failed: true` is a failure, never a success."""
    if not isinstance(body, dict):
        raise BootogResultError(f"{action} returned an unexpected response.")
    if body.get("failed") is True or body.get("exception"):
        raise BootogResultError(f"{action} failed. {_result_reason(body)}".strip())
    return body


def _result_reason(body: dict[str, Any], limit: int = 200) -> str:
    parts: list[str] = []
    if isinstance(body.get("message"), str) and body["message"].strip():
        parts.append(body["message"].strip())
    for failure in (body.get("failures") or [])[:3]:
        text = failure if isinstance(failure, str) else (failure or {}).get("message") if isinstance(failure, dict) else None
        if isinstance(text, str) and text.strip() and text.strip() not in parts:
            parts.append(text.strip())
    return " ".join(parts)[:limit]


_RETRYABLE_STATUS = {502, 503, 504}
# X-Client-Type values this client may send. The gateway skips its CSRF check for
# "Mobile" and its session validation for "FrontEnd"; the latter is never allowed.
_ALLOWED_CLIENT_TYPES = {"mobile": "Mobile"}


class BootogClient:
    def __init__(
        self,
        auth: BootogAuth | None = None,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._settings = settings or get_settings()
        headers = (auth or BootogAuth()).headers(self._settings.bootog_api_token)
        client_type = _client_type(self._settings.bootog_client_type)
        if client_type:
            headers["X-Client-Type"] = client_type
        self._http = httpx.AsyncClient(
            base_url=self._settings.bootog_api_base_url + "/",
            headers=headers,
            timeout=self._settings.bootog_timeout_seconds,
            transport=transport,
        )

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        retries = max(0, self._settings.bootog_get_retries)
        for attempt in range(retries + 1):
            try:
                return await self._request("GET", path, params=params)
            except BootogApiError as exc:
                retryable = exc.timed_out or exc.status_code is None or exc.status_code in _RETRYABLE_STATUS
                if attempt >= retries or not retryable:
                    raise
                await asyncio.sleep(self._settings.bootog_retry_backoff_seconds * (attempt + 1))

    async def post(self, path: str, json: Any = None) -> Any:
        return await self._write("POST", path, json=json)

    async def post_form(self, path: str, fields: dict[str, Any]) -> Any:
        """multipart/form-data, as the host app's FormData posts. Never retried."""
        parts = {key: (None, _form_value(value)) for key, value in fields.items()}
        return await self._write("POST", path, files=parts)

    async def put(self, path: str, json: Any = None) -> Any:
        return await self._write("PUT", path, json=json)

    async def _write(self, method: str, path: str, **kwargs: Any) -> Any:
        token = await self._xsrf_request_token()
        headers = {XSRF_HEADER: token} if token else None
        return await self._request(method, path, headers=headers, **kwargs)

    async def _xsrf_request_token(self) -> str | None:
        """The anti-forgery request token the gateway issued to this caller, fetching one
        when none has been issued on this client yet. None when the gateway issues none
        (e.g. unauthenticated): the write then fails with the gateway's own answer."""
        token = self._cookie(XSRF_REQUEST_COOKIE)
        if token:
            return token
        try:
            await self._request("GET", CSRF_INIT_PATH)
        except BootogApiError as exc:
            logger.warning("bootog anti-forgery token request failed: HTTP %s", exc.status_code)
            return None
        return self._cookie(XSRF_REQUEST_COOKIE)

    def _cookie(self, name: str) -> str | None:
        for cookie in self._http.cookies.jar:
            if cookie.name == name and cookie.value:
                return cookie.value
        return None

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
            reason, code = _error_reason(response)
            # Endpoint, status, Bootog's code and its (short) error text only — never
            # headers, cookies, tokens or request bodies.
            logger.warning(
                "bootog %s %s -> HTTP %s%s%s",
                method, path, response.status_code, f" code={code}" if code else "", f": {reason}" if reason else "",
            )
            if code == CSRF_REJECTED:
                base = (
                    "Bootog's gateway rejected the change: its anti-forgery check failed, "
                    "so nothing was saved."
                )
                raise BootogApiError(base, response.status_code, code=code)
            if response.status_code in (401, 403):
                base = "Not authorized to access this Bootog data."
            else:
                base = f"The Bootog service returned an error (HTTP {response.status_code})."
            raise BootogApiError(f"{base} {reason}" if reason else base, response.status_code, code=code)

        if response.status_code == 204 or not response.content:
            return None
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


def _client_type(value: str | None) -> str | None:
    if not value:
        return None
    allowed = _ALLOWED_CLIENT_TYPES.get(value.strip().casefold())
    if allowed is None:
        logger.warning("BOOTOG_CLIENT_TYPE=%r is not an allowed value; no X-Client-Type header is sent", value[:20])
    return allowed


def _form_value(value: Any) -> str:
    """FormData semantics: booleans as "true"/"false", None as an empty string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _error_reason(response: httpx.Response, limit: int = 200) -> tuple[str | None, str | None]:
    """Bootog's error text and code, in the shapes the host app reads: {"error": ...,
    "code": ...}, {"message": ...} or ASP.NET validation {"title": ..., "errors": {...}}."""
    try:
        body = response.json()
    except ValueError:
        return None, None
    if not isinstance(body, dict):
        return None, None
    code = body.get("code") if isinstance(body.get("code"), str) else None
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
    return reason[:limit] or None, code
