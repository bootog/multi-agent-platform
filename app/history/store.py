"""Agent chat history: conversations and their messages, kept in the dedicated
AGENT_DB_NAME database — never in the core application database.

One database holds every conversation. A conversation is one row in a sessions table,
keyed by its session_id (the same id the chat API and the LangGraph thread use); its
messages are rows in a messages table. History is a record of what was said; it does not
replace the LangGraph checkpoint.

Each store has its own pair of tables, created with the same columns:
- `history_store`: `contact_agent_sessions` / `contact_agent_messages` (Contact agents);
- `frontorder_history_store`: `frontorder_enquiry_sessions` / `frontorder_enquiry_messages`
  (Frontend Order Enquiry Agent).

The same database also keeps `agent_run_audit`: one row per agent run (status, timing,
scope, LLM token usage, API calls), written by agents that record runs (the Frontend
Order Enquiry Agent).

History is optional at runtime: when PostgreSQL is unreachable or misconfigured the
chat keeps working without it, and initialization is retried on later use."""

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import asyncpg

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_RETRY_SECONDS = 30.0
_CONNECT_TIMEOUT_SECONDS = 10.0
_COMMAND_TIMEOUT_SECONDS = 10.0
_TITLE_MAX = 80
_EMPTY_SESSION_TTL = "1 day"  # a New Chat nobody wrote in is dropped after this

# Chat tables. {sessions} / {messages} are the table names of the store (see
# AgentHistoryStore.__init__): each agent can keep its conversations in its own tables.
CHAT_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS {sessions} (
        session_id  uuid        PRIMARY KEY,
        agent_id    text        NOT NULL,
        owner_id    text        NOT NULL,
        title       text,
        created_at  timestamptz NOT NULL DEFAULT now(),
        updated_at  timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS {sessions}_owner_idx
        ON {sessions} (owner_id, agent_id, updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS {messages} (
        id          bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id  uuid        NOT NULL REFERENCES {sessions} (session_id) ON DELETE CASCADE,
        role        text        NOT NULL CHECK (role IN ('user', 'assistant')),
        content     text        NOT NULL,
        result      jsonb,
        is_error    boolean     NOT NULL DEFAULT false,
        created_at  timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS {messages}_session_idx
        ON {messages} (session_id, created_at, id)
    """,
)

# Shared by every agent (rows are told apart by agent_id).
AUDIT_SCHEMA = (
    # One row per agent run: status, timing, scope, LLM token usage and a sanitized
    # `detail` (steps, API calls, model calls, business outcomes). Never tokens,
    # headers or customer contact details. Token columns are NULL when the model
    # integration did not report usage (unavailable, not zero).
    """
    CREATE TABLE IF NOT EXISTS agent_run_audit (
        run_id         text        PRIMARY KEY,
        agent_id       text        NOT NULL,
        session_id     text,
        owner_id       text        NOT NULL,
        status         text        NOT NULL,
        intent         text,
        provider_id    text,
        date_from      date,
        date_to        date,
        started_at     timestamptz NOT NULL,
        finished_at    timestamptz NOT NULL,
        duration_ms    integer     NOT NULL,
        llm_calls      integer     NOT NULL DEFAULT 0,
        input_tokens   integer,
        output_tokens  integer,
        total_tokens   integer,
        detail         jsonb       NOT NULL,
        error          text,
        created_at     timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS agent_run_audit_owner_idx
        ON agent_run_audit (owner_id, agent_id, started_at DESC)
    """,
)

# Owner + agent scope every statement: a caller only ever sees or changes their own
# conversations with this agent.
_UPSERT_SESSION = """
    INSERT INTO {sessions} AS s (session_id, agent_id, owner_id, title)
    VALUES ($1, $2, $3, $4)
    ON CONFLICT (session_id) DO UPDATE
       SET updated_at = now(), title = COALESCE(s.title, EXCLUDED.title)
     WHERE s.owner_id = EXCLUDED.owner_id AND s.agent_id = EXCLUDED.agent_id
    RETURNING session_id, created_at
"""
_TOUCH_SESSION = """
    UPDATE {sessions} SET updated_at = now()
     WHERE session_id = $1 AND owner_id = $2 AND agent_id = $3
    RETURNING session_id
"""
_INSERT_MESSAGE = """
    INSERT INTO {messages} (session_id, role, content, result, is_error)
    VALUES ($1, $2, $3, $4, $5)
"""
_PRUNE_EMPTY = """
    DELETE FROM {sessions} s
     WHERE s.owner_id = $1 AND s.agent_id = $2
       AND s.updated_at < now() - interval '{ttl}'
       AND NOT EXISTS (SELECT 1 FROM {messages} m WHERE m.session_id = s.session_id)
"""
# Conversations without messages (a New Chat not used yet) are not history.
_LIST_SESSIONS = """
    SELECT s.session_id, s.title, s.created_at, s.updated_at, count(m.id) AS message_count,
           (SELECT f.content FROM {messages} f
             WHERE f.session_id = s.session_id AND f.role = 'user'
             ORDER BY f.created_at, f.id LIMIT 1) AS first_message
      FROM {sessions} s
      JOIN {messages} m ON m.session_id = s.session_id
     WHERE s.owner_id = $1 AND s.agent_id = $2
     GROUP BY s.session_id
     ORDER BY s.updated_at DESC
     LIMIT $3
"""
_GET_SESSION = """
    SELECT session_id, title, created_at, updated_at
      FROM {sessions}
     WHERE session_id = $1 AND owner_id = $2 AND agent_id = $3
"""
_GET_MESSAGES = """
    SELECT id, role, content, result, is_error, created_at
      FROM {messages}
     WHERE session_id = $1
     ORDER BY created_at, id
     LIMIT $2
"""
_DELETE_SESSIONS = """
    DELETE FROM {sessions}
     WHERE owner_id = $1 AND agent_id = $2 AND session_id = ANY($3::uuid[])
    RETURNING session_id
"""
_INSERT_RUN_AUDIT = """
    INSERT INTO agent_run_audit (
        run_id, agent_id, session_id, owner_id, status, intent, provider_id, date_from, date_to,
        started_at, finished_at, duration_ms, llm_calls, input_tokens, output_tokens, total_tokens,
        detail, error)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
    ON CONFLICT (run_id) DO NOTHING
"""


class HistoryUnavailable(Exception):
    """The history database can't be used right now (not configured or unreachable)."""


class HistoryConfigError(Exception):
    """The history database can't be set up with the current configuration."""


@dataclass(frozen=True)
class HistorySession:
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


@dataclass(frozen=True)
class HistoryMessage:
    id: int
    role: str
    content: str
    result: dict[str, Any] | None
    is_error: bool
    created_at: datetime


def session_uuid(session_id: str | None) -> uuid.UUID | None:
    """The UUID behind a session id (hex or hyphenated form); None if it isn't one."""
    try:
        return uuid.UUID(str(session_id))
    except (TypeError, ValueError):
        return None


def public_session_id(value: uuid.UUID) -> str:
    """Session ids are exchanged with the UI in the chat API's form (32 hex digits)."""
    return value.hex


def title_from(message: str | None) -> str | None:
    """A history title from a user message: whitespace collapsed, cut at a word boundary."""
    text = re.sub(r"\s+", " ", message or "").strip()
    if len(text) <= _TITLE_MAX:
        return text or None
    cut = text[: _TITLE_MAX - 1]
    return (cut.rsplit(" ", 1)[0] if " " in cut[20:] else cut).rstrip(" ,.;:-") + "…"


def _date_or_none(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


class AgentHistoryStore:
    def __init__(
        self,
        sessions_table: str = "contact_agent_sessions",
        messages_table: str = "contact_agent_messages",
    ) -> None:
        # Table names are fixed in code (never user input); checked anyway because they
        # are put into the SQL text.
        for name in (sessions_table, messages_table):
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
                raise ValueError(f"Invalid table name: {name!r}")
        self.sessions_table = sessions_table
        self.messages_table = messages_table
        names = {"sessions": sessions_table, "messages": messages_table, "ttl": _EMPTY_SESSION_TTL}
        self._schema = [statement.format(**names) for statement in CHAT_SCHEMA] + list(AUDIT_SCHEMA)
        self._upsert_session = _UPSERT_SESSION.format(**names)
        self._touch_session = _TOUCH_SESSION.format(**names)
        self._insert_message = _INSERT_MESSAGE.format(**names)
        self._prune_empty = _PRUNE_EMPTY.format(**names)
        self._list_sessions = _LIST_SESSIONS.format(**names)
        self._get_session = _GET_SESSION.format(**names)
        self._get_messages = _GET_MESSAGES.format(**names)
        self._delete_sessions = _DELETE_SESSIONS.format(**names)

        self._settings: Settings | None = None
        self._enabled = True
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()
        self._next_attempt = 0.0

    def configure(self, settings: Settings | None = None, *, enabled: bool = True) -> None:
        """Use other settings (tests) or switch history off. Call while stopped."""
        self._settings, self._enabled, self._next_attempt = settings, enabled, 0.0

    @property
    def ready(self) -> bool:
        return self._pool is not None

    async def start(self) -> bool:
        """Create the database and tables if needed and open the pool. Never raises:
        a failure is logged and history stays off until a later retry."""
        async with self._lock:
            if self._pool is not None or not self._enabled:
                return self._pool is not None
            settings = self._settings or get_settings()
            try:
                self._pool = await self._open(settings)
            except HistoryConfigError as exc:
                logger.error("[AGENT_HISTORY] chat history disabled: %s", exc)
            except (OSError, asyncio.TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
                logger.error(
                    "[AGENT_HISTORY] chat history unavailable (%s: %s); retrying in %ss",
                    type(exc).__name__,
                    exc,
                    int(_RETRY_SECONDS),
                )
            if self._pool is None:
                self._next_attempt = time.monotonic() + _RETRY_SECONDS
                return False
            logger.info(
                "[AGENT_HISTORY] chat history ready (database %s, tables %s / %s)",
                settings.agent_db_name, self.sessions_table, self.messages_table,
            )
            return True

    async def close(self) -> None:
        async with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()

    async def _open(self, settings: Settings) -> asyncpg.Pool:
        name = settings.agent_db_name
        if not name:
            raise HistoryConfigError("AGENT_DB_NAME is not set.")
        if not settings.db_user:
            raise HistoryConfigError("DB_USER is not set.")
        connect = {
            "host": settings.db_host,
            "port": settings.db_port,
            "user": settings.db_user,
            "password": settings.db_password,
            "timeout": _CONNECT_TIMEOUT_SECONDS,
        }
        try:
            conn = await asyncpg.connect(database=name, **connect)
        except asyncpg.InvalidCatalogNameError:
            await self._create_database(name, connect)
            conn = await asyncpg.connect(database=name, **connect)
        except asyncpg.InvalidAuthorizationSpecificationError as exc:
            raise HistoryConfigError(f"PostgreSQL rejected DB_USER/DB_PASSWORD ({type(exc).__name__}).") from exc
        try:
            async with conn.transaction():
                for statement in self._schema:
                    await conn.execute(statement)
        except asyncpg.InsufficientPrivilegeError as exc:
            raise HistoryConfigError(f"DB_USER may not create tables in database {name!r}.") from exc
        finally:
            await conn.close()
        return await asyncpg.create_pool(
            database=name,
            min_size=1,
            max_size=5,
            command_timeout=_COMMAND_TIMEOUT_SECONDS,
            init=_init_connection,
            **connect,
        )

    @staticmethod
    async def _create_database(name: str, connect: dict[str, Any]) -> None:
        """Creates AGENT_DB_NAME through the server's maintenance database. Only ever
        creates; never drops or alters an existing database."""
        conn = await asyncpg.connect(database="postgres", **connect)
        try:
            if await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
                return
            await conn.execute(f"CREATE DATABASE {_quote_ident(name)}")
            logger.info("[AGENT_HISTORY] created database %s", name)
        except asyncpg.DuplicateDatabaseError:
            pass  # created concurrently by another instance
        except asyncpg.InsufficientPrivilegeError as exc:
            raise HistoryConfigError(
                f"database {name!r} does not exist and DB_USER lacks the CREATEDB privilege. "
                "Create it manually or grant CREATEDB."
            ) from exc
        finally:
            await conn.close()

    async def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None and self._enabled and time.monotonic() >= self._next_attempt:
            await self.start()
        if self._pool is None:
            raise HistoryUnavailable()
        return self._pool

    # --- Writes made while chatting (never raise: chat must not depend on history) ---

    async def create_session(self, session_id: str, owner: str, agent: str) -> datetime | None:
        """Registers a new, empty conversation; returns its server timestamp."""
        sid = session_uuid(session_id)
        if sid is None:
            return None
        try:
            pool = await self._require_pool()
            async with pool.acquire() as conn, conn.transaction():
                await conn.execute(self._prune_empty, owner, agent)
                row = await conn.fetchrow(self._upsert_session, sid, agent, owner, None)
            return row["created_at"] if row else None
        except HistoryUnavailable:
            return None
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, asyncio.TimeoutError) as exc:
            logger.warning("[AGENT_HISTORY] session=%s not recorded: %s", sid.hex, type(exc).__name__)
            return None

    async def record_message(
        self,
        session_id: str,
        owner: str,
        agent: str,
        role: str,
        content: str,
        result: dict[str, Any] | None = None,
        is_error: bool = False,
    ) -> bool:
        """Appends a message. A user message creates the conversation if needed (and
        titles it); an assistant message is only added to a conversation that exists,
        so a reply finishing after the conversation was deleted doesn't bring it back."""
        sid = session_uuid(session_id)
        if sid is None or not content:
            return False
        try:
            pool = await self._require_pool()
            async with pool.acquire() as conn, conn.transaction():
                if role == "user":
                    found = await conn.fetchrow(self._upsert_session, sid, agent, owner, title_from(content))
                else:
                    found = await conn.fetchrow(self._touch_session, sid, owner, agent)
                if found is None:
                    return False
                await conn.execute(self._insert_message, sid, role, content, result, is_error)
            return True
        except HistoryUnavailable:
            return False
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, asyncio.TimeoutError) as exc:
            logger.warning("[AGENT_HISTORY] session=%s %s message not recorded: %s", sid.hex, role, type(exc).__name__)
            return False

    async def record_agent_run(
        self,
        run_id: str,
        agent_id: str,
        session_id: str | None,
        owner_id: str,
        status: str,
        intent: str | None,
        provider_id: str | None,
        date_from: str | None,
        date_to: str | None,
        started_at: datetime,
        finished_at: datetime,
        duration_ms: int,
        llm_calls: int,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        detail: dict[str, Any],
        error: str | None = None,
    ) -> bool:
        """Stores one run's audit row. Returns False (and logs why) when it could not be
        stored; the run itself is never failed because of monitoring."""
        try:
            pool = await self._require_pool()
            await pool.execute(
                _INSERT_RUN_AUDIT,
                run_id, agent_id, session_id, owner_id, status, intent, provider_id,
                _date_or_none(date_from), _date_or_none(date_to), started_at, finished_at, duration_ms,
                llm_calls, input_tokens, output_tokens, total_tokens, detail, error,
            )
            return True
        except HistoryUnavailable:
            logger.warning("[AGENT_HISTORY] run=%s audit not recorded: database unavailable", run_id)
            return False
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, asyncio.TimeoutError) as exc:
            logger.warning("[AGENT_HISTORY] run=%s audit not recorded: %s", run_id, type(exc).__name__)
            return False

    # --- Reads and deletes for the history API (raise HistoryUnavailable) ---
    # --- Lists the caller's conversations for a specific agent
    async def list_sessions(self, owner: str, agent: str, limit: int = 100) -> list[HistorySession]:
        pool = await self._require_pool()
        rows = await self._run(pool.fetch(self._list_sessions, owner, agent, limit))
        return [
            HistorySession(
                session_id=public_session_id(r["session_id"]),
                title=r["title"] or title_from(r["first_message"]) or "New conversation",
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["message_count"],
            )
            for r in rows
        ]

    async def get_session(
        self, session_id: str, owner: str, agent: str, limit: int = 1000
    ) -> tuple[HistorySession, list[HistoryMessage]] | None:
        sid = session_uuid(session_id)
        if sid is None:
            return None
        pool = await self._require_pool()

        async def read() -> tuple[asyncpg.Record | None, list[asyncpg.Record]]:
            async with pool.acquire() as conn:
                head = await conn.fetchrow(self._get_session, sid, owner, agent)
                return head, (await conn.fetch(self._get_messages, sid, limit) if head else [])

        head, rows = await self._run(read())
        if head is None:
            return None
        messages = [
            HistoryMessage(
                id=r["id"],
                role=r["role"],
                content=r["content"],
                result=r["result"],
                is_error=r["is_error"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
        first_user = next((m.content for m in messages if m.role == "user"), None)
        session = HistorySession(
            session_id=public_session_id(head["session_id"]),
            title=head["title"] or title_from(first_user) or "New conversation",
            created_at=head["created_at"],
            updated_at=head["updated_at"],
            message_count=len(messages),
        )
        return session, messages

    async def delete_sessions(self, session_ids: list[str], owner: str, agent: str) -> list[str]:
        """Deletes the given conversations (their messages cascade) in one transaction.
        Returns the ids actually deleted; ids of other owners/agents are left alone."""
        ids = list({sid for sid in map(session_uuid, session_ids) if sid is not None})
        if not ids:
            return []
        pool = await self._require_pool()

        async def delete() -> list[asyncpg.Record]:
            async with pool.acquire() as conn, conn.transaction():
                return await conn.fetch(self._delete_sessions, owner, agent, ids)

        rows = await self._run(delete())
        return [public_session_id(r["session_id"]) for r in rows]

    @staticmethod
    async def _run(operation):
        try:
            return await operation
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, asyncio.TimeoutError) as exc:
            logger.warning("[AGENT_HISTORY] query failed: %s", type(exc).__name__)
            raise HistoryUnavailable() from exc


history_store = AgentHistoryStore()
# The Frontend Order Enquiry Agent keeps its conversations in its own tables.
frontorder_history_store = AgentHistoryStore("frontorder_enquiry_sessions", "frontorder_enquiry_messages")
