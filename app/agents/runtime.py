"""Agent run registry + live event channel, shared by all agents.

Each run gets its own channel. Nodes publish structured events while the graph
executes; SSE subscribers replay what already happened and then follow live.

Also shared by the chat agents: ChatSessions (who owns a conversation, one message at a
time, idle expiry) and owner_fingerprint (an opaque caller id from the bearer token)."""

import asyncio
import base64
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

_MAX_RUNS = 200
_FINISHED_TTL_SECONDS = 30 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunChannel:
    def __init__(self, run_id: str, agent: str):
        self.run_id = run_id
        self.agent = agent
        self.events: list[dict[str, Any]] = []
        self.finished = False
        self.finished_at: float | None = None
        self._cond = asyncio.Condition()

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    async def close(self) -> None:
        async with self._cond:
            self.finished = True
            self.finished_at = time.monotonic()
            self._cond.notify_all()

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        index = 0
        while True:
            async with self._cond:
                await self._cond.wait_for(lambda: index < len(self.events) or self.finished)
                batch = self.events[index:]
                index = len(self.events)
                done = self.finished
            for event in batch:
                yield event
            if done and index >= len(self.events):
                return


class RunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, RunChannel] = {}
        self._tasks: set[asyncio.Task] = set()

    def create(self, agent: str) -> RunChannel:
        self._prune()
        channel = RunChannel(uuid.uuid4().hex, agent)
        self._runs[channel.run_id] = channel
        return channel

    def get(self, run_id: str) -> RunChannel | None:
        return self._runs.get(run_id)

    def track(self, task: asyncio.Task) -> None:
        """Keep a reference so background runs are not garbage-collected mid-flight."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _prune(self) -> None:
        now = time.monotonic()
        for run_id, channel in list(self._runs.items()):
            if channel.finished_at and now - channel.finished_at > _FINISHED_TTL_SECONDS:
                del self._runs[run_id]
        while len(self._runs) >= _MAX_RUNS:
            oldest = next(iter(self._runs))
            del self._runs[oldest]


class RunEmitter:
    """Publishes the event shapes the UI understands: plan, step and run."""

    def __init__(self, channel: RunChannel):
        self._channel = channel

    @property
    def run_id(self) -> str:
        return self._channel.run_id

    async def plan(self, steps: list[dict[str, str]]) -> None:
        await self._publish({"type": "plan", "steps": steps})

    async def step(
        self,
        step: str,
        status: str,
        message: str,
        data: dict[str, Any] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        event: dict[str, Any] = {"type": "step", "step": step, "status": status, "message": message}
        if duration_ms is not None:
            event["durationMs"] = duration_ms
        if data:
            event["data"] = data
        await self._publish(event)

    async def finish(self, status: str, message: str, result: dict[str, Any]) -> None:
        await self._publish({"type": "run", "status": status, "message": message, "result": result})
        await self._channel.close()

    async def _publish(self, event: dict[str, Any]) -> None:
        await self._channel.publish({"runId": self.run_id, "agent": self._channel.agent, "timestamp": utc_now(), **event})


registry = RunRegistry()


# --- chat sessions (shared by the chat agents) ---------------------------------------


class ChatSessions:
    """Bookkeeping beside the checkpointer: who owns a session, one turn at a time,
    and expiry of idle sessions."""

    def __init__(self, checkpointer, ttl_seconds: float = 4 * 3600, max_sessions: int = 500):
        self._checkpointer = checkpointer
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._owners: dict[str, str] = {}
        self._agents: dict[str, str] = {}
        self._last_used: dict[str, float] = {}
        self._busy: set[str] = set()

    async def create(self, owner: str, agent: str) -> str:
        """Register a new session for this owner and agent; returns its id."""
        await self._prune()
        session_id = uuid.uuid4().hex
        self._owners[session_id] = owner
        self._agents[session_id] = agent
        self._last_used[session_id] = time.monotonic()
        return session_id

    async def restore(self, session_id: str, owner: str, agent: str) -> None:
        """Re-registers a conversation reopened from chat history under its own id."""
        await self._prune()
        self._owners[session_id] = owner
        self._agents[session_id] = agent
        self._last_used[session_id] = time.monotonic()

    def owner_of(self, session_id: str) -> str | None:
        """Fingerprint of the session's owner, or None for an unknown session."""
        return self._owners.get(session_id)

    def agent_of(self, session_id: str) -> str | None:
        """Canonical key of the Contact agent the session was started for."""
        return self._agents.get(session_id)

    def begin(self, session_id: str) -> bool:
        """Mark a turn as running. False when the session is already running one."""
        if session_id in self._busy:
            return False
        self._busy.add(session_id)
        self._last_used[session_id] = time.monotonic()
        return True

    def end(self, session_id: str) -> None:
        """Mark the session's turn as finished."""
        self._busy.discard(session_id)
        self._last_used[session_id] = time.monotonic()

    async def delete(self, session_id: str) -> None:
        """Forget the session and delete its saved graph state."""
        self._owners.pop(session_id, None)
        self._agents.pop(session_id, None)
        self._last_used.pop(session_id, None)
        await self._checkpointer.adelete_thread(session_id)

    async def _prune(self) -> None:
        """Delete idle sessions (unused longer than the TTL), then the least recently used
        ones if needed so that one more session fits under `max_sessions`. Sessions
        running a turn are never deleted."""
        now = time.monotonic()
        idle = [s for s, t in self._last_used.items() if s not in self._busy and now - t > self._ttl]
        oldest = sorted((t, s) for s, t in self._last_used.items() if s not in self._busy and s not in idle)
        idle += [s for _, s in oldest[: max(0, len(self._last_used) - len(idle) - self._max + 1)]]
        for session_id in idle:
            await self.delete(session_id)




def owner_fingerprint(bearer_token: str | None) -> str:
    """Opaque id of the caller; the token itself is never stored. Uses the JWT subject
    when available so a silently refreshed token keeps the same conversation. Bootog
    still validates the token itself on every API call."""
    identity = bearer_token or "anonymous"
    try:
        payload = bearer_token.split(".")[1]  # type: ignore[union-attr]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if isinstance(claims, dict) and isinstance(claims.get("sub"), str):
            identity = "sub:" + claims["sub"]
    except (AttributeError, IndexError, ValueError):
        pass
    return hashlib.sha256(identity.encode()).hexdigest()
