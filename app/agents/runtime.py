"""Agent run registry + live event channel, shared by all agents.

Each run gets its own channel. Nodes publish structured events while the graph
executes; SSE subscribers replay what already happened and then follow live."""

import asyncio
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
