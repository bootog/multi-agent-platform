"""Run monitoring for the Frontend Order Enquiry Agent.

One `RunMonitor` per run collects what happened — graph steps, Bootog API calls, LLM
calls with their token usage, and business outcomes — and turns it into one audit
record at the end. The graph code only calls `step`, `api_call`, `llm_call` and
`outcome`; where the record is stored is decided by the caller (graph.py saves it in
the agent database through app.history).

What is never recorded: Authorization headers, bearer tokens, cookies, request bodies,
customer names, emails or phone numbers. Error texts are the user-safe messages of
BootogApiError (no URLs or tokens), cut to a short length.

"LLM tokens" here are model usage counts (input/output tokens), unrelated to the
Bootog bearer token used for API authentication."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_ERROR_MAX = 300


def _short(text: str | None) -> str | None:
    return text[:_ERROR_MAX] if text else None


@dataclass
class LlmUsage:
    model: str | None
    duration_ms: int
    # None = the model integration did not report usage for this call (not zero).
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class RunMonitor:
    run_id: str
    agent: str
    session_id: str | None = None
    # Opaque caller id (hash of the JWT subject; see owner_fingerprint). Never the token.
    owner_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    steps: list[dict[str, Any]] = field(default_factory=list)
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[LlmUsage] = field(default_factory=list)
    outcomes: dict[str, int] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter, repr=False)

    def step(self, step_id: str, status: str, duration_ms: int, error: str | None = None) -> None:
        self.steps.append(
            {
                "step": step_id,
                "status": status,
                "finishedAt": datetime.now(timezone.utc).isoformat(),
                "durationMs": duration_ms,
                "error": _short(error),
            }
        )

    def api_call(
        self,
        operation: str,
        correlation_id: str,
        status_code: int | None,
        duration_ms: int,
        attempts: int | None = None,
        page: Any = None,
        page_size: Any = None,
        result_count: int | None = None,
        timed_out: bool = False,
        error: str | None = None,
    ) -> None:
        self.api_calls.append(
            {
                "operation": operation,
                "correlationId": correlation_id,
                "statusCode": status_code,
                "durationMs": duration_ms,
                # Retries = attempts - 1 (only GETs are retried, see BootogClient.get).
                "attempts": attempts,
                "page": page,
                "pageSize": page_size,
                "resultCount": result_count,
                "timedOut": timed_out,
                "success": error is None,
                "error": _short(error),
            }
        )

    def llm_call(self, usage: LlmUsage) -> None:
        self.llm_calls.append(usage)

    def outcome(self, **counts: int) -> None:
        """Confirmed business outcomes, e.g. outcome(enquiries_retrieved=12).
        Values add up when the same key is reported more than once."""
        for key, value in counts.items():
            self.outcomes[key] = self.outcomes.get(key, 0) + int(value)

    def token_usage(self) -> dict[str, Any]:
        """Usage summed over every model call of the run. A total is None ("unavailable")
        when any call did not report it — a partial sum is never shown as the total."""

        def total(name: str) -> int | None:
            values = [getattr(call, name) for call in self.llm_calls]
            if not values or any(v is None for v in values):
                return None
            return sum(values)

        models = sorted({c.model for c in self.llm_calls if c.model})
        return {
            "modelCalls": len(self.llm_calls),
            "models": models,
            "inputTokens": total("input_tokens"),
            "outputTokens": total("output_tokens"),
            "totalTokens": total("total_tokens"),
            "modelDurationMs": sum(c.duration_ms for c in self.llm_calls),
            "usageAvailable": bool(self.llm_calls) and all(c.total_tokens is not None for c in self.llm_calls),
            # No pricing source is configured in this project, so no cost is estimated.
            "estimatedCostUsd": None,
        }

    def duration_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)

    def summary(self) -> dict[str, Any]:
        """Small monitoring block included in the run result for the UI."""
        return {
            "runId": self.run_id,
            "sessionId": self.session_id,
            "durationMs": self.duration_ms(),
            "apiCalls": len(self.api_calls),
            "tokenUsage": self.token_usage(),
        }

    def audit_record(
        self,
        status: str,
        intent: str | None,
        provider_id: str | None,
        date_from: str | None,
        date_to: str | None,
        error: str | None,
    ) -> dict[str, Any]:
        """Everything about the run, ready to store. Keys match
        AgentHistoryStore.record_agent_run."""
        usage = self.token_usage()
        return {
            "run_id": self.run_id,
            "agent_id": self.agent,
            "session_id": self.session_id,
            "owner_id": self.owner_id or "unknown",
            "status": status,
            "intent": intent,
            "provider_id": provider_id,
            "date_from": date_from,
            "date_to": date_to,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc),
            "duration_ms": self.duration_ms(),
            "llm_calls": usage["modelCalls"],
            "input_tokens": usage["inputTokens"],
            "output_tokens": usage["outputTokens"],
            "total_tokens": usage["totalTokens"],
            "detail": {
                "steps": self.steps,
                "apiCalls": self.api_calls,
                "llmCalls": [
                    {
                        "model": c.model,
                        "durationMs": c.duration_ms,
                        "inputTokens": c.input_tokens,
                        "outputTokens": c.output_tokens,
                        "totalTokens": c.total_tokens,
                    }
                    for c in self.llm_calls
                ],
                "tokenUsage": usage,
                "outcomes": self.outcomes,
            },
            "error": _short(error),
        }
