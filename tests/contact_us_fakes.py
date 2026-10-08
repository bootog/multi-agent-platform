"""Shared fakes for the Contact Us chat tests: a Bootog API on httpx.MockTransport and
a scripted LLM. Nothing here touches the network."""

import re
import uuid
from datetime import date, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from langchain_core.messages import AIMessage

from app.agents.contact_us.agent_graph import run_contact_us_chat_turn
from app.agents.contact_us.understanding import TurnUnderstanding
from app.agents.runtime import RunChannel
from app.integrations.bootog import BootogClient

TODAY = date.today()


def _days_ago(n: int) -> str:
    return (TODAY - timedelta(days=n)).isoformat() + "T10:00:00"


RECORDS: list[dict[str, Any]] = [
    {
        "id": "req-bharathi", "firstName": "Bharathi", "lastName": "Krishnan", "emailId": "bharathidemo@mailinator.com",
        "phoneNumberCode": "+91", "phone": "9876543210", "moduleName": "Contact-Us", "status": 4,
        "comments": "Need an inspection quote ~ {\"x\":1}", "createdDate": _days_ago(2),
    },
    {
        "id": "req-diana", "firstName": "Diana", "lastName": "Prince", "emailId": "diana@example.com", "phone": "5550100",
        "address1": "1 Main St", "city": "Austin", "state": "TX", "zipcode": "73301", "county": "Travis",
        "country": "USA", "moduleName": "Contact-Us", "status": 3, "createdDate": _days_ago(9),
    },
    {
        "id": "req-arun-1", "firstName": "Arun", "lastName": "Kumar", "emailId": "arun.k@example.com",
        "moduleName": "contact", "status": 4, "createdDate": _days_ago(1),
    },
    {
        "id": "req-arun-2", "firstName": "Arun", "lastName": "Raj", "emailId": "arun.r@example.com",
        "moduleName": "contact", "status": 4, "createdDate": _days_ago(20),
    },
]


class FakeBootog:
    """Answers the documented Bootog calls; records every request."""

    def __init__(self, records: list[dict[str, Any]] | None = None, fail_contact_us: bool = False):
        self.records = records if records is not None else RECORDS
        self.fail_contact_us = fail_contact_us
        self.calls: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = urlparse(str(request.url)).path
        query = {k: v[0] for k, v in parse_qs(urlparse(str(request.url)).query).items()}
        if path.endswith("/ContactUs"):
            if self.fail_contact_us:
                return httpx.Response(503, json={"message": "Service unavailable"})
            rows = self.records
            if name := query.get("ContactName"):
                rows = [r for r in rows if name.casefold() in f"{r['firstName']} {r['lastName']}".casefold()]
            if m := re.search(r"createdDate>=(\d{4}-\d{2}-\d{2})", query.get("filter", "")):
                rows = [r for r in rows if r["createdDate"][:10] >= m.group(1)]
            if m := re.search(r"createdDate<=(\d{4}-\d{2}-\d{2})", query.get("filter", "")):
                rows = [r for r in rows if r["createdDate"][:10] <= m.group(1)]
            return httpx.Response(200, json={"count": len(rows), "data": rows})
        if path.endswith("/ProviderUserMapping/GetProviderUserMappingBySession"):
            return httpx.Response(200, json={"data": [{"providerId": "p1", "providerName": "Inspection Depot"}]})
        return httpx.Response(404, json={"message": "not found"})

    def contact_us_calls(self) -> list[dict[str, str]]:
        return [
            {k: v[0] for k, v in parse_qs(urlparse(str(c.url)).query).items()}
            for c in self.calls
            if urlparse(str(c.url)).path.endswith("/ContactUs")
        ]


class ScriptedLlm:
    """Stands in for the chat model: returns scripted interpretations in order and a
    reply that echoes the facts it was given (so tests can inspect them)."""

    def __init__(self, understandings: list[dict[str, Any]] | None = None):
        self.understandings = [TurnUnderstanding.model_validate(u) for u in understandings or []]
        self.facts: list[str] = []

    def push(self, **understanding: Any) -> None:
        self.understandings.append(TurnUnderstanding.model_validate({"request_summary": "test", **understanding}))

    def with_structured_output(self, schema):
        llm = self

        class _Structured:
            async def ainvoke(self, messages, config=None):
                return llm.understandings.pop(0)

        return _Structured()

    async def ainvoke(self, messages, config=None):
        self.facts.append(messages[-1].content)
        return AIMessage("reply")


async def turn(graph, session_id: str, message: str | None, llm, bootog: FakeBootog):
    channel = RunChannel(uuid.uuid4().hex, "contact-us")
    client = BootogClient(transport=bootog.transport())
    state = await run_contact_us_chat_turn(channel, session_id, message, client, graph=graph, llm=llm)
    return state, channel.events


def steps(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(e["step"], e["status"]) for e in events if e["type"] == "step" and e["status"] != "running"]


def final(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in events if e["type"] == "run")
