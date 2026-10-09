"""Shared fakes for the Contact Us chat tests: a Bootog API on httpx.MockTransport and
a scripted LLM. Nothing here touches the network.

FakeBootog follows the verified server contracts: the Gridify filter grammar on
GET /ContactUs (status clauses, FirstName/LastName/EmailId `=*x/i` contains, assignee,
dates; `ContactName` is ignored like the real service), page/pageSize paging, the plain
array of SearchProviders with prefix matching, multipart AddVendorUser returning the
provider id in `value`, idempotent SaveProviderVendor and bulk-update-file."""

import json
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
TENANT_ID = "11111111-1111-1111-1111-111111111111"
CUSTOMER_ROLE_ID = "ad709ed3-560a-41ab-ad59-c8a059b48a15"
CARRIER_ROLE_ID = "4f3bc8d4-a4d5-4677-947f-ebf48bf60369"
ROLE_IDS = {
    "Customer": CUSTOMER_ROLE_ID,
    "InsuranceCarrier": CARRIER_ROLE_ID,
    "InsuranceAgency": "a0000000-0000-0000-0000-000000000002",
    "RealEstateCompany": "a0000000-0000-0000-0000-000000000003",
    "Contractor": "a0000000-0000-0000-0000-000000000005",
}


def _days_ago(n: int) -> str:
    return (TODAY - timedelta(days=n)).isoformat() + "T10:00:00"


RECORDS: list[dict[str, Any]] = [
    {
        "id": "req-bharathi", "firstName": "Bharathi", "lastName": "Krishnan", "emailId": "bharathidemo@mailinator.com",
        "phoneNumberCode": "+91", "phone": "9876543210", "moduleName": "Contact-Us", "status": 4,
        "comments": "Need an inspection quote ~ {\"x\":1}", "createdDate": _days_ago(2),
        "assignedUserId": "u-michael", "assignedUserName": "Michael Rowan",
    },
    {
        "id": "req-diana", "firstName": "Diana", "lastName": "Prince", "emailId": "diana@example.com", "phone": "5550100123",
        "address1": "1 Main St", "city": "Austin", "state": "TX", "zipcode": "73301", "county": "Travis",
        "country": "USA", "latitude": 30.27, "longitude": -97.74, "companyName": "Themyscira Insurance",
        "moduleName": "Contact-Us", "status": 3, "createdDate": _days_ago(9),
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

PROVIDERS: list[dict[str, Any]] = [
    {"providerId": "p-alex", "providerName": "Alex Inspections", "providerAddress": "1 Bay St, Jacksonville, FL, 32202 Duval"},
    {"providerId": "p-alex-2", "providerName": "Alex Inspections Pro", "providerAddress": "9 Elm St, Miami, FL, 33101 Dade"},
    {"providerId": "p-ameri", "providerName": "AmeriPro Inspection Corporation", "providerAddress": "5 Oak Ave, Tampa, FL"},
    {"providerId": "p-blank", "providerName": "   ", "providerAddress": "nowhere"},
    {"providerId": "p-coastal", "providerName": "Coastal Home Inspections", "providerAddress": "2 Beach Rd, Jacksonville, FL"},
]

SUBCATEGORIES: list[dict[str, Any]] = [
    {"id": "sc-roof", "name": "Roofing", "contractorCategory": "Exterior", "isActive": True},
    {"id": "sc-plumb", "name": "Plumbing", "contractorCategory": "Interior", "isActive": True},
    {"id": "sc-plumb-em", "name": "Emergency Plumbing", "contractorCategory": "Interior", "isActive": True},
    {"id": "sc-blank", "name": "  ", "contractorCategory": "Interior", "isActive": True},
]


class FakeBootog:
    """Answers the Bootog calls the agent makes; records every request."""

    def __init__(self, records: list[dict[str, Any]] | None = None, fail_contact_us: bool = False):
        self.records = [dict(r) for r in (records if records is not None else RECORDS)]
        self.providers = [dict(p) for p in PROVIDERS]
        self.subcategories = [dict(s) for s in SUBCATEGORIES]
        self.fail_contact_us = fail_contact_us
        # email -> [{"providerId", "providerName", "roleName"}]
        self.accounts: dict[str, list[dict[str, Any]]] = {}
        # Failure switches for the write calls.
        self.create_failure: str | None = None  # "failed" (HTTP 200 failed:true) | "timeout" | "timeout_after_create"
        self.fail_link = False
        self.fail_status_update = False
        self.fail_subcategories = False
        # Gateway anti-forgery: tokens issued on GETs, required on POSTs.
        self.issue_xsrf = True
        self.enforce_csrf = True
        self.xsrf_pairs: dict[str, str] = {}
        self.calls: list[httpx.Request] = []
        self.created_forms: list[dict[str, str]] = []
        self.vendor_links: list[dict[str, Any]] = []
        self.status_updates: list[dict[str, Any]] = []
        self.subcategory_inserts: list[Any] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    # --- routing -------------------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        response = self._route(request)
        if request.method == "GET" and self.issue_xsrf:
            response = self._with_xsrf_tokens(response)
        return response

    # --- gateway anti-forgery (CsrfTokenIssuerMiddleware / CsrfValidationMiddleware) ------

    def _with_xsrf_tokens(self, response: httpx.Response) -> httpx.Response:
        cookie_token = "ck-" + uuid.uuid4().hex
        request_token = "rq-" + uuid.uuid4().hex
        self.xsrf_pairs[request_token] = cookie_token
        cookies = [
            ("set-cookie", f"{name}={value}; Domain=.bootog.com; Path=/; Secure; SameSite=None")
            for name, value in (("XSRF-TOKEN", cookie_token), ("XSRF-REQUEST-TOKEN", request_token))
        ]
        return httpx.Response(response.status_code, headers=[*response.headers.multi_items(), *cookies], content=response.content)

    def _csrf_rejection(self, request: httpx.Request) -> httpx.Response | None:
        if request.method != "POST" or not self.enforce_csrf:  # the gateway checks POST only
            return None
        cookies = dict(c.split("=", 1) for c in request.headers.get("cookie", "").split("; ") if "=" in c)
        token = request.headers.get("x-xsrf-token")
        if token and self.xsrf_pairs.get(token) == cookies.get("XSRF-TOKEN"):
            return None
        return httpx.Response(403, json={"error": "CSRF validation failed", "code": "CSRF_INVALID"})

    def _route(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = urlparse(str(request.url)).path
        query = {k: v[0] for k, v in parse_qs(urlparse(str(request.url)).query).items()}
        if rejected := self._csrf_rejection(request):
            return rejected
        if path.endswith("/csrf-init"):
            return httpx.Response(200, json={"status": "csrf-ready"})
        if path.endswith("/ContactUs"):
            return self._contact_us(query)
        if path.endswith("/ContactUs/GetContactUsAssignedUser"):
            users = {r["assignedUserId"]: r["assignedUserName"] for r in self.records if r.get("assignedUserId")}
            users["u-mia"] = "Mia Rowan-Lee"
            return httpx.Response(200, json=[{"assignedUserId": k, "assignedUserName": v} for k, v in users.items()])
        if path.endswith("/ContactUs/bulk-update-file"):
            return self._bulk_update(json.loads(request.content))
        if path.endswith("/ProviderUserMapping/GetProviderUserMappingBySession"):
            return httpx.Response(200, json={"data": [{"providerId": "p1", "providerName": "Inspection Depot"}]})
        if path.endswith("/Provider/SearchProviders"):
            return self._search_providers(query)
        if path.endswith("/Provider/SaveProviderVendor"):
            body = json.loads(request.content)
            self.vendor_links.append(body)
            if self.fail_link:
                return httpx.Response(500, json={"message": "Database unavailable"})
            return httpx.Response(200, json={"failed": False, "message": "", "failures": []})
        if path.endswith("/UserDetail/GetUserRolesByEmailId"):
            return self._accounts(query.get("emailId", ""))
        if path.endswith("/Roles/GetRoles"):
            name = query.get("roleNames")
            rows = [{"id": ROLE_IDS[name], "name": name, "isDefault": True}] if name in ROLE_IDS else []
            return httpx.Response(200, json={"count": len(rows), "data": rows})
        if path.endswith("/UserDetail/AddVendorUser"):
            return self._add_vendor_user(request)
        if path.endswith("/ContractorSubCategory"):
            size = int(query.get("pageSize", "0")) or len(self.subcategories)
            page = int(query.get("page", "1"))
            rows = self.subcategories[(page - 1) * size: page * size]
            return httpx.Response(200, json={"count": len(self.subcategories), "data": rows})
        if path.endswith("/ProviderContractorSubCategory/BulkInsert"):
            self.subcategory_inserts.append(json.loads(request.content))
            if self.fail_subcategories:
                return httpx.Response(200, json={"failed": True, "message": "Sub-category not found"})
            return httpx.Response(200, json={"failed": False, "message": ""})
        return httpx.Response(404, json={"message": "not found"})

    # --- endpoints -----------------------------------------------------------------------

    def _contact_us(self, query: dict[str, str]) -> httpx.Response:
        if self.fail_contact_us:
            return httpx.Response(503, json={"message": "Service unavailable"})
        flt = query.get("filter", "")
        rows = self.records
        if m := re.search(r"\((status=\d+(?:\|status=\d+)*)\)", flt):
            wanted = {int(x) for x in re.findall(r"status=(\d+)", m.group(1))}
            rows = [r for r in rows if r.get("status") in wanted]
        else:
            excluded = {int(x) for x in re.findall(r"status!=(\d+)", flt)}
            rows = [r for r in rows if r.get("status") not in excluded]
        if m := re.search(r"createdDate>=(\d{4}-\d{2}-\d{2})", flt):
            rows = [r for r in rows if r["createdDate"][:10] >= m.group(1)]
        if m := re.search(r"createdDate<=(\d{4}-\d{2}-\d{2})", flt):  # midnight, like the service
            rows = [r for r in rows if r["createdDate"] <= m.group(1)]
        if m := re.search(r"createdDate<(\d{4}-\d{2}-\d{2})", flt):
            rows = [r for r in rows if r["createdDate"][:10] < m.group(1)]
        if m := re.search(r"(?:^|,)id=([A-Za-z0-9-]+)", flt):
            rows = [r for r in rows if r["id"] == m.group(1)]
        if ids := re.findall(r"assignedUserId=([A-Za-z0-9-]+)", flt):
            rows = [r for r in rows if r.get("assignedUserId") in ids]
        for field, key in (("FirstName", "firstName"), ("LastName", "lastName"), ("EmailId", "emailId")):
            if m := re.search(rf"{field}=\*([^,/]+)/i", flt):
                rows = [r for r in rows if m.group(1).casefold() in (r.get(key) or "").casefold()]
        size = int(query.get("pageSize", "12"))
        page = int(query.get("page", "1"))
        return httpx.Response(200, json={"count": len(rows), "data": rows[(page - 1) * size: page * size]})

    def _bulk_update(self, body: dict[str, Any]) -> httpx.Response:
        self.status_updates.append(body)
        if self.fail_status_update:
            return httpx.Response(500, json={"message": "Update failed.", "error": "db"})
        for r in self.records:
            if r["id"] in body["ids"]:
                r["status"] = body["updateValues"]["Status"]
        return httpx.Response(200, json={"message": "Update completed successfully."})

    def _search_providers(self, query: dict[str, str]) -> httpx.Response:
        prefix = (query.get("providerName") or "").casefold()
        rows = [p for p in self.providers if p["providerName"].casefold().startswith(prefix)]
        rows.sort(key=lambda p: p["providerName"])
        size = int(query.get("pageSize", "10"))
        page = int(query.get("page", "1"))
        return httpx.Response(200, json=rows[(page - 1) * size: page * size])

    def _accounts(self, email: str) -> httpx.Response:
        roles = self.accounts.get(email)  # exact, case-sensitive like the real service
        if not roles:
            return httpx.Response(200, json=[])
        providers: dict[str, dict[str, Any]] = {}
        for r in roles:
            p = providers.setdefault(r["providerId"], {"providerId": r["providerId"], "providerName": r.get("providerName"), "roles": []})
            p["roles"].append({"roleId": ROLE_IDS.get(r["roleName"]), "roleName": r["roleName"]})
        return httpx.Response(200, json=[{"userId": "user-" + email, "userEmail": email, "providers": list(providers.values())}])

    def _add_vendor_user(self, request: httpx.Request) -> httpx.Response:
        content_type = request.headers.get("content-type", "")
        assert content_type.startswith("multipart/form-data"), content_type  # [FromForm]
        form = {
            m.group(1).decode(): m.group(2).decode()
            for m in re.finditer(rb'name="([^"]+)"\r\n\r\n(.*?)\r\n--', request.content, re.S)
        }
        self.created_forms.append(form)
        if self.create_failure == "failed":
            return httpx.Response(200, json={"value": None, "failed": True, "message": "Company name already exists", "failures": []})
        if self.create_failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        role = next(name for name, rid in ROLE_IDS.items() if rid == form["RoleId"])
        email = form["Email"]
        if any(r["roleName"] == role for r in self.accounts.get(email, [])):
            return httpx.Response(200, json={"failed": True, "message": "User Email Id Already Exists!", "failures": []})
        provider_id = TENANT_ID if form["IsSkipTenant"] == "false" else str(uuid.uuid4())
        self.accounts.setdefault(email, []).append({"providerId": provider_id, "providerName": form["ProviderName"], "roleName": role})
        if self.create_failure == "timeout_after_create":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(
            200, json={"value": provider_id, "failed": False, "message": "User added successfully", "failures": [], "exception": None}
        )

    # --- inspection helpers --------------------------------------------------------------

    def contact_us_calls(self) -> list[dict[str, str]]:
        return [
            {k: v[0] for k, v in parse_qs(urlparse(str(c.url)).query).items()}
            for c in self.calls
            if urlparse(str(c.url)).path.endswith("/ContactUs")
        ]

    def paths(self) -> list[str]:
        return [urlparse(str(c.url)).path.split("/api/v1/")[-1] for c in self.calls]

    def writes(self) -> list[str]:
        """POST/PUT calls sent, in order."""
        return [
            urlparse(str(c.url)).path.split("/api/v1/")[-1]
            for c in self.calls
            if c.method in ("POST", "PUT")
        ]


class ScriptedLlm:
    """Stands in for the chat model: returns scripted interpretations in order and a
    reply that echoes the facts it was given (so tests can inspect them)."""

    def __init__(self, understandings: list[dict[str, Any]] | None = None):
        self.understandings = [TurnUnderstanding.model_validate(u) for u in understandings or []]
        self.facts: list[str] = []
        self.prompts: list[str] = []

    def push(self, **understanding: Any) -> None:
        self.understandings.append(TurnUnderstanding.model_validate({"request_summary": "test", **understanding}))

    def with_structured_output(self, schema):
        llm = self

        class _Structured:
            async def ainvoke(self, messages, config=None):
                llm.prompts.append("\n".join(str(m.content) for m in messages))
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
