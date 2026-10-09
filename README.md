# Bootog Multi-Agent Platform

FastAPI + LangGraph backend for the Bootog agents. The Angular host
(`paperless-micro-frontends/projects/host/src/app/agenticai`) talks only to this
service; this service calls the Bootog APIs.

```
Angular (agenticai) ──HTTP/SSE──> multi-agent-platform (LangGraph) ──> Bootog API
```

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
.\.venv\Scripts\python.exe -m pytest
```

| Variable | Default | Purpose |
|---|---|---|
| `BOOTOG_API_BASE_URL` | `https://api.dev.bootog.com/api/v1` | Bootog API base URL |
| `BOOTOG_API_TOKEN` | – | Optional service token, used only when the caller forwards none |
| `BOOTOG_TIMEOUT_SECONDS` | `20` | Per-request timeout |
| `CONTACT_US_PAGE_SIZE` | `50` | Contact Us requests read per run (newest first); also the page size of searches |
| `BOOTOG_GET_RETRIES` | `2` | Extra attempts for GETs on timeout / transport error / 502-504. POST/PUT are never retried |
| `BOOTOG_RETRY_BACKOFF_SECONDS` | `0.3` | Linear backoff between GET retries |
| `BOOTOG_MAX_PAGES` | `10` | Page cap for lookups that page through a list (request search, providers, sub-categories). Hitting it is reported, never hidden |
| `BOOTOG_CLIENT_TYPE` | – | Not needed: writes pass the gateway's anti-forgery check with the caller's own token (see below). Only `Mobile` is accepted (it makes the gateway skip that check) and only if the Bootog team authorises it; any other value, notably `FrontEnd` (skips session validation), is ignored with a warning |
| `ALLOWED_ORIGINS` | `http://localhost:4201` | CORS origins (comma-separated) |
| `OPENAI_API_KEY` | – | Required for the chat agent. Never logged or sent to the browser |
| `OPENAI_CHAT_MODEL` | `gpt-4.1-mini` | Model for LangGraph text reasoning (understanding, extraction, replies) |
| `OPENAI_TEMPERATURE` | `0.2` | Empty = model default (needed for reasoning models) |
| `OPENAI_TIMEOUT_SECONDS` | `30` | Per LLM call |
| `OPENAI_REALTIME_MODEL` | – | Reserved for realtime/voice features; not used for chat reasoning |
| `DB_HOST` / `DB_PORT` | `localhost` / `5432` | PostgreSQL server of the agent chat-history database |
| `DB_USER` / `DB_PASSWORD` | – | PostgreSQL login (needs CREATEDB only if `AGENT_DB_NAME` doesn't exist yet) |
| `AGENT_DB_NAME` | – | Dedicated chat-history database (e.g. `AgentDB`), created with its tables on startup if missing. Never the core app database |

### Chat history (`app/history/store.py`)

One database (`AGENT_DB_NAME`), two tables: `contact_agent_sessions` (one row per
conversation, keyed by the chat `session_id`) and `contact_agent_messages` (FK, `ON DELETE
CASCADE`). Everything is scoped to the caller (JWT-subject fingerprint) and the Contact agent.

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/agents/contact-us/session` | New Chat: new session id |
| `GET /api/v1/agents/contact-us/history?agentKey=` | Stored conversations, newest first |
| `GET /api/v1/agents/contact-us/history/{sessionId}?agentKey=` | One conversation's messages |
| `DELETE /api/v1/agents/contact-us/history` `{sessionIds, agentKey}` | Permanently delete selected conversations |

`POST /contact-us/chat` records each turn (user message, then the agent's reply). A session id
that is no longer live (restart, idle expiry) but is in the caller's history is continued under
the same id, and its messages seed the LangGraph thread as conversational context. If PostgreSQL
is unreachable, chat works as before and the history endpoints return 503.

`.env` in the project root is loaded at startup; real environment variables win.

Auth: the caller's `Authorization: Bearer …`, `X-Tenant-Id`, `X-Role-Id` and `X-Device-Id`
headers are forwarded to Bootog, so the agent acts with the signed-in user's access. Tokens
are never logged.

Anti-forgery: the gateway (`CsrfValidationMiddleware`) checks authenticated POSTs. On
authenticated requests it sets the `XSRF-TOKEN` cookie and the `XSRF-REQUEST-TOKEN` cookie
(a request token bound to the signed-in user); a POST passes when that cookie comes back with
the request token in `X-XSRF-TOKEN` — what the host's AuthInterceptor does. `BootogClient`
does the same: it keeps the cookies for the turn and sends the token on every write, calling
`GET /csrf-init` first when no request has issued one. A rejection is reported with its
code (`CSRF_INVALID`), never as success. Logs carry endpoint, status, Bootog's code and short
error text only.

## Layout

```
app/
  main.py                        FastAPI app, CORS, routers
  core/config.py, logging.py     settings from env; step logging (ids/status/timing only)
  integrations/bootog/           the only code that calls Bootog
    client.py                    BootogClient: auth headers, timeouts, GET-only retries, Result checks
    providers.py                 session companies, SearchProviders, SaveProviderVendor
    contact_us.py                GET ContactUs (filter grammar, search, paging), assignees, bulk-update-file
    accounts.py                  GetUserRolesByEmailId, Roles/GetRoles, AddVendorUser (multipart)
    contractors.py               ContractorSubCategory, ProviderContractorSubCategory/BulkInsert
    customers.py                 run-workflow stub: creation happens in the chat agent
  agents/
    runtime.py                   run registry + per-run event channel (SSE source)
    contact_us/
      state.py                   ContactUsAgentState
      schemas.py                 request/response models, customer draft mapping
      tools.py                   LangChain tools over the Bootog APIs
      nodes.py                   graph nodes; each emits running/completed/failed/blocked
      graph.py                   the LangGraph workflow + run executor
      understanding.py           LLM decision layer (typed interpretation, replies)
      agent_nodes.py             chat nodes around the existing ones; activity emitter; reply facts
      conversions.py             conversion types (role, provider type, form, steps), validation, FormData
      conversion_nodes.py        conversion / provider / assignee nodes
      agent_graph.py             chat state machine (router), sessions, turn executor
  llm/openapi.py                 get_llm(): the shared chat model
  prompts/contact_us.py          chat agent prompts
  api/routes/contact_us.py       HTTP endpoints
```

## Contact Us Agent — chat

The Contact Us page is a chat workspace. Each user message is one LangGraph turn over
a checkpointed session (`agents/contact_us/agent_graph.py`):

```
start_turn → understand_message (LLM) → [resolve_provider] → [search_providers]
  → [resolve_assigned_user] → [get_contact_us_data] → [match_request]
  → [select_contact_request → prepare_customer_data] → [merge_customer_fields]
  → conversion:
       start_conversion → [choose_conversion_type] → [resolve_target_provider]
       → validate_customer_data ─ missing/invalid → ask_for_missing   (waits)
       → check_existing_account → resolve_role → [select_subcategories]
       → confirm_conversion                                          (waits for "yes")
       → create_account → verify_account → [link_provider_vendor] → [save_subcategories]
       → update_request_status → refresh_requests
  → respond (LLM) → END
```

- **LLM** (`understanding.py`, prompts in `app/prompts/contact_us.py`): turns the message
  into a typed `TurnUnderstanding` (intent, goal, request reference, date/status/assignee/
  company filters, form values, partner type, target provider, option picks, extra notes)
  and writes the reply from structured facts.
- **Router** (`next_step`): deterministic; decides the next node from that interpretation
  plus state. The LLM never chooses or builds an API call, and never supplies an id.
- **Memory**: selected request, user-supplied fields (latest value wins, empty values
  ignored), extra notes, the conversion in progress (`conversion`: type, role, provider,
  completed/failed steps, created ids) and history persist per session. Extra notes are
  never sent to Bootog.
- **Sessions** live in memory (`InMemorySaver`); swap for a persistent LangGraph saver to
  survive restarts. A session belongs to the caller that created it (JWT `sub`). After a
  restart only the transcript is restored; a half-finished conversion is then caught by the
  existing-account check (nothing is created twice) but has to be resumed in Bootog.

### Searching

`GET /ContactUs` ignores the `ContactName` query parameter, so names and emails are searched
with the filter grammar (`FirstName=*x/i`, `LastName=*x/i`, `EmailId=*x/i`) over every page up
to `BOOTOG_MAX_PAGES`, then verified against the full name/email. Several matches are always
shown, numbered, for the user to choose (by number, request id, date, status or assignee); the
chat UI's request cards send the request id. A number is resolved to that card's id before
anything is fetched and only counts when the message contains it; naming the person already
selected keeps the selection. Plain listings show open requests; a listing by assignee or
period covers every status unless the user asks for open ones. Date ranges include the whole
last day (`createdDate<next day`). Status filters replace the open-status clause
(`(status=11|status=12)`), assignee filters resolve the name through
`ContactUs/GetContactUsAssignedUser` (cached 10 minutes per session) and add
`(assignedUserId=…)`. Provider search uses `Provider/SearchProviders` (prefix match, every page);
when nothing starts with the text, or a location is given, the full list is matched instead.

### Conversions

The Contact Us screen offers **Customer**, **B2B** and **Partners**; each maps to one entry in
`conversions.py` (verified against `contact-us.component.ts`, `customer-detail-popup`,
`agency-popup` and the Identity/Providers services):

| Type | Role (`Roles/GetRoles`) | ProviderType | IsSkipTenant | After `AddVendorUser` |
|---|---|---|---|---|
| Customer | `Customer` | 1 | false | status 11 |
| B2B Client (Insurance Carrier) | `InsuranceCarrier` | 4 | true | `SaveProviderVendor` → status 11 |
| Partner — Insurance Agency | `InsuranceAgency` | 2 | true | `SaveProviderVendor` → status 11 |
| Partner — Real Estate Agency | `RealEstateCompany` | 3 | true | `SaveProviderVendor` → status 11 |
| Partner — Contractor | `Contractor` | 5 | true | `SaveProviderVendor` → sub-category `BulkInsert` → status 11 |

- Required: first/last name, email, phone (10 digits, sent as `(xxx) xxx-xxxx`), street,
  city, state, zip, county, country; companies also need the company name. Website, bio,
  years in business, revenue, address line 2 and coordinates are optional. Coordinates come
  from the request or the user, never invented (sent empty otherwise).
- Status 11 = `CRMCompleted`, set with `PUT ContactUs/bulk-update-file {"updateValues": {"Status": 11}}`.
- `AddVendorUser` is multipart, sent once. Its `value` is the company (provider) id, used as
  `vendorId`. The password is generated server-side per call, sent only in that request and
  emailed by Bootog as a temporary password; it is never stored, logged, traced or shown.
- Nothing is created without confirming exactly what is shown; any later change asks again.
- An email that already has the type's role is refused; one with other roles is explained.
- A failed step keeps what succeeded; "retry" resumes at the failed step. If the
  `AddVendorUser` answer is lost (timeout / 5xx), the retry first checks Bootog and adopts the
  account if it exists, so nothing is created twice. The request is marked CRM Completed only
  after every earlier step succeeded; the list is then re-read.

| Endpoint | |
|---|---|
| `POST /api/v1/agents/contact-us/chat` | `{sessionId?, message?}` → `{runId, sessionId, sessionRestarted}` (202). No message = "Run Agent" (start the session and load the latest requests) |
| `DELETE /api/v1/agents/contact-us/chat/{sessionId}` | Clear the conversation |
| `GET /api/v1/agents/runs/{runId}/events` | SSE: `step` events (with `title`, `activityId`; status `running/completed/failed/blocked/waiting`) and a final `run` event whose `result` holds the reply, workflow status, request cards and customer data |

Tests: `pytest` runs everything offline (scripted LLM, mocked Bootog). With
`RUN_LLM_TESTS=1` it also runs full conversations against the real chat model.

## Contact Us Agent — run workflow

```
initialize_run → select_provider → get_contact_us_data ─┬─ (no request selected) → END
                                                        └─ select_contact_request → prepare_customer_data
                                                           → create_customer → verify_customer → END
```

A failed or blocked step ends the run. Later steps never run.

| Endpoint | |
|---|---|
| `GET /api/v1/agents/contact-us/providers` | Companies for the session's company filter |
| `POST /api/v1/agents/contact-us/runs` | Start a run → `{runId, mode, steps}` (202) |
| `GET /api/v1/agents/runs/{runId}/events` | Server-Sent Events: `plan`, `step`, final `run` |

Body for `POST …/runs`: `{providerId?, providerName?, contactRequestId?, options}`.
Without `contactRequestId` the run retrieves requests. With it, the run creates a
customer from that request. `options` (the Agent Behavior settings) are validated
and stored, but no Phase 1 step acts on them.

### Customer creation runs in the chat agent

The run workflow cannot ask for missing details or a confirmation, so its
`create_customer` step ends the run as **blocked** after mapping the customer details,
and sends nothing. Customers, B2B clients and partners are created in the chat agent
(see Conversions above).
