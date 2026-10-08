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
| `CONTACT_US_PAGE_SIZE` | `50` | Contact Us requests read per run (newest first) |
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

Auth: the caller's `Authorization: Bearer …`, `X-Tenant-Id` and `X-Role-Id` headers
are forwarded to Bootog, so the agent acts with the signed-in user's access. Tokens
are never logged.

## Layout

```
app/
  main.py                        FastAPI app, CORS, routers
  core/config.py, logging.py     settings from env; step logging (ids/status/timing only)
  integrations/bootog/           the only code that calls Bootog
    client.py                    BootogClient: auth headers, timeouts, safe errors
    providers.py                 GET ProviderUserMapping/GetProviderUserMappingBySession
    contact_us.py                GET ContactUs (documented filter + targetProviderId)
    customers.py                 customer creation — NOT CONFIGURED (see below)
  agents/
    runtime.py                   run registry + per-run event channel (SSE source)
    contact_us/
      state.py                   ContactUsAgentState
      schemas.py                 request/response models, customer draft mapping
      tools.py                   LangChain tools over the Bootog APIs
      nodes.py                   graph nodes; each emits running/completed/failed/blocked
      graph.py                   the LangGraph workflow + run executor
      understanding.py           LLM decision layer (typed interpretation, replies)
      agent_nodes.py             chat nodes around the existing ones; activity emitter
      agent_graph.py             chat state machine, sessions, turn executor
  llm/openapi.py                 get_llm(): the shared chat model
  prompts/contact_us.py          chat agent prompts
  api/routes/contact_us.py       HTTP endpoints
```

## Contact Us Agent — chat

The Contact Us page is a chat workspace. Each user message is one LangGraph turn over
a checkpointed session (`agents/contact_us/agent_graph.py`):

```
start_turn → understand_message (LLM) → [resolve_provider] → [get_contact_us_data]
  → [match_request] → [select_contact_request → prepare_customer_data]
  → [merge_customer_fields] → [validate_customer_data]
       ├─ missing/invalid → ask_for_missing  (turn ends, waiting for the user)
       └─ complete → create_customer → verify_customer
  → respond (LLM) → END
```

- **LLM** (`understanding.py`, prompts in `app/prompts/contact_us.py`): turns the message
  into a typed `TurnUnderstanding` (intent, goal, request reference, date/company
  filters, customer field values, extra notes) and writes the reply from structured facts.
- **Router** (`next_step`): deterministic; decides the next node from that interpretation
  plus state. The LLM never chooses or builds an API call.
- **Nodes**: the existing workflow nodes do all Bootog work unchanged.
- **Memory**: selected request, user-supplied fields (latest value wins, empty values
  ignored), extra notes and history persist per session. Extra notes are never sent to
  Bootog; the customer payload only ever has `CUSTOMER_FIELDS` keys.
- **Sessions** live in memory (`InMemorySaver`); swap for a persistent LangGraph saver to
  survive restarts. A session belongs to the caller that created it (JWT `sub`).

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

### Customer creation is not configured

`ContactUS.pdf` does not document a customer-creation endpoint or payload. Its only
customer entry is `UserDetail/GetCustomerDetails`, with no parameters or response.
So `create_customer` ends the run as **blocked** with "Customer creation API not
yet configured", after the customer details have been mapped from the request.
Nothing is sent. To enable it, implement `CustomerApi.create` in
`integrations/bootog/customers.py` against the documented endpoint and map
`CustomerDraft` to its payload fields.
