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
  api/routes/contact_us.py       HTTP endpoints
```

## Contact Us Agent

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
