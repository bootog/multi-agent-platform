# Frontend Order Enquiry Agent (package)

A LangGraph agent that answers questions about BOOTOG order enquiries, provider first:
"How many enquiries today?" → providers ranked by count; "Show me <provider>" → that
provider's enquiries; "Now only converted ones" → same provider and dates, converted.

Architecture, data flow, API contracts and limitations:
[docs/frontend-order-inquiry-architecture.md](../../../docs/frontend-order-inquiry-architecture.md).

## Run it

```python
from app.agents.frontorder_enquiry import run_frontorder_enquiry_agent

state = await run_frontorder_enquiry_agent(
    channel,            # app.agents.runtime.RunChannel — receives plan/step/run events
    client,             # BootogClient carrying the caller's own bearer token
    "How many enquiries came in today?",
    session_id=session_id,   # same id for every message of one conversation
    owner_id=owner_id,       # opaque caller id for the audit record
)
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `BOOTOG_API_BASE_URL` | `https://api.dev.bootog.com/api/v1` | Bootog API (shared setting) |
| `BOOTOG_TIMEOUT_SECONDS`, `BOOTOG_GET_RETRIES`, `BOOTOG_RETRY_BACKOFF_SECONDS` | `20`, `2`, `0.3` | HTTP behaviour (shared) |
| `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL` | –, `gpt-4.1-mini` | Chat model for understanding (shared) |
| `FRONTORDER_ENQUIRY_TIMEZONE` | `Asia/Kolkata` | Business timezone for "today" and dates — set it to the business's zone |
| `FRONTORDER_ENQUIRY_PAGE_SIZE` | `50` | Enquiries per page of a provider's list |
| `FRONTORDER_ENQUIRY_MAX_PAGES` | `5` | Page cap for follow-up checks (every record is read) |
| `FRONTORDER_ENQUIRY_SUMMARY_PAGE_SIZE` | `10` | Providers per page of the provider summary |
| `FRONTORDER_ENQUIRY_PROVIDER_MATCH_LIMIT` | `10` | Most providers offered for an ambiguous name |
| `FRONTORDER_ENQUIRY_MAX_RANGE_DAYS` | `366` | Longest date range one request may ask for |
| `DB_*`, `AGENT_DB_NAME` | – | Optional: run audit table `agent_run_audit` (shared) |

No provider names, ids or business data are configured anywhere: they come from Bootog.

## Moving it to another Python backend

1. Copy this folder and `app/api/routes/frontorder_enquiry.py` (adjust the import paths).
2. Python packages: `langgraph`, `langchain-core`, `langchain-openai` (or any chat model
   with structured output), `httpx`, `pydantic`, `tzdata` (Windows), `fastapi` for the route.
3. Provide the platform pieces it imports — each is small:
   - `app.integrations.bootog.client`: `BootogClient` (`get(path, params, stats=None)`,
     `aclose()`), `BootogApiError` (`message`, `status_code`, `timed_out`), `BootogAuth`.
   - `app.agents.runtime`: `RunChannel`, `RunEmitter` (`plan`, `step`, `finish`), `utc_now`.
   - `app.llm`: `get_llm()` and `LlmNotConfiguredError`.
   - `app.core.logging`: `get_logger`, `log_step`.
4. Replace `sessions.py` with your session registry (`create`, `owner_of`, `agent_of`,
   `begin`, `end`) and any LangGraph checkpointer. `owner_fingerprint` hashes the JWT
   subject; keep an equivalent so a refreshed token keeps its conversation.
5. Audit: pass `audit_sink=<async fn(record) -> bool>` to `run_frontorder_enquiry_agent`,
   or keep `graph.save_audit` and the `agent_run_audit` table from `app/history/store.py`.
6. Copy `tests/frontorder_enquiry/` — the tests use an in-memory fake of the Bootog API
   and a scripted model, so they run without network, database or API keys.
