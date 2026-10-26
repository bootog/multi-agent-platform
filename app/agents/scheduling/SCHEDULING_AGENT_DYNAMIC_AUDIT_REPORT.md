# Scheduling Agent — Dynamic Configuration Audit

Scope: `app/agents/scheduling/` and the modules it depends on. Date: 2026-10-26. Branch: `agenticV2`.

## A. Executive summary

**Overall assessment.** The Scheduling Agent is already mostly data-driven where it matters.
Order statuses, status ids, counts, service types, programs, providers, producers, skill sets,
credentials, working hours, blocked days and booked appointments all come from live Bootog API
calls. Every LLM-supplied reference is checked against that data before anything is written.
Its operational limits (page size, slot days, search radius, batch size) were already in
`app/core/config.py`. Most of the remaining literals are deliberate: they mirror Bootog API
contracts or the host UI's business rules, and they should stay fixed.

**Main hardcoding issues found**
1. LLM history windows (12 and 8 messages) were buried in `understanding.py`.
2. Availability limits (at most 14 days, 8 slots per day) and the order fetch page size (50,
   in two places) were literals in `nodes.py`.
3. Intent names and enum values were each written out twice. `Intent` repeated
   `INTENT_LABELS`. The log channel and direction Literals repeated `CONTACT_TYPES` and
   `PERSON_TYPES`. `reference_kind` repeated `_REFERENCE_TITLES`.
4. Nothing checked that the routing tables (`INTENT_TOOLS`, `_ACTION_INTENTS`, `TOOLS`,
   `INTENT_LABELS`) agreed with each other.
5. The UNDERSTAND prompt lists the intents in a third place (deferred, see F).

**Improvements implemented (12).** Five new settings with defaults equal to the old values.
Intent and enum Literals are now derived from one table each. Duplicated literals are
centralised. Routing consistency is checked when the module is imported. A guard stops a zero
history window from sending the whole conversation.

**Important limitations**
- At first the scheduling test sources were missing from the working tree; only compiled `.pyc`
  files existed. They have since been **recovered** from unreachable git blobs, verified against
  that bytecode, and restored to `tests/` (see E). They are not committed yet.
- No live Bootog or LLM calls were made.

## B. Hardcoding inventory

Classification: **A** = safe to make dynamic, **B** = possible with safeguards, **C** = must stay fixed.

### B.1 Implemented

| # | File · location | Current (before) | Class | Approach | Action | Reason / risk |
|---|---|---|---|---|---|---|
| 1 | `understanding.py` · `understand()` | `_history(history, 12)` | A | `SCHEDULING_UNDERSTAND_HISTORY_MESSAGES` (default 12) | Implemented | Operational token/context budget. Default unchanged. Risk: none at default |
| 2 | `understanding.py` · `compose_reply()` | `_history(history, 8)` | A | `SCHEDULING_REPLY_HISTORY_MESSAGES` (default 8) | Implemented | Same as #1 |
| 3 | `understanding.py` · `_history()` | `messages[-limit:]` | A | `limit = max(1, limit)` | Implemented (safeguard) | With a configurable limit, `0` would slice `[-0:]` and send the **whole** history. Unreachable at the old literals |
| 4 | `understanding.py` · `Intent` | 16-value `Literal[...]` repeating `INTENT_LABELS` keys | B | `Intent = Literal[tuple(INTENT_LABELS)]`, the pattern the file already used for `ResponseResult` | Implemented | Same values in the same order. The JSON schema sent to the LLM is byte-identical (verified) |
| 5 | `understanding.py` · `LogDetails.channel` | `Literal["phone_call","sms","email","note"]` | B | `domain.LOG_CHANNELS` (= `CONTACT_TYPES` minus `email_and_sms`) | Implemented | Identical values and order. `missing_log_fields` now uses the same list (equivalent check) |
| 6 | `understanding.py` · `LogDetails.direction` | `Literal["inbound","outbound"]` | B | From `domain.PERSON_TYPES` keys | Implemented | Identical |
| 7 | `understanding.py` · `TurnUnderstanding.reference_kind` / `nodes.py` · `_REFERENCE_TITLES` | Kinds written twice | B | Single `domain.REFERENCE_TITLES`. The Literal is derived from it | Implemented | Identical values and titles. `_REFERENCE_TITLES` was private |
| 8 | `nodes.py` · `_window()` | Cap `14` days | A | `SCHEDULING_MAX_SLOT_DAYS` (default 14) | Implemented | Existing `max(1, …)` floor kept, so a cap ≤ 0 still yields 1 day |
| 9 | `nodes.py` · `_availability()` → `domain.available_slots(per_day_limit=8)` | Implicit default `8` | A | `SCHEDULING_SLOTS_PER_DAY` (default 8), floored at 1 | Implemented | `domain.available_slots` keeps its own default of 8 for other callers |
| 10 | `nodes.py` · `_all_orders()`, `_orders_created_between()` | `batch = 50` (twice) | A | `SCHEDULING_ORDER_FETCH_PAGE_SIZE` (default 50) via `_fetch_page_size()`, floored at 1 | Implemented | Paging, dedupe and truncation logic unchanged. Risk: Bootog may cap large `pageSize` values, but you only hit that by opting in |
| 11 | `nodes.py` · `_producers_for()`, `find_producers()` | `items[:25]` (twice) | A | Module constant `_PRODUCERS_KEPT = 25` | Implemented | Same value in one place. Kept as a constant, not a setting, because it is the list the user picks "#n" from |
| 12 | `graph.py` · `INTENT_TOOLS` / `nodes._ACTION_INTENTS` / `TOOLS` | Independent tables, no check | B | `_check_routing()` raises `RuntimeError` at import on drift | Implemented | Fails fast instead of a runtime `KeyError` in `INTENT_LABELS[intent]`. Uses `raise`, not `assert`, so it is not stripped by `-O` |

### B.2 Intentionally retained or deferred

| # | File · location | Current | Class | Action | Reason |
|---|---|---|---|---|---|
| 13 | `graph.py` · `AGENT_KEY = "scheduling"` | Literal | C | Retained | Key stored in the chat-history DB and used by session ownership checks and the `/agents/scheduling` routes |
| 14 | `graph.py` · `recursion_limit: 25` | Literal | A | Retained | A turn visits ≤ 4 nodes. This is a crash guard, not tuning, and making it configurable adds no value |
| 15 | `graph.py` · `_RUN_STATUS`, `_order_card` / `_producer_card` / `_ranking_card` field lists, `[:15]` | Literals | C | Retained | Response contract consumed by the Angular host |
| 16 | `nodes.py` · `ACTIVITY_TITLES` and step ids | Literals | C | Retained | Live Activity step ids are a frontend contract. Titles are UI copy |
| 17 | `nodes.py` · `CAPABILITIES` | List of strings | A | Deferred (F) | Prompt-like text. Moving it is cosmetic |
| 18 | `nodes.py` · `_WRITE_TYPES`, `execute_action` guards (`askedRunId`, action id match) | Literals | C | Retained | Security invariant: only a shown and confirmed action of a known type can write |
| 19 | `nodes.py` · `_ACTION_INTENTS` | Card type → intent | C | Retained (now checked by #12) | Mirrors `ChatAction.type` Literal in `app/api/routes/scheduling.py` (API contract, outside scope) |
| 20 | `nodes.py` · `_distance_limit()` clamp `1..500` | Literal | C | Retained | Mirrors `ChatAction.max_distance_miles le=500` in the API schema. Changing one without the other breaks parity |
| 21 | `nodes.py` · `_context_for_llm` caps `[:60]`, `[:10]`, `[:15]`; `build_facts` caps `[:5]`, `[:3]`, `[:8]`; reference `[:200]` | Literals | A | Deferred (F) | Token-budget and presentation caps. Changing them changes what the LLM sees, so they need prompt evaluation first |
| 22 | `nodes.py` · `_search_orders` `page_size=10` | Literal | A | Retained | Lookup bound for number/customer search. It is a different concept from `SCHEDULING_PAGE_SIZE`, so the two were not merged |
| 23 | `nodes.py` · input truncations `[:100]`, `[:500]`, `[:60]` | Literals | C | Retained | Input sanitisation before values reach Bootog |
| 24 | `nodes.py` · `list_reference` "Which list do you need: …" text | Literal | A | Retained | Generating it from `REFERENCE_TITLES` would change the user-facing wording ("Order sub-statuses") |
| 25 | `nodes.py` · `prepare_log` defaults (`direction="outbound"`, `caller_type="policyholder"/"other"`, party `customer`) | Literals | B | Retained | Business defaults that mirror the host Add Log popup |
| 26 | `nodes.py` · `_assign_one` halt on 401/403 | Literal | C | Retained | Security: stop the batch when Bootog refuses access |
| 27 | `nodes.py` · error, notice and `fallback_reply` texts | Literals | A | Retained | User-facing copy. No i18n system exists to plug into |
| 28 | `domain.py` · `DASHBOARD_GROUPS`, status aliases in `match_status_cards` | Literals | B | Deferred (F) | Mirrors the Orders page grouping. `GetOrderHeaders` returns names and counts but not groups. Unknown cards already fall into "Other" dynamically |
| 29 | `domain.py` · `ASSIGNABLE_STATUSES`, `SCHEDULABLE_STATUSES`, `_CLOSED_ORDER_STATUSES` | Literals | C | Retained | Business rules copied from the host (`assignOrderPopup`). No API exposes them |
| 30 | `domain.py` · `SLOT_STEP_MINUTES = 60`, duration fallback 60 | Literals | C | Retained | Must match the host scheduling calendar grid, or offered slots would differ from the UI |
| 31 | `domain.py` · `CONTACT_TYPES`, `CALLER_TYPES`, `PERSON_TYPES`, `RESPONSE_RESULTS`, `VISIBLE_TO_ALL` | Code maps | C | Retained | Bootog enum codes (API contract) |
| 32 | `domain.py` · `log_payload` fields and formats (`%m/%d/%Y`, unpadded `h:mm AM`), constant flags | Literals | C | Retained | `POST ClientContactLog` contract. The time format deliberately differs from `format_clock` (zero-padded) |
| 33 | `domain.py` · `assign_payload` / `schedule_payload` (`scheduleDate = "<day>T00:00:00.000Z"`, `isPackage`) | Literals | C | Retained | `POST Order/AssignOrSchedule` contract |
| 34 | `domain.py` · `MATCHED_GROUP`, ranking rules, excluded-reason texts, tie-breaks | Literals | C | Retained | Verified semantics of Bootog's "matched" group. Business logic |
| 35 | `domain.py` · `UUID_RE`, `_strip_html(limit=300)` | Literals | C | Retained | Validation and sanitisation |
| 36 | `domain.py` · `LocalClock` (browser UTC offset, default 0) | Behaviour | C | Retained | Timezone semantics unchanged by requirement |
| 37 | `integrations/bootog/scheduling.py` · `PRODUCER_GROUPS`, `CALENDAR_FROM`, `GetOrderUsers` params, contact-log page sizes 20/5 | Literals | C | Retained | API contract, and shared integration module (out of scope) |
| 38 | `app/prompts/scheduling.py` · intent list inside `UNDERSTAND_SYSTEM_PROMPT` | Text | B | Deferred (F) | Third copy of the intent names. Generating it would change the prompt text and therefore LLM behaviour |
| 39 | `app/api/routes/scheduling.py` · `ChatAction` Literals (`type`, `channel`) | Literals | C | Retained | Public request schema (out of scope) |
| 40 | `state.py` · `append_history` cap 40 (`contact_us.state._HISTORY_LIMIT`) | Shared literal | C | Retained | Shared with the Contact agents. Changing it is out of scope |
| 41 | `understanding.py` · `_history` / `_json` / `_child` duplicated from `contact_us/understanding.py` | Code duplication | B | Retained | Extracting them would modify the Contact Us agent |
| 42 | LLM provider and model | — | — | Already dynamic | `OPENAI_CHAT_MODEL`, `OPENAI_TEMPERATURE`, `OPENAI_TIMEOUT_SECONDS` via `app.llm.get_llm()` |
| 43 | Bootog base URL, timeout, retries, max pages | — | — | Already dynamic | `BOOTOG_*` settings |
| 44 | Page size, slot days, radius, radius step and max, batch cap | — | — | Already dynamic | Existing `SCHEDULING_*` settings |

## C. Changes implemented

**Files modified**

| File | Change |
|---|---|
| `app/core/config.py` (shared) | +5 fields on `Settings` and +5 `os.getenv` reads, next to the existing `scheduling_*` keys. Additive only |
| `app/agents/scheduling/understanding.py` | Derived Literals (#4–#7). Configurable history windows (#1, #2) with a ≥ 1 guard (#3) |
| `app/agents/scheduling/domain.py` | Added `REFERENCE_TITLES` and `LOG_CHANNELS`. `missing_log_fields` uses `LOG_CHANNELS` |
| `app/agents/scheduling/nodes.py` | Settings used for slot cap, slots per day and fetch page size. `_PRODUCERS_KEPT`. Reference titles from `domain` |
| `app/agents/scheduling/graph.py` | `_check_routing()` at import |

**Configuration keys added** (environment variable → `Settings` field, default = previous literal)

| Env var | Field | Default | Effect |
|---|---|---|---|
| `SCHEDULING_MAX_SLOT_DAYS` | `scheduling_max_slot_days` | `14` | Most days of slots one request may show (floored at 1) |
| `SCHEDULING_SLOTS_PER_DAY` | `scheduling_slots_per_day` | `8` | Open slots listed per day (floored at 1) |
| `SCHEDULING_ORDER_FETCH_PAGE_SIZE` | `scheduling_order_fetch_page_size` | `50` | `pageSize` per `GetOrderDetails` call when listing every match (floored at 1) |
| `SCHEDULING_UNDERSTAND_HISTORY_MESSAGES` | `scheduling_understand_history_messages` | `12` | Chat messages sent to the understanding LLM call (floored at 1) |
| `SCHEDULING_REPLY_HISTORY_MESSAGES` | `scheduling_reply_history_messages` | `8` | Chat messages sent to the reply LLM call (floored at 1) |

Overrides follow the existing pattern: a real environment variable wins over `.env`, read once
by the `lru_cache`d `get_settings()`. A non-integer value fails at startup with `ValueError`,
exactly like the existing `SCHEDULING_*` keys. Zero or negative values are clamped where they
are used.

**Why behaviour is unchanged.** Every default equals the literal it replaced. Every derived
Literal has the same members in the same order. The `TurnUnderstanding` JSON schema given to
`with_structured_output` was snapshotted before the change and compared after: it is
byte-identical. `INTENT_TOOLS`, `TOOLS`, graph edges, state keys, API payloads and public
names are untouched.

**Change outside the agent directory.** `app/core/config.py` is the project's single settings
system, and it already holds the Scheduling keys. Adding fields there reuses the existing
pattern, as the task requires, instead of creating a second config mechanism. The fields have
defaults, so no other module's behaviour changes. `app.main` imports cleanly.

## D. Items intentionally retained

- **API contracts** (#31–#33, #37, #39). These are field names, enum codes, date and time
  formats and query parameters that Bootog or the Angular host parse. Making them dynamic
  could only break integrations.
- **Business rules mirrored from the host UI** (#25, #28–#30, #34). The assignable,
  schedulable and closed statuses, the 60-minute slot grid, producer ranking and log defaults
  must match what staff see in the Orders page. No Bootog endpoint provides them. Replacing
  them with LLM output is prohibited.
- **Security and safety** (#18, #23, #26, #35). The write allow-list, confirm-after-seen and
  stale-action checks, input truncation, UUID validation and halting on 401/403 all stay fixed.
- **Graph and state structure** (#13–#16). `AGENT_KEY`, node names, state keys and the result
  card shape are contracts with the history DB and the frontend.
- **Shared code** (#40, #41). Changes there would also modify the Contact Us agents.

## E. Testing results

**Baseline (before any change).** `tests/` contains no committed test sources. The Scheduling
tests exist only as Python 3.12 bytecode (`tests/__pycache__/test_{scheduling_agent,
assign_agent,batch_assign,radius_expansion}.cpython-312-pytest-9.1.1.pyc`, dated 20–22 Oct).
I copied them into a scratch `tests` package as sourceless modules and collected them through
one-line wrappers (`from tests.test_x import *`), using the project venv:

```
PYTHONPATH="<scratch>;<repo>" venv/Scripts/python.exe -m pytest -q -p no:cacheprovider \
  --rootdir <scratch> -o python_files="run_*.py" <scratch>
```

Result: **59 passed**. They cover intent → tool routing, dashboard and status mapping, order
listing and paging, local-date filters, assign (single, reassign, double-confirm, unverified,
server error), batch assign, radius expansion, schedule with recheck, add-log collection, stale
confirmations and reference lists.

**After the change**

| Check | Result |
|---|---|
| Same 59 compiled tests | **59 passed** |
| 20 new verification tests, now permanent in `tests/test_scheduling_config.py` | **20 passed** (79 total) |
| `TurnUnderstanding.model_json_schema()` before vs after | Byte-identical (`cmp`) |
| `import app.main` (startup imports, including `_check_routing`) | OK |
| `py_compile` of every changed file | OK |
| `pytest tests` (committed tests) | 0 tests exist |
| `git diff` review | Only the 5 files above. No unrelated edits |

The new verification tests check:
- defaults equal the old literals, in both the dataclass and the env parsing
- env overrides are read
- a non-integer value raises at startup
- a history window of 0 or a negative number sends only the latest message
- `understand` and `compose_reply` send exactly the configured number of messages (fake LLM)
- the slot-day cap works at its default, when overridden and at 0
- slots per day works at 8, 3 and 0 (0 is clamped to 1)
- the fetch page size works at 50, 7 and 0 (0 is clamped to 1)
- all 16 intents route to the same node as before, and blocked or unknown intents go to `respond`
- every card action maps to its intent
- `_check_routing` rejects an unknown intent and an unknown tool node
- the LLM model rejects an invented intent and the `email_and_sms` channel
- `missing_log_fields` behaves the same
- the Literal members equal the old hardcoded lists, in the same order

**Permanent tests (follow-up).** The 20 checks now live in `tests/test_scheduling_config.py`.
They use only `app.*` modules and in-memory fakes. Their fixture removes every `SCHEDULING_*`
environment variable for each test, so a local `.env` can't change the results. The repository's
normal command (`python -m pytest`) collects and passes all 20, including when conflicting
`SCHEDULING_*` values are set in the environment.

**Recovery of the original 59 tests (follow-up).** What was searched:
- all local and remote branches (remote heads equal the local refs)
- the stash (empty)
- every commit reachable from any ref or the reflog (none ever contained these files)
- VS Code Local History (no entries)
- `~/.claude` and `C:\AI_agents` (no other copies)

`git fsck --unreachable` found 5 dangling blobs. These were files that had been staged and then
unstaged without a commit: the four test modules and `tests/live/assign_live_check.py`. To check
each blob against its `.pyc`, I recompiled it as pytest does (assertion rewriting, Python 3.12,
the original `co_filename`) and compared the results:

| Source | Code-object equality (recursive, all fields incl. line tables) | Serialized size | Normalised marshal bytes |
|---|---|---|---|
| `tests/test_scheduling_agent.py` | equal | equal | identical |
| `tests/test_assign_agent.py` | equal | equal | identical |
| `tests/test_batch_assign.py` | equal | equal | identical |
| `tests/test_radius_expansion.py` | equal | equal | identical |
| `tests/live/assign_live_check.py` | equal | equal | raw bytes identical |

The raw `.pyc` bytes of the four pytest modules differ only in marshal's back-reference
encoding. That encoding depends on object identity when the file is written: after re-serializing
both, they are identical. Comments are not compiled, so the bytecode can't vouch for them, but
the files are the exact staged contents (`git hash-object` equals each blob id). All five were
restored. `tests/live/__init__.py` was not restored. Its `.pyc` holds no code and no docstring,
so its source (comments or blank at most) can't be recovered. `tests.live` still imports as a
namespace package without it.

With the sources restored, `python -m pytest` runs **79 passed, 0 failed, 0 skipped**: 16
scheduling agent, 16 assign, 18 batch assign, 9 radius expansion and 20 config. All original
tests are now reproducible from the working tree, once committed. The live check is not part of
the suite: it isn't named `test_*`, and it needs a real Bootog token.

**Not verified**
- Live Bootog calls. There are no credentials in the session, and production writes must be avoided.
- Real LLM interpretation quality.
- The Angular frontend.
- Whether Bootog accepts `pageSize` values above 50 for `GetOrderDetails` (relevant only if
  `SCHEDULING_ORDER_FETCH_PAGE_SIZE` is raised). The Contact Us agent sends `page_size=200` to a
  different (providers) endpoint, which says nothing about this one.
- The recovered tests match the 20–22 Oct bytecode. Any edits made after that, and never compiled
  or staged, could not be recovered.

## F. Remaining opportunities (need approval)

1. **Commit the recovered scheduling test sources** (restored to `tests/`, still untracked).
   Until they are committed, a branch switch or clean could lose them again.
2. **Generate the intent section of `UNDERSTAND_SYSTEM_PROMPT` from a registry** (name, label,
   prompt guidance, tool). This would make the intent list fully single-source, but it changes
   the prompt text, so it needs an LLM evaluation run first.
3. **Make the LLM context and fact caps configurable** (#21) after measuring their token and
   quality impact.
4. **Dashboard groups from a backend source** (#28) if Bootog adds group metadata to `GetOrderHeaders`.
5. **Share `ChatAction` types between the API route and `_ACTION_INTENTS`** (#19, #39). This
   needs an API-layer change.
6. **Extract the shared LLM helpers** (`_history`, `_json`, `_child`) into a common module (#41).
   This touches Contact Us.
7. ~~Document the `SCHEDULING_*` variables in `README.md`~~. **Done:** all eleven are now in the
   settings table, with per-key out-of-range behaviour and the interactions below.
8. **Add range validation in `get_settings()`** (e.g. reject a negative radius). This would make
   some currently tolerated configurations fail at startup, so it is a behaviour change. Today the
   only validation for every `SCHEDULING_*` key is `int()` parsing: a non-integer, including a
   decimal, fails at startup. Out-of-range handling happens where each value is used, and it
   differs per key (third column). Recommended ranges were checked against every consumer in
   `nodes.py` and `understanding.py`:

   | Key (default) | Recommended | Handling of out-of-range values today | Why |
   |---|---|---|---|
   | `SCHEDULING_PAGE_SIZE` (10) | 1–50 | ≤ 0 → `pageSize` = number of orders shown. Not clamped: on an empty list `pageSize` is reported as the raw value (0 → "1 page" to the LLM, negative → "0 pages") | Only the `pageSize` reported with a list and the LLM's page count. It no longer limits orders read |
   | `SCHEDULING_SLOT_DAYS` (5) | 1 – `SCHEDULING_MAX_SLOT_DAYS` | ≤ 0 → 1. Above the cap → cap | Default window when the user doesn't say how many days |
   | `SCHEDULING_MAX_SLOT_DAYS` (14) | ≥ `SCHEDULING_SLOT_DAYS`, ≤ 31 | ≤ 0 → 1 | Below `SCHEDULING_SLOT_DAYS` it silently shortens the default window. 31 is a judgement, not a hard limit: the range is one `GetOrderServiceDetails` query and one card block per day |
   | `SCHEDULING_SLOTS_PER_DAY` (8) | 1–24 | ≤ 0 → 1 | 60-minute grid inside working hours: a day can't hold more than 24. The LLM sees only the first 15 slots overall; the card shows all |
   | `SCHEDULING_PRODUCER_RADIUS_MILES` (30) | 1–500 | Clamped to 1–500 | Matches the API's `maxDistanceMiles ≤ 500` card field |
   | `SCHEDULING_RADIUS_STEP_MILES` (30) | > 0 and < `SCHEDULING_MAX_RADIUS_MILES` | ≤ 0 → no expansion (one search at the start radius) | Each step is one more `GetOrderUsers` call per order still unmatched in a batch |
   | `SCHEDULING_MAX_RADIUS_MILES` (120) | > `SCHEDULING_PRODUCER_RADIUS_MILES`, ≤ 500 | ≤ start radius → no expansion. **Not clamped** | Above 500 it exceeds what a user can request through the card (`≤ 500`). Bootog's own upper limit is unverified |
   | `SCHEDULING_MAX_BATCH_ORDERS` (50) | 1 – `SCHEDULING_ORDER_FETCH_PAGE_SIZE` × `BOOTOG_MAX_PAGES` | **Not clamped**: ≤ 0 refuses every multi-order assignment | Bounds the writes one confirmation triggers. A batch can't exceed the readable list anyway, because truncated lists are refused |
   | `SCHEDULING_ORDER_FETCH_PAGE_SIZE` (50) | 50–100 | ≤ 0 → 1 | Orders readable = this × `BOOTOG_MAX_PAGES` (500 by default), and longer lists are marked truncated. `BOOTOG_MAX_PAGES` is shared with Contact Us, so tune this key instead. Bootog's maximum `pageSize` for `GetOrderDetails` is **unverified** |
   | `SCHEDULING_UNDERSTAND_HISTORY_MESSAGES` (12) | 2–40 | ≤ 0 → 1 | Stored history is capped at 40 (`contact_us.state._HISTORY_LIMIT`, also on restore). Below 2, the model can't see the question a "yes" answers |
   | `SCHEDULING_REPLY_HISTORY_MESSAGES` (8) | 1–40 | ≤ 0 → 1 | Same 40-message cap |

   A low-risk first step would be a startup **warning** (not an error) when a value is outside
   these ranges. It is not implemented and still needs sign-off.

## G. Final change summary

| Item | Count |
|---|---|
| Files inspected in full | 12: `scheduling/{__init__,graph,state,understanding,domain,nodes}.py`, `core/config.py`, `prompts/scheduling.py`, `integrations/bootog/scheduling.py`, `api/routes/scheduling.py`, `tests/conftest.py`, `README.md` (config section). Shared helpers in `contact_us/{state,history,understanding}.py` were checked for dependencies |
| Findings | 44 (12 implemented, 3 already dynamic, 29 retained or deferred) |
| Files modified | 5 (4 in the agent, plus `app/core/config.py`). Follow-ups: `README.md` (docs), new `tests/test_scheduling_config.py`, and 5 recovered files restored (`tests/test_{scheduling_agent,assign_agent,batch_assign,radius_expansion}.py`, `tests/live/assign_live_check.py`) |
| Changes outside the agent | `app/core/config.py`: additive settings in the existing pattern (see C). `README.md`: rows for all eleven `SCHEDULING_*` settings, plus notes on validation and interactions. `tests/`: one new test file and five restored originals |
| API, DB, frontend, other agents | No changes |

**Known risks and follow-ups.** Raising `SCHEDULING_ORDER_FETCH_PAGE_SIZE` beyond what Bootog
supports could return short pages. The paging loop handles that safely, but it would read more
pages. `_check_routing()` makes a mis-edited intent table fail at startup rather than at
request time; that is intended. The main follow-up is to commit the recovered test sources.
