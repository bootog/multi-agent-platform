"""Contact Us chat agent — LangGraph state machine.

One chat session = one checkpointed LangGraph thread; one user message = one graph
invocation (a "turn"). The turn ends when the agent has answered or needs something
from the user; the next message continues from the same saved state.

    START -> start_turn -> understand_message (LLM)
          -> [resolve_provider]                       company named in chat
          -> [search_providers]                       "find/list service providers"
          -> [resolve_assigned_user]                  "assigned to X"
          -> [get_contact_us_data]                    existing node (list / search all pages)
          -> [match_request]                          which request the user means
          -> [select_contact_request -> prepare_customer_data]   existing nodes
          -> [merge_customer_fields]                  values the user supplied
          -> conversion (Customer / B2B Client / Partner):
               start_conversion -> [choose_conversion_type] -> [resolve_target_provider]
               -> validate_customer_data ── missing/invalid -> ask_for_missing (waits)
               -> check_existing_account -> resolve_role -> [select_subcategories]
               -> confirm_conversion (waits for an explicit "yes")
               -> create_account -> verify_account -> [link_provider_vendor]
               -> [save_subcategories] -> update_request_status -> refresh_requests
          -> respond (LLM, from structured facts) -> END

`next_step` runs after every node and decides where to go from the LLM's
interpretation plus the structured state. The LLM never chooses an API call, and
nothing is created without a confirmation of exactly the data that is sent."""

import asyncio
import base64
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.contact_us import agent_nodes, conversion_nodes, nodes
from app.agents.contact_us.agent_nodes import (
    REFERENCE_KEYS,
    ChatEmitter,
    customer_view,
    listed_records,
    listed_this_turn,
    request_summary,
)
from app.agents.contact_us.conversion_nodes import active_type, conversion_fingerprint, conversion_of
from app.agents.contact_us.conversions import CONVERSION_OPERATIONS
from app.agents.contact_us.modules import agent_name, canonical_agent_key, module_filter
from app.agents.contact_us.state import ContactUsChatState
from app.agents.contact_us.understanding import ADVANCING_INTENTS, CONVERSION_INTENTS, REQUEST_INTENTS, WORKFLOW_INTENTS
from app.agents.runtime import RunChannel
from app.core.logging import get_logger
from app.integrations.bootog import BootogClient

logger = get_logger(__name__)

NodeFn = Callable[[ContactUsChatState, RunnableConfig], Awaitable[dict[str, Any]]]

NODES: dict[str, NodeFn] = {
    "understand_message": agent_nodes.understand_message,
    "resolve_provider": agent_nodes.resolve_provider,
    "search_providers": conversion_nodes.search_providers,
    "resolve_assigned_user": conversion_nodes.resolve_assigned_user,
    "get_contact_us_data": nodes.get_contact_us_data,
    "match_request": agent_nodes.match_request,
    "select_contact_request": nodes.select_contact_request,
    "prepare_customer_data": nodes.prepare_customer_data,
    "merge_customer_fields": agent_nodes.merge_customer_fields,
    "start_conversion": conversion_nodes.start_conversion,
    "choose_conversion_type": conversion_nodes.choose_conversion_type,
    "resolve_target_provider": conversion_nodes.resolve_target_provider,
    "validate_customer_data": agent_nodes.validate_customer_data,
    "ask_for_missing": agent_nodes.ask_for_missing,
    "check_existing_account": conversion_nodes.check_existing_account,
    "resolve_role": conversion_nodes.resolve_role,
    "select_subcategories": conversion_nodes.select_subcategories,
    "confirm_conversion": conversion_nodes.confirm_conversion,
    "create_account": conversion_nodes.create_account,
    "verify_account": conversion_nodes.verify_account,
    "link_provider_vendor": conversion_nodes.link_provider_vendor,
    "save_subcategories": conversion_nodes.save_subcategories,
    "update_request_status": conversion_nodes.update_request_status,
    "refresh_requests": conversion_nodes.refresh_requests,
}


def _recorded(name: str, fn: NodeFn) -> NodeFn:
    """Appends the node to `turn_steps`, which the router uses as its loop guard."""

    async def node(state: ContactUsChatState, config: RunnableConfig) -> dict[str, Any]:
        update = await fn(state, config)
        return {**update, "turn_steps": [*(state.get("turn_steps") or []), name]}

    return node


def next_step(state: ContactUsChatState) -> str:
    done = state.get("turn_steps") or []
    ran = done.count
    if state.get("execution_status") in ("failed", "blocked"):
        return "respond"
    if state.get("kickoff"):  # "Run Agent": load the latest requests and greet
        return "get_contact_us_data" if not ran("get_contact_us_data") else "respond"
    if not ran("understand_message"):
        return "understand_message"
    if state.get("company_name") and not ran("resolve_provider"):
        return "resolve_provider"

    intent = state.get("intent")
    if intent == "list_providers":
        return "search_providers" if not ran("search_providers") else "respond"
    if intent == "list_requests":
        filters = state.get("request_filters") or {}
        if filters.get("assigned_user_name"):
            if not ran("resolve_assigned_user"):
                return "resolve_assigned_user"
            if not filters.get("assigned_user_ids"):
                return "respond"  # unknown assignee: ask, never list everyone instead
        return "get_contact_us_data" if not ran("get_contact_us_data") else "respond"
    if intent not in WORKFLOW_INTENTS:
        return "respond"

    converting = state.get("operation") in CONVERSION_OPERATIONS
    ref = state.get("request_reference") or {}
    identifying = any(ref.get(k) for k in REFERENCE_KEYS)
    match = state.get("match_result") or {}

    # Which request? Retrieve first if nothing suitable is loaded.
    if identifying or (not state.get("contact_request") and (intent in REQUEST_INTENTS or converting)):
        if identifying and not ran("get_contact_us_data") and (
            not state.get("contact_requests") or state.get("request_filters")
        ):
            return "get_contact_us_data"
        if not ran("match_request"):
            return "match_request"
        if match.get("status") == "search":  # not loaded: search Bootog, then match again
            return "get_contact_us_data" if done[-1] == "match_request" else "match_request"
        if match.get("status") != "matched":
            # Ambiguous / not found / no reference: ask the user, keeping any details
            # they gave so they apply once the request is chosen.
            return "merge_customer_fields" if _unmerged(state, ran) else "respond"

    wanted = state.get("contact_request_id")
    if wanted and (state.get("contact_request") or {}).get("id") != wanted and not ran("select_contact_request"):
        return "select_contact_request"
    if (
        converting
        and state.get("contact_request")
        and not ran("prepare_customer_data")
        and (ran("select_contact_request") or not state.get("customer_payload"))
    ):
        return "prepare_customer_data"
    if _unmerged(state, ran):
        return "merge_customer_fields"

    # A conversion only advances when the user is working on it this turn.
    if not (converting and state.get("contact_request") and intent in ADVANCING_INTENTS):
        return "respond"
    return _conversion_step(state, ran)


def _conversion_step(state: ContactUsChatState, ran: Callable[[str], int]) -> str:
    if not ran("start_conversion"):
        return "start_conversion"
    conv = conversion_of(state)
    if conv.get("blocked"):
        return "respond"
    ctype = active_type(state)
    intent = state.get("intent")
    completed = conv.get("completed") or []

    if "create_account" in completed:
        # The account exists: only resume the remaining steps, never create again.
        if intent not in ("retry_operation", "confirm_action", *CONVERSION_INTENTS):
            return "respond"
        for step in ctype.steps():
            if step not in completed:
                return step if not ran(step) else "respond"
        if ran("update_request_status") and not ran("refresh_requests"):
            return "refresh_requests"
        return "respond"

    if ctype is None:
        return "choose_conversion_type" if not ran("choose_conversion_type") else "respond"
    turn = state.get("turn_conversion") or {}
    answered_provider = (state.get("answered_question") or {}).get("type") == "choose_provider" and turn.get("option_position")
    if ctype.needs_target_provider and (
        not conv.get("target_provider")
        or turn.get("target_provider_name")
        or conv.get("pending_provider_name")
        or answered_provider
    ):
        if not ran("resolve_target_provider"):
            return "resolve_target_provider"
        if not conv.get("target_provider"):
            return "respond"
    if not ran("validate_customer_data"):
        return "validate_customer_data"
    if state.get("missing_required_fields") or state.get("invalid_fields"):
        return "ask_for_missing" if not ran("ask_for_missing") else "respond"
    if not ran("check_existing_account"):
        # Every advancing turn: the email may have changed, or an earlier attempt's
        # account may have appeared (then it is adopted, never created again).
        return "check_existing_account"
    if not conv.get("role_id"):
        return "resolve_role" if not ran("resolve_role") else "respond"
    if ctype.needs_subcategories:
        wants_more = (
            turn.get("subcategory_names")
            or conv.get("pending_subcategory_names")
            or (
                (state.get("answered_question") or {}).get("type") == "choose_subcategory"
                and (turn.get("option_position") or turn.get("option_positions"))
            )
        )
        if (not conv.get("subcategories") or wants_more) and not ran("select_subcategories"):
            return "select_subcategories"
        if not conv.get("subcategories") or (state.get("pending_question") or {}).get("type") == "choose_subcategory":
            return "respond"

    current = conversion_fingerprint(state)
    failed = conv.get("failed") or {}
    if failed.get("step") == "create_account" and failed.get("kind") == "rejected" and failed.get("fingerprint") == current:
        # Bootog rejected exactly these details: resending them can only fail again (or
        # hit a half-created account). The user has to change something or cancel.
        return "respond"
    # Approval must answer a confirmation the user has SEEN: the question was pending
    # before this message (or this is a retry of a failed creation), it was not asked in
    # this same turn, and nothing changed since. A message misread as "yes" while it
    # answers another question can never create an account.
    answered_confirmation = (state.get("answered_question") or {}).get("type") == "confirm_conversion"
    retrying_creation = intent == "retry_operation" and failed.get("step") == "create_account"
    confirmed = (
        (intent == "confirm_action" and answered_confirmation or retrying_creation)
        and not ran("confirm_conversion")
        and conv.get("awaiting_fingerprint") == current
    )
    if confirmed:
        return "create_account" if not ran("create_account") else "respond"
    return "confirm_conversion" if not ran("confirm_conversion") else "respond"


def _unmerged(state: ContactUsChatState, ran: Callable[[str], int]) -> bool:
    has_updates = state.get("turn_field_updates") or state.get("turn_extra_information")
    return bool(has_updates) and not ran("merge_customer_fields")


def build_contact_us_chat_graph(checkpointer=None):
    graph = StateGraph(ContactUsChatState)
    graph.add_node("start_turn", agent_nodes.start_turn)
    for name, fn in NODES.items():
        graph.add_node(name, _recorded(name, fn))
    graph.add_node("respond", agent_nodes.respond)

    routes = [*NODES, "respond"]
    graph.add_edge(START, "start_turn")
    graph.add_conditional_edges("start_turn", next_step, routes)
    for name in NODES:
        graph.add_conditional_edges(name, next_step, routes)
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)


# In-memory checkpointer: sessions live as long as the process. Swap for a persistent
# LangGraph saver (e.g. Postgres) to keep conversations across restarts/instances.
CHECKPOINTER = InMemorySaver()
CONTACT_US_CHAT_GRAPH = build_contact_us_chat_graph(CHECKPOINTER)


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
        return self._owners.get(session_id)

    def agent_of(self, session_id: str) -> str | None:
        """Canonical key of the Contact agent the session was started for."""
        return self._agents.get(session_id)

    def begin(self, session_id: str) -> bool:
        if session_id in self._busy:
            return False
        self._busy.add(session_id)
        self._last_used[session_id] = time.monotonic()
        return True

    def end(self, session_id: str) -> None:
        self._busy.discard(session_id)
        self._last_used[session_id] = time.monotonic()

    async def delete(self, session_id: str) -> None:
        self._owners.pop(session_id, None)
        self._agents.pop(session_id, None)
        self._last_used.pop(session_id, None)
        await self._checkpointer.adelete_thread(session_id)

    async def _prune(self) -> None:
        now = time.monotonic()
        idle = [s for s, t in self._last_used.items() if s not in self._busy and now - t > self._ttl]
        oldest = sorted((t, s) for s, t in self._last_used.items() if s not in self._busy and s not in idle)
        idle += [s for _, s in oldest[: max(0, len(self._last_used) - len(idle) - self._max + 1)]]
        for session_id in idle:
            await self.delete(session_id)


sessions = ChatSessions(CHECKPOINTER)


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


_RUN_STATUS = {"failed": "failed", "blocked": "blocked", "waiting_for_user": "waiting"}


def chat_result(state: ContactUsChatState) -> dict[str, Any]:
    """`result` of the final `run` event: the reply plus what the UI shows with it."""
    match = state.get("match_result") or {}
    by_id = {r.get("id"): r for r in state.get("contact_requests") or []}
    requests, context = [], None
    if match.get("status") == "ambiguous":
        requests, context = [request_summary(by_id[i]) for i in match.get("candidate_ids", []) if i in by_id], "candidates"
    elif listed_this_turn(state):
        requests, context = [request_summary(r) for r in listed_records(state)[:25]], "listed"

    selected = state.get("contact_request")
    conv = conversion_of(state)
    show_customer = (
        state.get("operation") in CONVERSION_OPERATIONS
        or state.get("is_complete")
        or bool(conv.get("request_id") and selected and conv.get("request_id") == selected.get("id"))
        or state.get("intent") in ("provide_information", "show_request_details", *CONVERSION_OPERATIONS)
    )
    ctype = active_type(state)
    providers = (state.get("provider_results") or {}).get("providers") or (
        conv.get("provider_candidates") if (state.get("pending_question") or {}).get("type") == "choose_provider" else None
    )
    return {
        "sessionId": state.get("session_id"),
        "agentKey": state.get("agent_key"),
        "reply": state.get("agent_response"),
        "workflowStatus": state.get("workflow_status"),
        "intent": state.get("intent"),
        "operation": state.get("operation"),
        "conversionType": ctype.key if ctype else None,
        "awaitingUserInput": bool(state.get("awaiting_user_input")),
        "pendingQuestion": state.get("pending_question"),
        "providerName": state.get("provider_name"),
        "requests": requests,
        "requestsContext": context,
        "requestsTotal": state.get("contact_requests_total") if context == "listed" else None,
        "selectedRequest": request_summary(selected) if selected else None,
        "customer": customer_view(state) if show_customer else None,
        "providers": providers or [],
        "extraInformation": state.get("extra_information") or {},
    }


async def run_contact_us_chat_turn(
    channel: RunChannel,
    session_id: str,
    message: str | None,
    client: BootogClient,
    graph=None,
    llm=None,
    agent_key: str | None = None,
    agent_label: str | None = None,
) -> ContactUsChatState:
    """Run one turn, streaming activity into `channel`. Always closes the channel and client."""
    graph = graph or CONTACT_US_CHAT_GRAPH
    emitter = ChatEmitter(channel)
    configurable: dict[str, Any] = {"thread_id": session_id, "bootog_client": client, "emitter": emitter}
    if llm is not None:
        configurable["llm"] = llm
    config: RunnableConfig = {"configurable": configurable, "recursion_limit": 60}
    inputs: ContactUsChatState = {
        "run_id": channel.run_id,
        "session_id": session_id,
        "agent_key": canonical_agent_key(agent_key),
        "agent_name": agent_name(agent_key, agent_label),
        "module_filter": module_filter(agent_key),
        "kickoff": not message,
        "current_user_message": message,
        "conversation_history": [{"role": "user", "content": message}] if message else [],
    }
    try:
        state = await graph.ainvoke(inputs, config=config)
    except Exception as exc:  # graph-level crash: never report success
        logger.exception("[CONTACT_US_AGENT] session=%s run=%s crashed: %s", session_id, channel.run_id, type(exc).__name__)
        reply = "Something went wrong while I was working on that. Please try again."
        state = {**inputs, "workflow_status": "failed", "agent_response": reply}
        try:
            await graph.aupdate_state(
                config,
                {"conversation_history": [{"role": "assistant", "content": reply}], "workflow_status": "failed"},
                as_node="respond",
            )
        except Exception:
            logger.warning("[CONTACT_US_AGENT] session=%s could not record the failure", session_id)
    finally:
        await client.aclose()

    status = _RUN_STATUS.get(state.get("workflow_status") or "", "completed")
    await emitter.finish(status, state.get("agent_response") or "", chat_result(state))
    return state


async def run_turn_in_session(
    channel: RunChannel,
    session_id: str,
    message: str | None,
    client: BootogClient,
    agent_key: str | None = None,
    agent_label: str | None = None,
) -> None:
    try:
        await run_contact_us_chat_turn(channel, session_id, message, client, agent_key=agent_key, agent_label=agent_label)
    finally:
        sessions.end(session_id)


def start_turn_task(
    channel: RunChannel,
    session_id: str,
    message: str | None,
    client: BootogClient,
    agent_key: str | None = None,
    agent_label: str | None = None,
) -> asyncio.Task:
    return asyncio.create_task(run_turn_in_session(channel, session_id, message, client, agent_key, agent_label))
