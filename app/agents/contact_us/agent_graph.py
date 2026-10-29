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
               -> resolve_role -> check_existing_account -> [select_subcategories]
               -> confirm_conversion (waits for an explicit "yes")
               -> create_account -> verify_account -> [link_provider_vendor]
               -> [save_subcategories] -> update_request_status -> refresh_requests
          -> respond (LLM, from structured facts) -> END

`next_step` runs after every node and decides where to go from the LLM's
interpretation plus the structured state. The LLM never chooses an API call, and
nothing is created without a confirmation of exactly the data that is sent.

This is a separate graph from the one-shot run workflow (graph.py, POST /contact-us/runs)
because it is checkpointed: CHECKPOINTER saves the state under the session's thread_id
after every turn, and the next message continues from it. The run workflow has no memory
and needs none. Both graphs run the same step functions from nodes.py."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.contact_us import agent_nodes, conversion_nodes, nodes, request_actions
from app.agents.contact_us.agent_nodes import (
    REFERENCE_KEYS,
    ChatEmitter,
    customer_view,
    listed_records,
    listed_this_turn,
    provider_catalog_stale,
    request_filters_view,
    request_summary,
)
from app.agents.contact_us.conversion_nodes import active_type, conversion_fingerprint, conversion_of, conversion_steps
from app.agents.contact_us.conversions import CONVERSION_OPERATIONS
from app.agents.contact_us.modules import agent_name, canonical_agent_key, module_filter
from app.agents.contact_us.request_actions import status_change_view
from app.agents.contact_us.state import ContactUsChatState
from app.agents.contact_us.understanding import (
    ADVANCING_INTENTS,
    CONVERSION_INTENTS,
    REQUEST_ACTION_INTENTS,
    REQUEST_INTENTS,
    WORKFLOW_INTENTS,
)
# ChatSessions and owner_fingerprint are shared by all chat agents (app/agents/runtime.py).
from app.agents.runtime import ChatSessions, RunChannel, owner_fingerprint
from app.core.logging import get_logger
from app.integrations.bootog import BootogClient

logger = get_logger(__name__)

NodeFn = Callable[[ContactUsChatState, RunnableConfig], Awaitable[dict[str, Any]]]

NODES: dict[str, NodeFn] = {
    "understand_message": agent_nodes.understand_message,
    "resolve_provider": agent_nodes.resolve_provider,
    "search_providers": conversion_nodes.search_providers,
    "resolve_assigned_user": conversion_nodes.resolve_assigned_user,
    "load_filter_options": agent_nodes.load_filter_options,
    "get_tracking_history": request_actions.get_tracking_history,
    "prepare_status_change": request_actions.prepare_status_change,
    "change_request_status": request_actions.change_request_status,
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
    "present_details": conversion_nodes.present_details,
    "ask_edit_fields": conversion_nodes.ask_edit_fields,
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
    """The chat graph's router: which node runs next in this turn.

    Runs after every node. It decides from the LLM's interpretation (`intent`) and the
    structured state, and checks whether a node already ran this turn so no node loops.
    Returns a node name, or "respond" to end the turn with a reply.
    """
    done = state.get("turn_steps") or []
    ran = done.count  # ran("x") = how many times node "x" already ran in this turn
    if state.get("execution_status") in ("failed", "blocked"):
        return "respond"
    if state.get("kickoff"):  # "Run Agent": load the latest requests and greet
        return "get_contact_us_data" if not ran("get_contact_us_data") else _filter_options(state, ran)
    if not ran("understand_message"):
        return "understand_message"
    if (state.get("company_name") or state.get("company_pick")) and not ran("resolve_provider"):
        return "resolve_provider"

    if state.get("status_change_confirmed"):
        # The pending status change, confirmed this turn (change_request_status checks it
        # again), then the changed request's history.
        if not ran("change_request_status"):
            return "change_request_status"
        change = state.get("status_change") or {}
        if len(change.get("changed") or []) == 1 and not ran("get_tracking_history"):
            return "get_tracking_history"
        return "respond"

    intent = state.get("intent")
    if intent == "list_providers":
        return "search_providers" if not ran("search_providers") else "respond"
    if intent == "list_requests":
        if state.get("filter_problems"):
            return "respond"  # a filter could not be applied (or the company is unclear): ask, list nothing
        filters = state.get("request_filters") or {}
        if filters.get("assigned_user_name") or filters.get("assigned_to_me"):
            if not ran("resolve_assigned_user"):
                return "resolve_assigned_user"
            if not filters.get("assigned_user_ids"):
                return "respond"  # unknown assignee: ask, never list everyone instead
        return "get_contact_us_data" if not ran("get_contact_us_data") else _filter_options(state, ran)
    if intent not in WORKFLOW_INTENTS:
        return "respond"

    converting = state.get("operation") in CONVERSION_OPERATIONS
    ref = state.get("request_reference") or {}
    identifying = any(ref.get(k) for k in REFERENCE_KEYS)
    match = state.get("match_result") or {}
    # A partner needs no Contact Us request (the host's create-partner screen doesn't): with
    # none selected and none named, it is created on its own instead of asking for one.
    standalone_partner = (
        state.get("operation") == "create_partner" and not state.get("contact_request") and not identifying
    )
    # A status change naming list numbers has its request ids already; none means a number
    # was not in the list: nothing is prepared (never another request instead).
    targets = state.get("status_targets") if intent == "change_status" else None
    if targets is not None:
        if not targets:
            return "respond"
        return "prepare_status_change" if not ran("prepare_status_change") else "respond"

    # Which request? Retrieve first if nothing suitable is loaded.
    if identifying or (
        not state.get("contact_request") and (intent in REQUEST_INTENTS or converting) and not standalone_partner
    ):
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

    if intent in REQUEST_ACTION_INTENTS and state.get("contact_request"):
        step = "get_tracking_history" if intent == "show_history" else "prepare_status_change"
        return step if not ran(step) else "respond"

    # A conversion only advances when the user is working on it this turn.
    if not (converting and (state.get("contact_request") or standalone_partner) and intent in ADVANCING_INTENTS):
        return "respond"
    return _conversion_step(state, ran)


def _conversion_step(state: ContactUsChatState, ran: Callable[[str], int]) -> str:
    """The next node of a conversion the user is working on this turn.

    Order: set up the conversion -> its type -> provider -> validate the details -> role
    -> existing-account check -> sub-categories (Contractor) -> confirmation -> creation.
    Once the account exists, only the remaining steps after creation are resumed.
    """
    if not ran("start_conversion"):
        return "start_conversion"
    conv = conversion_of(state)
    if conv.get("blocked"):
        return "respond"
    ctype = active_type(state)
    intent = state.get("intent")
    completed = conv.get("completed") or []

    if "create_account" in completed:
        # The account exists: only resume the remaining steps, never create again. They run
        # straight on after a creation in this turn, or when the user asks to continue —
        # a correction or another message never re-runs them by itself.
        continuing = ran("create_account") or intent in ("retry_operation", "confirm_action", *CONVERSION_INTENTS)
        if not continuing:
            return "respond"
        for step in conversion_steps(ctype, conv):
            if step not in completed:
                return step if not ran(step) else "respond"
        if ran("update_request_status") and not ran("refresh_requests"):
            return "refresh_requests"
        return "respond"

    if ctype is None:
        return "choose_conversion_type" if not ran("choose_conversion_type") else "respond"
    if state.get("edit_request") and not state.get("turn_field_updates"):
        # "I need to change my address": ask which field / for the new value first.
        return "ask_edit_fields" if not ran("ask_edit_fields") else "respond"
    turn = state.get("turn_conversion") or {}
    answered_provider = (state.get("answered_question") or {}).get("type") == "choose_provider" and turn.get("option_position")
    if ctype.needs_target_provider and (
        not conv.get("target_provider")
        or turn.get("target_provider_name")
        or (turn.get("target_provider_id") and turn["target_provider_id"] != conv["target_provider"].get("id"))
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
    if not conv.get("role_id"):
        # First: the existing-account check compares role ids, as BOOTOG's duplicate check does.
        return "resolve_role" if not ran("resolve_role") else "respond"
    if not ran("check_existing_account"):
        # Every advancing turn: the email may have changed, or an earlier attempt's
        # account may have appeared (then it is adopted, never created again).
        return "check_existing_account"
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
    if ctype.auto_create:
        if _creation_authorized(state, conv, current):
            return "create_account" if not ran("create_account") else "respond"
        return "present_details" if not ran("present_details") else "respond"
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


def _filter_options(state: ContactUsChatState, ran: Callable[[str], int]) -> str:
    """After a request list: the filter card's company list, when not loaded recently."""
    if provider_catalog_stale(state) and not ran("load_filter_options"):
        return "load_filter_options"
    return "respond"


def _creation_authorized(state: ContactUsChatState, conv: dict[str, Any], fingerprint: str) -> bool:
    """Auto-created types (Customer): does THIS turn's user action authorise creating the
    account from the complete, valid draft? Yes for the conversion command itself, a
    submitted details form, a message that supplied the last missing details, an explicit
    "create" answering the summary the user saw (unchanged since), or a retry after a
    failed creation. No for a correction of an already complete draft, or while the user
    asked to review first: those show the summary and wait. A completed or created
    account never reaches here (the resume branch above handles it)."""
    if conv.get("review_hold") and not state.get("explicit_create"):
        return False
    if state.get("rejected_field_updates"):
        # The user tried to set a value that was not accepted: creating now would use the
        # old value they were replacing. Ask for a valid one first.
        return False
    if state.get("explicit_create"):
        # "Create" (card button or "create it") counts only for exactly the details the user
        # was shown: nothing changed since that summary.
        shown = conv.get("awaiting_fingerprint")
        if shown:
            return shown == fingerprint
        return bool(state.get("form_submitted") and not state.get("turn_field_updates"))
    if state.get("form_submitted") or state.get("completed_this_turn"):
        return True
    intent = state.get("intent")
    if intent in ("create_customer", "convert_request"):
        return True
    if (state.get("answered_question") or {}).get("type") == "choose_conversion_type":
        return True  # Customer picked from the "which account?" list
    failed = conv.get("failed") or {}
    return intent == "retry_operation" and failed.get("step") == "create_account"


def _unmerged(state: ContactUsChatState, ran: Callable[[str], int]) -> bool:
    """True when this turn brought details or notes that merge_customer_fields hasn't stored yet."""
    has_updates = state.get("turn_field_updates") or state.get("turn_extra_information")
    return bool(has_updates) and not ran("merge_customer_fields")


def build_contact_us_chat_graph(checkpointer=None):
    """Build and compile the chat graph: start_turn, every node in NODES, and respond.

    Every node except respond is followed by `next_step`, which picks the next node.

    Args:
        checkpointer: Saves each session's state between turns (None: no saving).
    """
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


sessions = ChatSessions(CHECKPOINTER)


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
        # The filter card: with a request list, or when a list filter has to be fixed first.
        "requestFilters": (
            request_filters_view(state)
            if context == "listed" or (state.get("intent") == "list_requests" and state.get("filter_problems"))
            else None
        ),
        # This turn's tracking history and status change (cards); never older ones.
        "history": state.get("history"),
        "statusChange": status_change_view(state),
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
    form: dict[str, Any] | None = None,
    viewer_user_id: str | None = None,
) -> ContactUsChatState:
    """Run one turn, streaming activity into `channel`. Always closes the channel and client.
    `form`: a chat card's structured submission ({action, request_id, fields}).
    `viewer_user_id`: the signed-in user's BOOTOG id ("requests assigned to me")."""
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
        "form_submission": form,
        "viewer_user_id": viewer_user_id,
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
    form: dict[str, Any] | None = None,
    viewer_user_id: str | None = None,
) -> None:
    """Run one turn, then always release the session for the next turn."""
    try:
        await run_contact_us_chat_turn(
            channel, session_id, message, client, agent_key=agent_key, agent_label=agent_label, form=form,
            viewer_user_id=viewer_user_id,
        )
    finally:
        sessions.end(session_id)


def start_turn_task(
    channel: RunChannel,
    session_id: str,
    message: str | None,
    client: BootogClient,
    agent_key: str | None = None,
    agent_label: str | None = None,
    form: dict[str, Any] | None = None,
    viewer_user_id: str | None = None,
) -> asyncio.Task:
    """Start `run_turn_in_session` in the background, so the API can answer right away."""
    return asyncio.create_task(
        run_turn_in_session(channel, session_id, message, client, agent_key, agent_label, form, viewer_user_id)
    )
