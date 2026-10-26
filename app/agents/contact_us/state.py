from typing import Annotated, Any, Literal, TypedDict

RunMode = Literal["retrieve", "create_customer"]
ExecutionStatus = Literal["running", "completed", "blocked", "failed"]


class ContactUsAgentState(TypedDict, total=False):
    run_id: str
    mode: RunMode
    provider_id: str | None
    provider_name: str | None
    contact_request_id: str | None
    # Agent Behavior settings from the UI. Carried for later phases; no Phase 1
    # step acts on them.
    options: dict[str, Any]

    contact_requests: list[dict[str, Any]]
    contact_requests_total: int
    contact_requests_truncated: bool  # a search hit the page limit before reading everything
    contact_request: dict[str, Any] | None
    customer_payload: dict[str, Any] | None
    missing_required_fields: list[str]
    customer_result: dict[str, Any] | None

    current_step: str
    execution_status: ExecutionStatus
    error: str | None
    started_at: str
    finished_at: str


# --- Chat agent ------------------------------------------------------------
# One LangGraph thread per chat session (checkpointed), one graph invocation per
# user message. Fields above keep their meaning, so the existing nodes run
# unchanged inside the chat graph:
#   contact_request_id / contact_request  -> the selected request (id / record)
#   customer_payload                      -> customer data; only CUSTOMER_FIELDS keys
#   missing_required_fields               -> labels still required

WorkflowStatus = Literal["ready", "running", "waiting_for_user", "completed", "blocked", "failed"]

_HISTORY_LIMIT = 40


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def append_history(existing: list[ChatMessage] | None, new: list[ChatMessage] | None) -> list[ChatMessage]:
    return ((existing or []) + (new or []))[-_HISTORY_LIMIT:]


class ContactUsChatState(ContactUsAgentState, total=False):
    session_id: str
    # Which Contact agent this session belongs to (Contact menu route segment) and the
    # GET /ContactUs module clause it works on. Fixed for the life of the session.
    agent_key: str
    agent_name: str
    module_filter: str
    # Fingerprint of the caller that owns the session; turns from anyone else are refused.
    owner: str
    conversation_history: Annotated[list[ChatMessage], append_history]

    # Set per turn
    current_user_message: str | None
    kickoff: bool  # "Run Agent": start the session without a user message
    intent: str | None
    request_summary: str | None  # neutral one-line label of what the user asked
    request_reference: dict[str, Any] | None
    request_filters: dict[str, Any] | None
    company_name: str | None
    turn_steps: list[str]  # nodes executed this turn (loop guard for routing)
    match_result: dict[str, Any] | None
    turn_field_updates: dict[str, str]  # customer fields stated in this message
    turn_extra_information: dict[str, str]
    tool_errors: list[str]
    last_tool_result: dict[str, Any] | None
    agent_response: str | None
    completed_steps: list[str]
    failed_steps: list[str]
    # Conversion details named in this message (see understanding.ConversionDetails):
    # partner_type, target_provider_name, option_position, subcategory_names.
    turn_conversion: dict[str, Any]
    answered_question: dict[str, Any] | None  # the pending question this message answers
    provider_query: dict[str, Any] | None  # "list/find providers": {"name", "location"}
    provider_results: dict[str, Any] | None
    refreshed: bool  # the request list was re-read after a conversion
    # Structured form from the chat UI ({"action", "requestId", "fields"}), instead of text.
    form_submission: dict[str, Any] | None
    form_submitted: bool  # this turn submitted the details form (an explicit user action)
    explicit_create: bool  # this turn explicitly asked to create the account now
    completed_this_turn: bool  # this turn's values completed a draft that was incomplete
    rejected_field_updates: dict[str, str]  # label -> why; the previous value was kept
    rejected_uploads: dict[str, str]  # upload label -> why; the attached image (if any) was kept
    filter_problems: list[str]  # list filters that could not be applied: nothing is listed, the user is asked
    filters_before: dict[str, Any] | None  # active_filters before this turn (restored when the assignee can't be resolved)
    viewer_user_id: str | None  # the signed-in user's BOOTOG id, from the host's session ("my requests")
    status_targets: list[str] | None  # request ids a status change names by list number this turn
    status_request: dict[str, Any] | None  # this turn's change_status ask: {status, comment}
    status_change_confirmed: bool  # this turn confirms the pending status change (chat "yes" or the card)
    history: dict[str, Any] | None  # the tracking history read this turn (card + facts)
    company_pick: dict[str, str] | None  # a company chosen from the offered ones (card or number)
    post_creation_edit: list[str]  # labels the user tried to change after the account existed
    edit_request: dict[str, Any] | None  # "I need to change my address" (no value yet)

    # Kept across turns
    operation: str | None  # ongoing workflow goal: create_customer / create_b2b_client / create_partner
    # The conversion in progress, bound to one request. Holds only ids, names and step
    # outcomes from real API responses — never credentials:
    #   type, request_id, role_id, role_name, target_provider {id, name},
    #   provider_candidates, subcategories [{id, name}], account {...},
    #   awaiting_fingerprint (what the user was asked to confirm), create_attempted,
    #   created_vendor_id, user_id, completed [steps], failed {step, message, outcome_unknown}
    conversion: dict[str, Any] | None
    assigned_users_cache: dict[str, Any] | None  # {"at": epoch seconds, "users": [{id, name}]}
    # The request-list filters in force, kept across turns so follow-ups only change what
    # they name: {statuses [int], all_statuses, only_open, created_from, created_to,
    # assigned_user_name, assigned_user_ids, page}. The company is `provider_id/_name`.
    active_filters: dict[str, Any] | None
    # SearchProviders (the Contact Us screen's company filter list): {"at", "providers", "truncated"}.
    provider_catalog: dict[str, Any] | None
    company_candidates: list[dict[str, str]] | None  # offered when a company name matched several
    assignee_candidates: list[dict[str, str]] | None  # offered when an assignee name matched several / none
    # A status change waiting for (or just past) confirmation. Ids and names come from BOOTOG
    # records only: {token, ids, requests [{id, name, status}], status, comment, state,
    # changed, failed, message}. Executed only for this token, once.
    status_change: dict[str, Any] | None
    listed_request_ids: list[str]  # order of the requests last shown to the user
    user_customer_fields: dict[str, str]  # values the user supplied (latest wins)
    # Images attached on the card for one request: {"request_id", "files": {key: {name,
    # content_type, size, sha, data (base64; dropped once sent with the account)}}}.
    # Kept apart from `conversion`, whose content is described to the LLM.
    conversion_files: dict[str, Any] | None
    known_fields: list[str]
    invalid_fields: dict[str, str]
    extra_information: dict[str, str]  # useful info outside the API schema; never sent
    pending_question: dict[str, Any] | None
    awaiting_user_input: bool
    created_customers: dict[str, str]  # request id -> customer id; never create twice
    workflow_status: WorkflowStatus
    is_complete: bool
