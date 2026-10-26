from typing import Any, Literal, TypedDict

ExecutionStatus = Literal["running", "completed", "blocked", "failed"]
View = Literal["provider_summary", "provider_detail", "clarification", "none"]


class EnquiryScope(TypedDict, total=False):
    """What the conversation is currently looking at. Kept across messages of one
    conversation (LangGraph thread), so "now show only converted ones" keeps the
    provider and dates. Ids only — no customer data."""

    action: str
    date_from: str | None  # inclusive YYYY-MM-DD (business timezone); None = no date filter
    date_to: str | None
    provider_id: str | None  # verified against Bootog's provider list
    provider_name: str | None
    page: int


class FrontOrderEnquiryAgentState(TypedDict, total=False):
    # --- kept across the messages of a conversation (checkpointed) ---
    scope: EnquiryScope
    # The numbered provider list shown last (summary page or name candidates), so the
    # next message can pick one by number. Each: {number, providerId, providerName}.
    # Ids come from Bootog's responses only.
    offered_providers: list[dict[str, Any]]

    # --- everything below is reset at the start of every message (see graph.new_turn) ---
    run_id: str
    session_id: str
    # Run input: the admin's chat message (None = "Run Agent"), and an optional provider id
    # and page sent by the UI (e.g. a click on a provider in the summary).
    message: str | None
    target_provider_id: str | None
    requested_page: int | None

    # understand_request: the interpretation of THIS message, before merging with scope.
    action: str
    request_summary: str
    requested: dict[str, Any]
    # Set when the request is unclear: the run stops and this question is returned.
    clarification: str | None
    # Providers offered when a name matched several (id + name only).
    provider_candidates: list[dict[str, str]]
    # resolve_scope: which view this message produces.
    view: View

    # summarize_providers: one page of providers ranked by matching enquiries.
    provider_summary: list[dict[str, Any]]
    providers_total: int

    # fetch_enquiries: raw records as returned by OrderEnquery for the selected provider.
    # The query is scoped to converted / not-converted enquiries to match the request, so
    # `enquiries_total` (the server's count) is the exact number of matching enquiries.
    raw_enquiries: list[Any]
    enquiries_total: int
    # True when more matching enquiries exist than were read (another page, or the
    # page cap of a follow-up request).
    enquiries_truncated: bool
    # validate_enquiries: well-formed records (dict with an id), de-duplicated by id.
    valid_enquiries: list[dict[str, Any]]
    malformed_count: int
    duplicate_count: int
    # select_enquiries: classify_enquiries (orderAction 6 = converted to order) then
    # filter_enquiries_for_intent. `selected_enquiries` holds the records the request
    # is about (converted ones for a converted-enquiry request, otherwise those not
    # converted); check_follow_up narrows it further for follow-up requests.
    converted_enquiry_ids: list[str]
    invalid_order_action_ids: list[str]
    selected_enquiries: list[dict[str, Any]]
    # check_follow_up (follow-up requests only): enquiry id -> newest contact time or None.
    follow_up_by_id: dict[str, dict[str, Any]]
    # prepare_enquiries: per-enquiry context handed to the next workflow stage.
    enquiry_contexts: list[dict[str, Any]]

    # respond: the chat reply and the result payload (display fields only).
    reply: str
    result: dict[str, Any]

    current_step: str
    execution_status: ExecutionStatus
    error: str | None
    started_at: str
    finished_at: str
