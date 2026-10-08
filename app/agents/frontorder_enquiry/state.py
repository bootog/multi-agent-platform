from typing import Any, Literal, TypedDict

ExecutionStatus = Literal["running", "completed", "blocked", "failed"]


class FrontOrderEnquiryAgentState(TypedDict, total=False):
    run_id: str

    # fetch_enquiries: raw records as returned by OrderEnquery.
    raw_enquiries: list[Any]
    enquiries_total: int
    enquiries_truncated: bool
    # validate_enquiries: well-formed records (dict with an id), de-duplicated by id.
    valid_enquiries: list[dict[str, Any]]
    malformed_count: int
    duplicate_count: int
    # filter_converted: orderAction == 6 -> ignored; unreadable orderAction -> skipped.
    eligible_enquiries: list[dict[str, Any]]
    converted_enquiry_ids: list[str]
    invalid_order_action_ids: list[str]
    # prepare_enquiries: per-enquiry context handed to the next workflow stage.
    enquiry_contexts: list[dict[str, Any]]

    current_step: str
    execution_status: ExecutionStatus
    error: str | None
    started_at: str
    finished_at: str
