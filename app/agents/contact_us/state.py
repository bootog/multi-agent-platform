from typing import Any, Literal, TypedDict

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
    contact_request: dict[str, Any] | None
    customer_payload: dict[str, Any] | None
    missing_required_fields: list[str]
    customer_result: dict[str, Any] | None

    current_step: str
    execution_status: ExecutionStatus
    error: str | None
    started_at: str
    finished_at: str
