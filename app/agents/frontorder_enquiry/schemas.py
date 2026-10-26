"""Frontend Order Enquiry models and pure record helpers.

Everything here is deterministic and side-effect free, so the same input always
gives the same split (filtering is idempotent and safe to re-run)."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app.agents.frontorder_enquiry.api_client import ORDER_ACTION_CONVERTED_TO_ORDER

# Request actions (see understanding.py) that are ABOUT converted enquiries ("show
# converted enquiries", "how many were converted"). Every other action — new
# enquiries, follow-up lists, sending follow-ups — is about enquiries that are still
# open, so converted ones are left out of those results (but never deleted or hidden
# from a converted-enquiry request).
CONVERTED_ACTIONS = {"list_converted"}
# Actions that need each enquiry's Contact Log history.
FOLLOW_UP_ACTIONS = {"list_followed_up", "list_not_followed_up"}


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class RunStarted(CamelModel):
    run_id: str
    steps: list[dict[str, str]]
    # The conversation this run belongs to. Send it back with the next message so the
    # agent keeps the provider, dates and kind of enquiries ("now only converted ones").
    session_id: str | None = None
    # True when the sessionId sent was no longer live (restart/expiry): a new
    # conversation was started and the previous context is not applied.
    session_restarted: bool = False


class FrontOrderEnquiryRunRequest(CamelModel):
    # The admin's chat message, e.g. "How many enquiries today?". No message = "Run
    # Agent": the provider summary of open enquiries, without a date filter.
    message: str | None = Field(None, max_length=4000)
    # A provider picked in the UI (e.g. from the provider summary). It is verified against
    # Bootog's provider list before use. Without it, a provider named in the message is
    # looked up by name; with neither, the agent answers with the provider summary.
    target_provider_id: str | None = None
    # Conversation id from a previous RunStarted, to continue its context.
    session_id: str | None = Field(None, max_length=64)
    # Page of the current view (provider summary or a provider's enquiries).
    page: int | None = Field(None, ge=1, le=10000)


# --- validation -----------------------------------------------------------


def enquiry_id(record: Any) -> str | None:
    """The enquiry's id as a string, or None when the record has no usable id."""
    if not isinstance(record, dict):
        return None
    value = record.get("id")
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_enquiries(records: list[Any]) -> tuple[list[dict[str, Any]], int, int]:
    """Keep dict records that carry an id, first occurrence per id.
    Returns (valid, malformed count, duplicate count)."""
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    malformed = duplicates = 0
    for record in records:
        record_id = enquiry_id(record)
        if record_id is None:
            malformed += 1
        elif record_id in seen:
            duplicates += 1
        else:
            seen.add(record_id)
            valid.append(record)
    return valid, malformed, duplicates


# --- orderAction classification and intent selection ------------------------
# Three separate steps, so each stays simple:
#   1. classify_enquiries          -> which records are converted / not converted
#   2. filter_enquiries_for_intent -> which of those the user asked about
#   3. later workflow stages act only on the selected records


def parse_order_action(value: Any) -> int | None:
    """orderAction as an int, or None when missing/unreadable. Accepts 6, 6.0 and "6"."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


@dataclass
class ClassifiedEnquiries:
    """Validated records grouped by orderAction. The records are not modified."""

    not_converted: list[dict[str, Any]] = field(default_factory=list)
    converted: list[dict[str, Any]] = field(default_factory=list)
    # orderAction missing/null/unreadable: we cannot tell whether the enquiry was
    # converted, so no request selects it; its id is reported instead.
    unknown: list[dict[str, Any]] = field(default_factory=list)


def classify_enquiries(records: list[dict[str, Any]]) -> ClassifiedEnquiries:
    """Group records by orderAction: 6 = converted to order, any other number = not
    converted, missing/unreadable = unknown."""
    classified = ClassifiedEnquiries()
    for record in records:
        action = parse_order_action(record.get("orderAction"))
        if action is None:
            classified.unknown.append(record)
        elif action == ORDER_ACTION_CONVERTED_TO_ORDER:
            classified.converted.append(record)
        else:
            classified.not_converted.append(record)
    return classified


def wants_converted(action: str | None) -> bool:
    """True when the request is about converted enquiries."""
    return action in CONVERTED_ACTIONS


def filter_enquiries_for_intent(action: str | None, classified: ClassifiedEnquiries) -> list[dict[str, Any]]:
    """The records the user asked about: converted ones for a converted-enquiry
    request, otherwise only those not converted yet."""
    return classified.converted if wants_converted(action) else classified.not_converted


# --- enquiry context ------------------------------------------------------
# Fields read from an OrderEnquery record. None of them is assumed present; a
# missing field is simply None in the context. The full record stays available in
# state (selected_enquiries) for later stages that need provider/fee/industry data.

CONTEXT_FIELDS: tuple[str, ...] = (
    "orderId",
    "jobId",
    "providerId",
    "providerName",
    "industryName",
    "serviceCategoryName",
    "assignedToUserName",
    "assignmentOrigin",
    "customerId",
    "customerFirstName",
    "customerLastName",
    "customerEmail",
    "customerPhone",
    "isSmsAgreed",
    "status",
    "mainServiceTypeId",
    "additionalServiceIds",
    "address",
    "city",
    "state",
    "zipCode",
    "county",
    "date",
    "fromTime",
    "toTime",
    "createdDateTime",
    "modifiedDateTime",
)


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def main_service_name(record: dict[str, Any]) -> str | None:
    """Name of the main service: the fee row flagged isMainService (the same lookup the
    .NET OrderEnqueryService uses for its emails)."""
    fees = record.get("fees")
    for fee in fees if isinstance(fees, list) else []:
        if isinstance(fee, dict) and fee.get("isMainService") is True:
            return _clean(fee.get("serviceTypeName"))
    return None


def build_enquiry_context(record: dict[str, Any]) -> dict[str, Any]:
    """The enquiry context handed to the next workflow stage."""
    context: dict[str, Any] = {"enquiryId": enquiry_id(record)}
    for name in CONTEXT_FIELDS:
        context[name] = _clean(record.get(name))
    context["orderAction"] = parse_order_action(record.get("orderAction"))
    context["mainServiceName"] = main_service_name(record)
    name = " ".join(p for p in (context["customerFirstName"], context["customerLastName"]) if p)
    context["customerName"] = name or None
    return context


def summarize_enquiry(context: dict[str, Any]) -> dict[str, Any]:
    """The subset returned in the run result for the chat to display. No email/phone.
    `status` and `orderAction` stay numeric: the Angular app already maps them to labels."""
    return {
        "enquiryId": context.get("enquiryId"),
        "jobId": context.get("jobId"),
        "customerName": context.get("customerName"),
        "createdDateTime": context.get("createdDateTime"),
        "mainService": context.get("mainServiceName"),
        "serviceCategory": context.get("serviceCategoryName"),
        "industry": context.get("industryName"),
        "providerName": context.get("providerName"),
        "assignedTo": context.get("assignedToUserName"),
        "status": context.get("status"),
        "orderAction": context.get("orderAction"),
        "city": context.get("city"),
        "state": context.get("state"),
        # Only set when the request asked about follow-up (None = not checked).
        "followedUp": context.get("followedUp"),
        "lastContactedAt": context.get("lastContactedAt"),
    }


# --- follow-up detection (OrderEnqueryTrackingHistory) ----------------------
# Verified in OrderEnqueryService.cs: the history table gets three kinds of rows.
#   1. the creation row ("Order Inquiry Initialize"),
#   2. ChangeStatus rows (status change only),
#   3. Contact Log rows written by OrderEnqueryLogUpdate (a person logged a call,
#      SMS, email or note).
# Only kind 3 ever fills the fields below, so a row with any of them is a real contact.
# `followUpNeeded` / `dateOfFollowUp` only mean a follow-up is *planned*, so they are
# deliberately not used.
CONTACT_LOG_FIELDS: tuple[str, ...] = ("userName", "contactName", "contactNumber", "emailTo", "smsTo", "note")


def is_contact_log(row: dict[str, Any]) -> bool:
    return any(isinstance(row.get(f), str) and row[f].strip() for f in CONTACT_LOG_FIELDS)


def latest_contact(history: list[dict[str, Any]]) -> str | None:
    """createdDateTime of the newest Contact Log row, or None when nobody has contacted
    the customer about this enquiry yet."""
    dates = [str(row.get("createdDateTime")) for row in history if is_contact_log(row) and row.get("createdDateTime")]
    return max(dates) if dates else None
