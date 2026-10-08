"""Frontend Order Enquiry models and pure record helpers.

Everything here is deterministic and side-effect free, so the same input always
gives the same split (filtering is idempotent and safe to re-run)."""

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

# Confirmed business rule: orderAction 6 = "Converted to Order". Never processed.
ORDER_ACTION_CONVERTED_TO_ORDER = 6


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class RunStarted(CamelModel):
    run_id: str
    steps: list[dict[str, str]]


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


# --- orderAction filtering ------------------------------------------------


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


def split_by_order_action(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Split validated records into (eligible, converted ids, invalid-orderAction ids).

    - orderAction == 6          -> converted, ignored
    - orderAction missing/null/unreadable -> set aside: we cannot confirm the
      enquiry is not already converted, so it is not sent further
    - any other orderAction     -> eligible
    """
    eligible: list[dict[str, Any]] = []
    converted: list[str] = []
    invalid: list[str] = []
    for record in records:
        action = parse_order_action(record.get("orderAction"))
        if action is None:
            invalid.append(enquiry_id(record) or "")
        elif action == ORDER_ACTION_CONVERTED_TO_ORDER:
            converted.append(enquiry_id(record) or "")
        else:
            eligible.append(record)
    return eligible, converted, invalid


# --- enquiry context ------------------------------------------------------
# Fields read from an OrderEnquery record. None of them is assumed present; a
# missing field is simply None in the context. The full record stays available in
# state (eligible_enquiries) for later stages that need provider/fee/industry data.

CONTEXT_FIELDS: tuple[str, ...] = (
    "orderId",
    "jobId",
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


def build_enquiry_context(record: dict[str, Any]) -> dict[str, Any]:
    """The enquiry context handed to the next workflow stage."""
    context: dict[str, Any] = {"enquiryId": enquiry_id(record)}
    for field in CONTEXT_FIELDS:
        context[field] = _clean(record.get(field))
    context["orderAction"] = parse_order_action(record.get("orderAction"))
    name = " ".join(p for p in (context["customerFirstName"], context["customerLastName"]) if p)
    context["customerName"] = name or None
    return context


def summarize_enquiry(context: dict[str, Any]) -> dict[str, Any]:
    """The subset returned in the run result. No email/phone."""
    return {
        "enquiryId": context.get("enquiryId"),
        "orderId": context.get("orderId"),
        "customerName": context.get("customerName"),
        "status": context.get("status"),
        "orderAction": context.get("orderAction"),
        "city": context.get("city"),
        "state": context.get("state"),
        "createdDateTime": context.get("createdDateTime"),
    }
