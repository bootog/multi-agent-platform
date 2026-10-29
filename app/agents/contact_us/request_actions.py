"""Chat nodes that act on Contact Us requests themselves (no account conversion).

    get_tracking_history   GET ContactUs/GetContactUsTrackingHistory (filter=contactUsId=<id>)
    prepare_status_change  validate the new status and the target requests; ask to confirm
    change_request_status  POST ContactUs/ChangeStatus {status, ids, comments?} — only for the
                           confirmed pending change, once — then read each request back

The request ids always come from BOOTOG records the user picked (the selected request, or
list numbers resolved to the listed ids); status ids from ContactUsStatus. A status change
runs only when this turn confirms exactly the change that was shown (its token), whatever
the LLM made of the message."""

import html
import re
import time
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.contact_us.nodes import StepFailed, StepResult, tracked_step
from app.agents.contact_us.state import ContactUsChatState
from app.agents.contact_us.tools import change_contact_us_status, get_contact_us_history, get_contact_us_requests
from app.integrations.bootog import BootogApiError

# ContactUsStatus (contact-us-status.model.ts). The Change Status popup offers every status
# except CRM Completed (set by converting a request) and Assigned (set by assigning a user).
CONTACT_US_STATUS_LABELS: dict[int, str] = {
    1: "Proposal Submitted", 2: "Pending Approval", 3: "Needs Clarification", 4: "Under Review",
    5: "Approved", 6: "Rejected", 7: "In Implementation", 8: "Completed", 9: "On Hold",
    10: "Cancelled", 11: "CRM Completed", 12: "Closed", 13: "Re Open", 14: "Re Scheduled",
    15: "Assigned", 16: "Left Message", 17: "Email Sent", 18: "Document Requested",
}
NOT_SETTABLE = {
    11: "CRM Completed is set by converting the request into an account",
    15: "Assigned is set by assigning the request to a user",
}
SETTABLE_STATUSES = {k: v for k, v in CONTACT_US_STATUS_LABELS.items() if k not in NOT_SETTABLE}
COMMENT_LIMIT = 1000
_HISTORY_SHOWN = 50
_TAGS = re.compile(r"<[^>]*>")
_BLOCK_END = re.compile(r"</(p|div|li|h\d)>|<br\s*/?>", re.IGNORECASE)


def status_value(name: str | None) -> int | None:
    """ContactUsStatus value for a status name ("under review", "Re-Open", "cancel" ...).

    Case, spaces and punctuation are ignored. Returns None when the name is not a status.
    """
    key = re.sub(r"[^a-z]", "", (name or "").casefold())
    if not key:
        return None
    for value, label in CONTACT_US_STATUS_LABELS.items():
        if re.sub(r"[^a-z]", "", label.casefold()) == key:
            return value
    # Common ways of naming a status that aren't its label.
    return {"cancel": 10, "canceled": 10, "approve": 5, "reject": 6, "reopen": 13, "hold": 9}.get(key)


def text_of(value: Any) -> str | None:
    """Plain text of a history comment / note (the host's editor stores HTML): tags
    dropped, entities decoded. Shown as text only — never inserted as HTML."""
    if value is None:
        return None
    raw = _BLOCK_END.sub("\n", str(value))
    text = html.unescape(_TAGS.sub("", raw))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip() or None


def full_name(record: dict[str, Any] | None) -> str:
    """"First Last" of a Contact Us record, or "Unnamed request" when it has no name."""
    return " ".join(p for p in ((record or {}).get("firstName"), (record or {}).get("lastName")) if p) or "Unnamed request"


# --- tracking history ------------------------------------------------------------------------


def _history_target(state: ContactUsChatState) -> dict[str, Any] | None:
    """The request whose history to read: the one a status change just changed, else the
    selected request."""
    change = state.get("status_change") or {}
    if "change_request_status" in (state.get("turn_steps") or []) and len(change.get("changed") or []) == 1:
        rid = change["changed"][0]
        return next((r for r in change.get("requests") or [] if r["id"] == rid), {"id": rid})
    record = state.get("contact_request")
    return {"id": record.get("id"), "name": full_name(record)} if record and record.get("id") else None


def _entry(row: dict[str, Any]) -> dict[str, Any]:
    """One tracking-history row as the history card shows it (HTML comments as plain text)."""
    status = row.get("contactUsStatus")
    return {
        "id": row.get("id"),
        "statusId": status if isinstance(status, int) else None,
        "status": CONTACT_US_STATUS_LABELS.get(status) if isinstance(status, int) else None,
        "comment": text_of(row.get("statusComments")),
        "note": text_of(row.get("note")),
        "user": row.get("userName") or row.get("createdByUserName") or None,
        "contactName": row.get("contactName") or None,
        "contactNumber": row.get("contactNumber") or None,
        "dateOfContact": row.get("dateOfContact"),
        "timeOfContact": row.get("timeOfContact"),
        "followUpNeeded": row.get("followUpNeeded") is True,
        "dateOfFollowUp": row.get("dateOfFollowUp"),
        "timeOfFollowUp": row.get("timeOfFollowUp"),
        "emailTo": row.get("emailTo") or None,
        "smsTo": row.get("smsTo") or None,
        "createdAt": row.get("createdDateTime"),
    }


@tracked_step("get_tracking_history", "Reading the request history", "Unable to read the request history.")
async def get_tracking_history(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    """Read the tracking history of the target request into `history` (for the card).

    The first `_HISTORY_SHOWN` entries are kept. When BOOTOG refuses, the step still
    completes, with the error recorded in `history`, so the reply says the history could
    not be loaded instead of claiming there is none.

    Raises:
        StepFailed: No request is selected.
    """
    target = _history_target(state)
    if not target:
        raise StepFailed("No request is selected to show the history of.")
    view: dict[str, Any] = {"requestId": target["id"], "name": target.get("name"), "entries": [], "total": 0}
    try:
        result = await get_contact_us_history.ainvoke({"contact_us_id": target["id"]}, config=config)
    except BootogApiError as exc:
        # The card and the reply say the history could not be loaded — never "no history".
        view["error"] = exc.message
        return StepResult("The history could not be loaded", {"history": view})
    entries = [_entry(r) for r in result["entries"]]
    view.update(total=result["total"], entries=entries[:_HISTORY_SHOWN],
                truncated=result["truncated"] or len(entries) > _HISTORY_SHOWN)
    message = f"{result['total']} history entr{'ies' if result['total'] != 1 else 'y'}" if entries else "No history yet"
    return StepResult(message, {"history": view}, data={"entries": len(entries)})


# --- status change -----------------------------------------------------------------------------


def _targets(state: ContactUsChatState) -> list[dict[str, Any]]:
    """The requests a change names: list numbers resolved to listed ids, else the selected one."""
    by_id = {r.get("id"): r for r in state.get("contact_requests") or []}
    selected = state.get("contact_request")
    if selected:
        by_id[selected.get("id")] = selected
    ids = state.get("status_targets") or ([selected["id"]] if selected and selected.get("id") else [])
    return [by_id[i] for i in ids if i in by_id]


def _settable_options() -> list[dict[str, Any]]:
    """The statuses the user may choose, as {"id", "name"} options for the card."""
    return [{"id": k, "name": v} for k, v in SETTABLE_STATUSES.items()]


@tracked_step("prepare_status_change", "Preparing the status change", "Unable to prepare the status change.")
async def prepare_status_change(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    """Prepare a status change and ask the user to confirm it. Nothing is sent here.

    The change is stored in `status_change` with a new token. Its `state` is
    "confirming" when a settable status was named, else "choosing" (the card offers the
    statuses). When every target request already has that status, there is nothing to
    change and nothing is pending.

    Raises:
        StepFailed: No request is selected.
    """
    ask = state.get("status_request") or {}
    records = _targets(state)
    if not records:
        raise StepFailed("No request is selected to change the status of.")
    wanted = status_value(ask.get("status"))
    problem = None
    if ask.get("status") and wanted is None:
        problem = f"“{ask['status']}” is not a Contact Us status"
    elif wanted in NOT_SETTABLE:
        problem = f"{NOT_SETTABLE[wanted]}; it can't be chosen here"
        wanted = None
    comment = (ask.get("comment") or "").strip()[:COMMENT_LIMIT] or None
    requests = [{"id": r["id"], "name": full_name(r), "status": CONTACT_US_STATUS_LABELS.get(r.get("status"))} for r in records]
    if wanted and all(r.get("status") == wanted for r in records):
        note = {"status_change": {"already": CONTACT_US_STATUS_LABELS[wanted], "requests": [r["name"] for r in requests]}}
        return StepResult(f"Already {CONTACT_US_STATUS_LABELS[wanted]} — nothing to change", {"last_tool_result": note, "status_change": None})
    change = {
        "token": uuid.uuid4().hex,
        "ids": [r["id"] for r in requests],
        "requests": requests,
        "status": {"id": wanted, "name": CONTACT_US_STATUS_LABELS[wanted]} if wanted else None,
        "comment": comment,
        "state": "confirming" if wanted else "choosing",
        "problem": problem,
        "at": time.time(),
        "run_id": state.get("run_id"),
    }
    names = ", ".join(r["name"] for r in requests)
    message = f"Confirm: {names} → {change['status']['name']}" if wanted else f"Choose the new status for {names}"
    return StepResult(
        message,
        {"status_change": change, "pending_question": {"type": "confirm_status_change"}, "awaiting_user_input": True},
    )


@tracked_step("change_request_status", "Changing the request status", "Changing the status failed.")
async def change_request_status(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    """Send the confirmed status change to BOOTOG, then read each request back to verify it.

    The outcome is recorded in `status_change`: "done" (all verified), "partial", "failed",
    or "unknown" (BOOTOG did not answer). Verified requests are updated in the
    conversation's copies of the list and of the selected request.

    Raises:
        StepFailed: This turn did not confirm a pending change.
    """
    change = dict(state.get("status_change") or {})
    # Enforced here, not by the prompt: exactly the change that was shown, confirmed now, once.
    if not (state.get("status_change_confirmed") and change.get("state") == "confirming"
            and (change.get("status") or {}).get("id") in SETTABLE_STATUSES and change.get("ids")):
        raise StepFailed("There is no confirmed status change to make.", {"status_change": change or None})
    status = change["status"]["id"]
    change.update(state="sending", run_id=state.get("run_id"))
    try:
        changed = await change_contact_us_status.ainvoke(
            {"request_ids": change["ids"], "status": status, "comments": change.get("comment")}, config=config
        )
    except BootogApiError as exc:
        unknown = exc.outcome_unknown
        change.update(state="unknown" if unknown else "failed", message=exc.message, changed=[], failed=[
            {"id": i, "reason": "no answer from BOOTOG" if unknown else "refused"} for i in change["ids"]
        ])
        return StepResult(f"The status was not changed. {exc.message}", {"status_change": change})

    # BOOTOG answers with the ids it changed; each one is read back to confirm the status.
    records: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, str]] = []
    confirmed: list[str] = []
    for rid in change["ids"]:
        if rid not in changed:
            failed.append({"id": rid, "reason": "BOOTOG did not report it as changed"})
            continue
        try:
            page = await get_contact_us_requests.ainvoke(
                {"request_id": rid, "provider_id": state.get("provider_id"), "module_filter": state.get("module_filter")},
                config=config,
            )
            record = next((r for r in page["records"] if r.get("id") == rid), None)
        except BootogApiError:
            record = None
        if record is None:
            failed.append({"id": rid, "reason": "changed according to BOOTOG, but it could not be read back to verify"})
        elif record.get("status") != status:
            failed.append({"id": rid, "reason": f"still shows {CONTACT_US_STATUS_LABELS.get(record.get('status'), 'another status')}"})
        else:
            confirmed.append(rid)
            records[rid] = record
    change.update(
        state="done" if confirmed and not failed else "partial" if confirmed else "failed",
        changed=confirmed, failed=failed, message=None,
    )
    update: dict[str, Any] = {"status_change": change}
    # The conversation's copies of those requests show the new status.
    if records:
        update["contact_requests"] = [records.get(r.get("id"), r) for r in state.get("contact_requests") or []]
        selected = state.get("contact_request")
        if selected and selected.get("id") in records:
            update["contact_request"] = {**selected, **records[selected["id"]]}
    label = change["status"]["name"]
    message = (f"{len(confirmed)} request{'s' if len(confirmed) != 1 else ''} changed to {label} (verified)"
               + (f"; {len(failed)} not changed" if failed else ""))
    return StepResult(message, update, data={"changed": len(confirmed), "failed": len(failed)})


def status_change_view(state: ContactUsChatState) -> dict[str, Any] | None:
    """The status-change card, for the turn that prepared, confirmed or cancelled it."""
    change = state.get("status_change") or {}
    if not change or change.get("run_id") != state.get("run_id"):
        return None
    return {
        "token": change.get("token") if change.get("state") in ("confirming", "choosing") else None,
        "state": change.get("state"),
        "requests": change.get("requests") or [],
        "status": change.get("status"),
        "comment": change.get("comment"),
        "problem": change.get("problem"),
        "statusOptions": _settable_options(),
        "changed": change.get("changed") or [],
        "failed": change.get("failed") or [],
        "message": change.get("message"),
    }


def status_change_facts(state: ContactUsChatState) -> dict[str, Any] | None:
    """What the reply LLM is told about this turn's status change, with what to say next.

    Returns None when this turn has no status change.
    """
    view = status_change_view(state)
    if not view:
        return None
    names = {r["id"]: r["name"] for r in view["requests"]}
    facts: dict[str, Any] = {
        "state": view["state"],
        "requests": [{"name": r["name"], "current_status": r["status"]} for r in view["requests"]],
        "new_status": (view["status"] or {}).get("name"),
        "comment": view["comment"],
    }
    if view["problem"]:
        facts["problem"] = view["problem"]
    state_next = {
        "choosing": "Ask which status to set (the card lists them); nothing was changed.",
        "confirming": "Ask the user to confirm this exact change (reply yes, or use the card). Nothing was changed yet.",
        "cancelled": "Say the status change was cancelled; nothing was changed.",
        "done": "Say the status was changed and verified in BOOTOG for these requests.",
        "partial": "Say which requests changed and which did not (with the reasons).",
        "failed": "Say the status was NOT changed and why.",
        "unknown": "Say BOOTOG did not answer, so it is not known whether the status changed; suggest checking the "
        "request before trying again. Do not say it changed.",
    }
    facts["next"] = state_next.get(view["state"] or "", "")
    if view["state"] in ("done", "partial", "failed", "unknown"):
        facts["changed"] = [names.get(i, i) for i in view["changed"]]
        facts["not_changed"] = [{"request": names.get(f["id"], f["id"]), "reason": f["reason"]} for f in view["failed"]]
        if view["message"]:
            facts["bootog_message"] = view["message"]
        if view["comment"] and view["state"] in ("done", "partial"):
            history = state.get("history") or {}
            latest = (history.get("entries") or [{}])[0] if not history.get("error") else {}
            facts["comment_recorded"] = (
                "yes — it is in the request history" if latest.get("comment") and view["comment"] in latest["comment"]
                else "not confirmed — the history does not show it"
            )
    return facts
