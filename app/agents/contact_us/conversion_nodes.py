"""Chat nodes that convert a Contact Us request into a BOOTOG account.

    start_conversion         bind the conversion to the selected request and type
    choose_conversion_type   ask which partner company type (waits for the user)
    resolve_target_provider  service provider the B2B client / partner goes under
    check_existing_account   GET UserDetail/GetUserRolesByEmailId (also recovers an
                             account an earlier, unanswered attempt did create)
    resolve_role             GET Roles/GetRoles for the type's default role
    select_subcategories     Contractor partners: GET ContractorSubCategory
    confirm_conversion       summary + explicit confirmation (waits for the user)
    create_account           POST UserDetail/AddVendorUser — at most once per conversion
                             (with the partner's Company Logo / User Headshot, if attached)
    verify_account           GET UserDetail/GetUserRolesByEmailId (account + role present)
    link_provider_vendor     POST Provider/SaveProviderVendor (idempotent)
    save_subcategories       POST ProviderContractorSubCategory/BulkInsert
    update_request_status    PUT ContactUs/bulk-update-file {Status: 11}, then GET ContactUs
                             (id=…) to confirm it reads CRM Completed. Never for a partner
                             created without a Contact Us request (see `standalone`)
    refresh_requests         GET ContactUs (the list after the change)
    search_providers         "list/find providers" (no conversion needed)
    resolve_assigned_user    "assigned to X" -> assignee id (GetContactUsAssignedUser)

Execution steps record their outcome in `conversion` (completed / failed), also on
failure, so "retry" resumes at the failed step and never creates a second account.
Critical identifiers come only from API responses, never from the LLM."""

import time
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.contact_us.conversions import (
    ALL_TYPES,
    CONVERSION_TYPES,
    CUSTOMER,
    PARTNER_FORM,
    PARTNER_TYPES,
    ConversionType,
    OPERATION_TYPE,
    build_draft,
    build_vendor_user_form,
    conversion_type,
    fingerprint,
    generate_password,
    partner_type_name,
    vendor_user_uploads,
)
from app.agents.contact_us.nodes import StepBlocked, StepFailed, StepResult, tracked_step
from app.agents.contact_us.state import ContactUsChatState
from app.agents.contact_us.tools import (
    get_accounts_by_email,
    get_contact_us_assigned_users,
    get_contact_us_requests,
    get_contractor_subcategories,
    get_default_role_id,
    link_vendor_to_provider,
    mark_contact_request_converted,
    save_contractor_subcategories,
    search_service_providers,
)
from app.core.logging import get_logger
from app.integrations.bootog import AccountApi, BootogApiError, RoleNotFoundError
from app.integrations.bootog.accounts import is_guid
from app.integrations.bootog.client import CSRF_REJECTED
from app.integrations.bootog.contact_us import CRM_COMPLETED_STATUS, name_tokens

logger = get_logger(__name__)

_OPTION_LIMIT = 10
_ASSIGNEE_CACHE_SECONDS = 600


def _emitter(config: RunnableConfig):
    return config["configurable"]["emitter"]


def conversion_of(state: ContactUsChatState) -> dict[str, Any]:
    return dict(state.get("conversion") or {})


def active_type(state: ContactUsChatState) -> ConversionType | None:
    return conversion_type(conversion_of(state).get("type"))


def draft_type(state: ContactUsChatState) -> ConversionType:
    """The form the draft follows: the conversion's type; the Partner form while a partner's
    company type is still being chosen; Customer by default."""
    ctype = active_type(state)
    if ctype is None and state.get("operation") == "create_partner":
        return PARTNER_FORM
    return ctype or CUSTOMER


# --- partners without a Contact Us request ------------------------------------------------
# The host's create-partner screen is opened from many places without a Contact Us request
# (provider management, file management, reports): its contactUsId is optional and the
# request is marked CRM Completed only when there is one. A partner conversion started with
# no request selected and none named is such a standalone conversion: no request_id, no
# update_request_status step, and its card is addressed by `form_id` (an internal token,
# never sent to BOOTOG). Customer and B2B conversions always need a request.

STANDALONE_FORM_PREFIX = "new-partner:"


def standalone(state: ContactUsChatState) -> bool:
    return not state.get("contact_request") and bool(conversion_of(state).get("standalone"))


def conversion_record(state: ContactUsChatState) -> dict[str, Any] | None:
    """The data the conversion starts from: the selected request, or nothing (an empty
    record) for a standalone partner; None when there is neither."""
    record = state.get("contact_request")
    if record:
        return record
    return {} if standalone(state) else None


def conversion_ref(state: ContactUsChatState) -> str | None:
    """What the conversion and its card belong to: the request id, or the standalone form id."""
    record = state.get("contact_request")
    if record:
        return record.get("id")
    return conversion_of(state).get("form_id") if standalone(state) else None


def conversion_steps(ctype: ConversionType, conv: dict[str, Any]) -> list[str]:
    """Execution steps of this conversion: without a request there is no status to update."""
    steps = ctype.steps()
    return [s for s in steps if s != "update_request_status"] if conv.get("standalone") else steps


def conversion_draft(state: ContactUsChatState) -> dict[str, Any] | None:
    record = conversion_record(state)
    if record is None:
        return None
    return build_draft(draft_type(state), record, state.get("user_customer_fields") or {})


def conversion_uploads(state: ContactUsChatState) -> dict[str, dict[str, Any]]:
    """Images attached to the selected request's form, when its type takes them."""
    held = state.get("conversion_files") or {}
    if not draft_type(state).uploads or held.get("request_id") != conversion_ref(state):
        return {}
    return {k: v for k, v in (held.get("files") or {}).items() if k in draft_type(state).uploads}


def conversion_fingerprint(state: ContactUsChatState) -> str:
    conv = conversion_of(state)
    parts: list[Any] = [
        conv.get("type"),
        conv.get("request_id"),
        conv.get("role_id"),
        (conv.get("target_provider") or {}).get("id"),
        sorted(s["id"] for s in conv.get("subcategories") or []),
        state.get("customer_payload") or {},
    ]
    uploads = sorted((k, f["sha"]) for k, f in conversion_uploads(state).items())
    if uploads:  # a changed image needs a new confirmation; none leaves the digest as it was
        parts.append(uploads)
    return fingerprint(*parts)


def _answered(state: ContactUsChatState, question: str) -> bool:
    return (state.get("answered_question") or {}).get("type") == question


def _norm(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def _waiting(update: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    return {**update, "pending_question": question, "awaiting_user_input": True}


def _failure(conv: dict[str, Any], step: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"conversion": {**conv, "failed": {"step": step, "message": message, **extra}}}


def _completed(conv: dict[str, Any], step: str, **changes: Any) -> dict[str, Any]:
    done = [s for s in conv.get("completed") or [] if s != step] + [step]
    return {**conv, **changes, "completed": done, "failed": None}


# --- setup -------------------------------------------------------------------------------


@tracked_step("start_conversion", "Preparing the conversion", "Unable to prepare the conversion.")
async def start_conversion(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    record = state.get("contact_request")
    request_id = record.get("id") if record else None
    conv = conversion_of(state)
    turn = state.get("turn_conversion") or {}
    operation = state.get("operation")
    update: dict[str, Any] = {}

    if record is None:
        # A partner without a Contact Us request. A finished one is not continued: the next
        # partner is a new conversion with its own form id.
        done = conversion_type(conv.get("type")) and set(
            conversion_steps(conversion_type(conv["type"]), conv)
        ) <= set(conv.get("completed") or [])
        if not conv.get("standalone") or conv.get("request_id") or done:
            # A card restored after the conversation's state was lost keeps its own form id.
            form_id = str(turn.get("form_id") or "")
            if not form_id.startswith(STANDALONE_FORM_PREFIX) or done:
                form_id = STANDALONE_FORM_PREFIX + uuid.uuid4().hex
            conv = {"request_id": None, "standalone": True, "form_id": form_id}
    elif conv.get("request_id") != request_id:
        previous = conv
        conv = {"request_id": request_id}
        if previous.get("standalone") and not previous.get("create_attempted"):
            # A partner started without a request is now linked to one: the company type,
            # provider and images chosen so far still apply (the details stay as well).
            conv.update({k: previous[k] for k in ("type", "target_provider", "provider_options",
                                                  "provider_options_truncated") if previous.get(k)})
            held = state.get("conversion_files") or {}
            if held.get("request_id") and held["request_id"] == previous.get("form_id"):
                update["conversion_files"] = {**held, "request_id": request_id}
    conv["blocked"] = None
    request_ref = request_id or conv.get("form_id")
    current = conversion_type(conv.get("type"))
    finished = bool(current) and set(conversion_steps(current, conv)) <= set(conv.get("completed") or [])
    failed = conv.get("failed") or {}
    # An account may exist for an unfinished conversion: creation completed, or its answer
    # was lost (timeout / 5xx). Switching type then could create a second account.
    in_flight = not finished and (
        "create_account" in (conv.get("completed") or [])
        or (failed.get("step") == "create_account" and bool(failed.get("outcome_unknown")))
    )
    # A finished conversion's partner subtype is not carried into a new one.
    wanted = _wanted_type(state, conv, turn, operation, retain_partner=not finished)

    if wanted and (state.get("created_customers") or {}).get(converted_key(request_ref, wanted)):
        conv["blocked"] = "already_converted"
        conv["blocked_type"] = wanted
        return StepResult(f"Already converted to {conversion_type(wanted).label}", {**update, "conversion": conv})

    if in_flight and wanted != conv.get("type"):
        conv["blocked"] = "type_locked"
        return StepResult(f"Already started as {current.label}", {**update, "conversion": conv})

    new = conversion_type(wanted)
    if finished or wanted != conv.get("type"):
        conv = _switched(state, conv, wanted)
        message = f"Converting to {new.label}" if new else "Account type needed"
        if current and not finished:
            message = f"Switched from {current.label} to {new.label if new else 'another type'}"
    else:
        message = f"Converting to {current.label}" if current else "Account type needed"
    # A provider or categories named before they can be used (type still being asked,
    # provider not chosen yet) are kept until the step that needs them runs.
    if turn.get("target_provider_name"):
        conv["pending_provider_name"] = turn["target_provider_name"]
    if turn.get("subcategory_names"):
        conv["pending_subcategory_names"] = turn["subcategory_names"]
    # "Let me check the details first": hold automatic creation until the user explicitly
    # asks for it (the summary's "Create customer", or "create it" answering it).
    if turn.get("review_first"):
        conv["review_hold"] = True
    if state.get("explicit_create"):
        conv["review_hold"] = False
    if conv.get("standalone"):
        message += " — not linked to a Contact Us request"
    return StepResult(message, {**update, "conversion": conv})


def _wanted_type(
    state: ContactUsChatState, conv: dict[str, Any], turn: dict[str, Any], operation: str | None, retain_partner: bool
) -> str | None:
    """The conversion type this turn asks for. "Partner" alone keeps an already chosen
    partner subtype, but never turns a Customer/B2B conversion into one: it asks. A
    conversion of unnamed kind ("proceed") keeps the type in progress, or asks."""
    if operation == "convert_request":
        pos = turn.get("option_position") if _answered(state, "choose_conversion_type") else None
        if pos and 1 <= pos <= len(ALL_TYPES):
            return ALL_TYPES[pos - 1].key
        if turn.get("partner_type") in CONVERSION_TYPES:
            return turn["partner_type"]
        return conv.get("type") if retain_partner else None
    if operation != "create_partner":
        return OPERATION_TYPE.get(operation or "")
    partner_keys = [t.key for t in PARTNER_TYPES]
    pos = turn.get("option_position") if _answered(state, "choose_partner_type") else None
    if turn.get("partner_type") in partner_keys:
        return turn["partner_type"]
    if pos and 1 <= pos <= len(PARTNER_TYPES):
        return PARTNER_TYPES[pos - 1].key
    if retain_partner and conv.get("type") in partner_keys:
        return conv["type"]
    return None


def _switched(state: ContactUsChatState, conv: dict[str, Any], wanted: str | None) -> dict[str, Any]:
    """A new conversion of the same request. Contact and address details the user gave
    stay in `user_customer_fields` (shared by every form); everything type-specific is
    reset — role, account check, confirmation, categories, failures. A service provider
    or company name from the previous conversion is carried over only where the new type
    uses it, and is flagged so the confirmation calls it out."""
    new = conversion_type(wanted)
    fresh: dict[str, Any] = {"request_id": conv["request_id"], "type": wanted}
    if conv.get("standalone"):  # still the same partner form, still without a request
        fresh.update(standalone=True, form_id=conv["form_id"])
    carried: dict[str, str] = {}
    previous_provider = conv.get("target_provider")
    if previous_provider and (new is None or new.needs_target_provider):
        fresh["target_provider"] = previous_provider
        carried["Service provider"] = previous_provider["name"]
    company = (state.get("user_customer_fields") or {}).get("companyName")
    given_now = "companyName" in (state.get("turn_field_updates") or {})
    if company and not given_now and (new is None or "companyName" in new.fields):
        carried["Company Name"] = company
    fresh["carried_over"] = carried
    return fresh


def converted_key(request_id: str | None, type_key: str | None) -> str:
    """`created_customers` key: one finished conversion per request AND type."""
    return f"{request_id}:{type_key}"


async def choose_conversion_type(state: ContactUsChatState, config: RunnableConfig) -> dict[str, Any]:
    if state.get("operation") == "create_partner":
        question, options = "choose_partner_type", [partner_type_name(t) for t in PARTNER_TYPES]
        message = "Waiting for the partner company type"
    else:  # the user asked to convert without saying into what
        question, options = "choose_conversion_type", [t.label for t in ALL_TYPES]
        message = "Waiting for the account type"
    await _emitter(config).step("choose_conversion_type", "waiting", message)
    return _waiting({"current_step": "choose_conversion_type"}, {"type": question, "options": options})


# --- providers ----------------------------------------------------------------------------


async def find_providers(config: RunnableConfig, name: str | None, location: str | None) -> tuple[list[dict], bool]:
    """Name: Bootog's prefix search first; if nothing starts with it, a contains/word match
    over the full (bounded) list. Location: SearchProviders has no location parameter, so
    addresses are matched over the full (bounded) list."""
    name = (name or "").strip()[:100] or None
    result = await search_service_providers.ainvoke({"provider_name": name}, config=config)
    providers, truncated = result["providers"], result["truncated"]
    if name and not providers:
        result = await search_service_providers.ainvoke({"provider_name": None}, config=config)
        tokens = name_tokens(name)
        providers = [p for p in result["providers"] if tokens and all(t in _norm(p["name"]) for t in tokens)]
        truncated = result["truncated"]
    if location and location.strip():
        place = _norm(location)
        providers = [p for p in providers if place in _norm(p.get("address"))]
    return providers, truncated


def _provider_option(p: dict[str, Any]) -> dict[str, Any]:
    return {"id": p["id"], "name": p["name"], "address": p.get("address")}


@tracked_step("resolve_target_provider", "Finding the service provider", "Unable to look up service providers.")
async def resolve_target_provider(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    turn = state.get("turn_conversion") or {}
    candidates = conv.get("provider_candidates") or []
    pos = turn.get("option_position") if _answered(state, "choose_provider") else None
    name = (turn.get("target_provider_name") or conv.get("pending_provider_name") or "").strip()
    picked_id = turn.get("target_provider_id")
    conv["pending_provider_name"] = None
    if not conv.get("provider_options"):
        # The details card's "Select Service Provider" list (SearchProviders, every page,
        # bounded), loaded once per conversion like the host's B2B / Partner screens.
        everything = await search_service_providers.ainvoke({"all_pages": True}, config=config)
        conv["provider_options"] = [_provider_option(p) for p in everything["providers"]]
        conv["provider_options_truncated"] = everything["truncated"]
    options_all = conv["provider_options"]

    chosen = None
    if picked_id:
        # Picked in the card: only an id from the list BOOTOG returned is accepted.
        chosen = next((p for p in [*options_all, *candidates] if p["id"] == picked_id), None)
        if chosen is None:
            conv["provider_candidates"] = options_all[:_OPTION_LIMIT]
            return StepResult(
                "The selected service provider is not in BOOTOG's provider list — asking again",
                _waiting({"conversion": conv}, {"type": "choose_provider", "status": "invalid_choice"}),
            )
    elif pos and candidates and 1 <= pos <= len(candidates):
        chosen = candidates[pos - 1]
    elif name:
        found, truncated = await find_providers(config, name, None)
        exact = [p for p in found if _norm(p["name"]) == _norm(name)]
        if len(exact) == 1 or len(found) == 1:
            chosen = (exact or found)[0]
        else:
            options = (exact or found)[:_OPTION_LIMIT]
            conv["provider_candidates"] = [_provider_option(p) for p in options]
            status = "ambiguous" if options else "not_found"
            question = {"type": "choose_provider", "status": status, "requested": name, "more": len(found) > len(options)}
            return StepResult(
                f"{len(found)} providers match “{name}” — asking which one" if options else f"No provider matches “{name}”",
                _waiting({"conversion": conv}, question),
                data={"matches": len(found), "truncated": truncated},
            )
    if chosen is None:
        # Nothing named yet: ask (the card offers every provider; the chat the first few).
        conv["provider_candidates"] = options_all[:_OPTION_LIMIT]
        return StepResult(
            "Asking which service provider to use",
            _waiting({"conversion": conv}, {"type": "choose_provider", "status": "needed"}),
        )

    changed = (conv.get("target_provider") or {}).get("id") != chosen["id"]
    conv["target_provider"] = {"id": chosen["id"], "name": chosen["name"]}
    conv["provider_candidates"] = None
    conv["carried_over"] = {k: v for k, v in (conv.get("carried_over") or {}).items() if k != "Service provider"}
    if changed and conversion_type(conv.get("type")) and conversion_type(conv["type"]).needs_subcategories:
        # Sub-categories are provider-specific: the chosen ones and the numbered options
        # offered for the previous provider no longer apply. The same provider keeps them.
        conv["subcategories"] = []
        conv["subcategory_candidates"] = None
    return StepResult(f"Using {chosen['name']}", {"conversion": conv}, data={"providerName": chosen["name"]})


@tracked_step("search_providers", "Searching service providers", "Unable to search service providers.")
async def search_providers(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    query = state.get("provider_query") or {}
    providers, truncated = await find_providers(config, query.get("name"), query.get("location"))
    results = {
        "query": query,
        "total": len(providers),
        "truncated": truncated,
        "providers": providers[:25],
    }
    message = f"Found {len(providers)} provider{'s' if len(providers) != 1 else ''}"
    if truncated:
        message += " (stopped at the page limit)"
    return StepResult(message, {"provider_results": results}, data={"found": len(providers)})


# --- assignees ----------------------------------------------------------------------------


async def assigned_users(state: ContactUsChatState, config: RunnableConfig) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """GetContactUsAssignedUser for the agent's module, cached 10 minutes: (users, state update)."""
    cache = state.get("assigned_users_cache") or {}
    if cache.get("module") == state.get("module_filter") and time.time() - cache.get("at", 0) < _ASSIGNEE_CACHE_SECONDS:
        return cache.get("users") or [], {}
    users = await get_contact_us_assigned_users.ainvoke({"module_filter": state.get("module_filter")}, config=config)
    return users, {"assigned_users_cache": {"at": time.time(), "module": state.get("module_filter"), "users": users}}


@tracked_step("resolve_assigned_user", "Finding the assignee", "Unable to look up assigned users.")
async def resolve_assigned_user(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    """Assignee name (or "me") -> assignedUserId from GetContactUsAssignedUser. One match is
    used and kept for later turns; several (or none) are offered to choose from, nothing is
    listed and the filters stay as they were — never a list without the asked-for filter."""
    filters = dict(state.get("request_filters") or {})
    users, update = await assigned_users(state, config)
    if filters.get("assigned_to_me"):
        # The signed-in user's own BOOTOG id (from the host's session), never a name match.
        me = state.get("viewer_user_id")
        found = [u for u in users if me and u["id"] == me]
        requested = "you"
        if not me:
            reason = "your BOOTOG user could not be identified from this session"
        else:
            reason = "no requests are assigned to you"
    else:
        requested = filters.get("assigned_user_name")
        wanted = _norm(requested)
        exact = [u for u in users if _norm(u["name"]) == wanted]
        tokens = name_tokens(wanted)
        found = exact or [u for u in users if tokens and all(t in _norm(u["name"]) for t in tokens)]
        reason = None
    if len(found) == 1:
        user = found[0]
        filters.update(assigned_user_ids=[user["id"]], assigned_user_name=None, assigned_to_me=False)
        active = {**(state.get("active_filters") or {}), "assigned_user_id": user["id"], "assigned_user_name": user["name"]}
        return StepResult(
            f"Requests assigned to {user['name']}",
            {**update, "request_filters": filters, "active_filters": active, "assignee_candidates": None},
        )
    candidates = [] if filters.get("assigned_to_me") else (found or users)[:_OPTION_LIMIT]
    note = {
        "assignee": {
            "status": "ambiguous" if len(found) > 1 else "not_found",
            "requested": requested,
            "reason": reason,
            "candidates": [u["name"] for u in candidates],
        }
    }
    return StepResult(
        f"No unique assignee matched “{requested}”",
        {
            **update,
            "last_tool_result": note,
            # This turn's change is undone: the list keeps the filters it had.
            "active_filters": state.get("filters_before") if state.get("filters_before") is not None else state.get("active_filters"),
            "filter_problems": [*(state.get("filter_problems") or []), f"assigned user '{requested}'"],
            "assignee_candidates": candidates or None,
            **({"pending_question": {"type": "choose_assignee", "options": [u["name"] for u in candidates]},
                "awaiting_user_input": True} if candidates else {}),
        },
    )


# --- checks ---------------------------------------------------------------------------------


def is_type_role(role: dict[str, Any], ctype: ConversionType, role_id: str | None) -> bool:
    """The role this conversion assigns, as AddVendorUser's duplicate check sees it: the
    same RoleId as the resolved default role. Other providers' copies of a role share its
    name but not its id, so they don't count. By name only when an id is unknown (that
    can only over-block, never let a duplicate through)."""
    if role_id and role.get("roleId"):
        return str(role["roleId"]).casefold() == role_id.casefold()
    return str(role.get("roleName") or "").casefold() == ctype.role_name.casefold()


@tracked_step("check_existing_account", "Checking for an existing account", "Unable to check for an existing account.")
async def check_existing_account(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    ctype = active_type(state)
    email = (state.get("customer_payload") or {}).get("email") or ""
    failed = conv.get("failed") or {}
    if failed.get("step") == "create_account" and failed.get("outcome_unknown") and failed.get("email", email) != email:
        conv = await _settle_earlier_attempt(config, conv, ctype, email)
    try:
        accounts = await lookup_accounts(config, email)
    except BootogApiError as exc:
        # Not knowing is never "no account": stop before any creation.
        conv["account"] = {"email": email, "verified": False, "error": exc.message}
        message = f"I couldn't verify {email} because BOOTOG returned an error, so nothing was created. {exc.message}"
        raise StepFailed(message, {"conversion": conv}) from exc

    roles = [r for a in accounts for r in a["roles"]]
    with_role = [r for r in roles if is_type_role(r, ctype, conv.get("role_id"))]
    removed = [r for a in accounts for r in a.get("removedRoles") or [] if is_type_role(r, ctype, conv.get("role_id"))]
    conv["account"] = {
        "email": email,
        "verified": True,
        "exists": bool(accounts),
        "has_role": bool(with_role),
        # Only a removed (isDelete) copy of the role: BOOTOG refuses to add it again.
        "role_removed": bool(removed) and not with_role,
        "roles": sorted({r["roleName"] for r in roles})[:10],
    }
    failed = conv.get("failed") or {}
    # Only an attempt for THIS email can have created this email's account.
    lost_answer = (
        failed.get("step") == "create_account" and failed.get("outcome_unknown") and failed.get("email", email) == email
    )
    if with_role and lost_answer and ctype.auto_create and not failed.get("role_absent_before"):
        # Customer: a Customer role found after an attempt whose answer was lost, when it
        # was not known to be absent before that attempt, is reported as an existing
        # Customer — never claimed as created by it, and the request is not marked
        # converted. (Known absent before: that attempt wrote it; adopted below.)
        conv.update(blocked="existing_role", failed=None)
        conv["account"]["found_after_failed_attempt"] = True
        return StepResult(f"{email}: Customer already exists — not created again", {"conversion": conv})
    if with_role and lost_answer:
        # An earlier attempt whose answer was lost did create the account: adopt it.
        vendor = next((r["providerId"] for r in with_role if is_guid(r.get("providerId"))), None)
        user = next((a["userId"] for a in accounts if any(r in with_role for r in a["roles"])), None)
        conv = _completed(conv, "create_account", created_vendor_id=conv.get("created_vendor_id") or vendor, user_id=user)
        return StepResult("The earlier attempt did create the account — resuming", {"conversion": conv})
    if with_role:
        conv["blocked"] = "existing_role"
        return StepResult(f"{email} is already registered as {ctype.role_name}", {"conversion": conv})
    if removed:
        # AddVendorUser matches UserRoles without IsDelete: sending it can only be refused.
        conv["blocked"] = "existing_role"
        return StepResult(f"{email} has a removed {ctype.role_name} role — not sent", {"conversion": conv})
    message = f"Existing login found (roles: {', '.join(conv['account']['roles']) or 'none'})" if accounts else "No existing account"
    return StepResult(message, {"conversion": conv})


async def _settle_earlier_attempt(
    config: RunnableConfig, conv: dict[str, Any], ctype: ConversionType, email: str
) -> dict[str, Any]:
    """An earlier creation for ANOTHER email of this request got no answer, so it may have
    created this request's account. BOOTOG is asked about that email before anything is
    done for the new one: no role -> that attempt left nothing, carry on; the role, or no
    answer -> stop, or one request would end up with two accounts."""
    failed = conv["failed"]
    earlier = failed["email"]
    try:
        accounts = await lookup_accounts(config, earlier)
    except BootogApiError as exc:
        message = (
            f"An earlier attempt to create the account for {earlier} got no answer, and BOOTOG could not be asked "
            f"about it now, so nothing was created for {email}. {exc.message}"
        )
        conv["failed"] = {**failed, "message": message}
        raise StepFailed(message, {"conversion": conv}) from exc
    if any(is_type_role(r, ctype, conv.get("role_id")) for a in accounts for r in a["roles"]):
        message = (
            f"An earlier attempt to create the account for {earlier} got no answer, and {earlier} now has the "
            f"{ctype.role_name} role: that attempt may have created this request's account. Nothing was created "
            f"for {email}, so the request does not get a second account. Switch the email back to {earlier} to "
            "verify and finish it, or have an administrator check that account."
        )
        conv["failed"] = {**failed, "message": message, "earlier_email_has_role": True}
        raise StepFailed(message, {"conversion": conv})
    conv["failed"] = None  # confirmed: that attempt left no account with the role
    return conv


@tracked_step("resolve_role", "Looking up the role", "Unable to look up the role.")
async def resolve_role(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    ctype = active_type(state)
    try:
        role_id = await get_default_role_id.ainvoke({"role_name": ctype.role_name}, config=config)
    except BootogApiError as exc:  # RoleNotFoundError, HTTP errors, timeouts
        # Without a verified role nothing is created; the card shows why.
        conv["role_unverified"] = exc.message
        raise StepFailed(f"The {ctype.role_name} role could not be verified, so nothing was created. {exc.message}",
                         {"conversion": conv}) from exc
    conv.update(role_id=role_id, role_name=ctype.role_name, role_unverified=None)
    return StepResult(f"Role {ctype.role_name} resolved", {"conversion": conv})


@tracked_step("select_subcategories", "Matching service categories", "Unable to load service categories.")
async def select_subcategories(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    turn = state.get("turn_conversion") or {}
    provider_id = (conv.get("target_provider") or {}).get("id")
    page = await get_contractor_subcategories.ainvoke({"target_provider_id": provider_id}, config=config)
    items = page["items"]
    selected = {s["id"]: s for s in conv.get("subcategories") or []}

    offered = conv.get("subcategory_candidates") or []
    if _answered(state, "choose_subcategory"):
        picks = [*(turn.get("option_positions") or []), *([turn["option_position"]] if turn.get("option_position") else [])]
        for pos in picks:
            if 1 <= pos <= len(offered):
                selected.setdefault(offered[pos - 1]["id"], {"id": offered[pos - 1]["id"], "name": offered[pos - 1]["name"]})

    # Only an exact name selects a category by itself. A partial match ("plumbing" vs
    # "Plumbing for a new Addition or Remodel - INSTALL") is offered for the user to pick,
    # even when it is the only one.
    names = turn.get("subcategory_names") or conv.get("pending_subcategory_names") or []
    unmatched: list[str] = []
    partial: dict[str, list[dict[str, str]]] = {}
    for wanted in names:
        key = _norm(wanted)
        exact = [i for i in items if _norm(i["name"]) == key]
        if len(exact) == 1:
            selected.setdefault(exact[0]["id"], {"id": exact[0]["id"], "name": exact[0]["name"]})
            continue
        words = key.split()
        loose = exact or [i for i in items if words and all(w in _norm(i["name"]) for w in words)]
        if loose:
            partial[wanted] = [{"id": i["id"], "name": i["name"]} for i in loose]
        else:
            unmatched.append(wanted)
    conv["pending_subcategory_names"] = None

    conv["subcategories"] = list(selected.values())
    note = {
        "subcategories": {
            "selected": [s["name"] for s in conv["subcategories"]],
            "not_found": unmatched,
            "need_a_choice": list(partial),
            "available": page["total"],
            "truncated": page["truncated"],
        }
    }
    if conv["subcategories"] and not partial:
        conv["subcategory_candidates"] = None
        chosen = ", ".join(s["name"] for s in conv["subcategories"])
        return StepResult(f"Service categories: {chosen}", {"conversion": conv, "last_tool_result": note})

    # Numbered options: the partial matches first (per term), else a first sample.
    options: list[dict[str, str]] = []
    for matches in partial.values():
        for item in matches:
            if item["id"] not in selected and item not in options:
                options.append(item)
    options = (options or [{"id": i["id"], "name": i["name"]} for i in items if i["id"] not in selected])[:15]
    conv["subcategory_candidates"] = options
    question = {
        "type": "choose_subcategory",
        "for_terms": list(partial),
        "options": [o["name"] for o in options],
        "more_matches": sum(len(v) for v in partial.values()) > len(options),
        "available": page["total"],
    }
    return StepResult(
        "Asking which service categories the contractor offers",
        _waiting({"conversion": conv, "last_tool_result": note}, question),
    )


async def present_details(state: ContactUsChatState, config: RunnableConfig) -> dict[str, Any]:
    """Auto-created types (Customer): the details are complete and valid, but this turn did
    not authorise creation (the user asked to review, corrected a complete draft, or the
    last attempt failed). Show the editable summary and wait for an explicit "create"."""
    conv = conversion_of(state)
    conv["awaiting_fingerprint"] = conversion_fingerprint(state)
    reason = (
        "review_requested" if conv.get("review_hold")
        else "after_failure" if conv.get("failed")
        else "details_changed"
    )
    await _emitter(config).step("present_details", "waiting", "Details ready — waiting for the go-ahead to create")
    return _waiting(
        {"conversion": conv, "current_step": "present_details"}, {"type": "ready_to_create", "reason": reason}
    )


async def ask_edit_fields(state: ContactUsChatState, config: RunnableConfig) -> dict[str, Any]:
    """"I need to change my address" without a new value: ask which field(s), offering them."""
    fields = (state.get("edit_request") or {}).get("fields") or []
    await _emitter(config).step("ask_edit_fields", "waiting", "Waiting for the new value(s)")
    return _waiting({"current_step": "ask_edit_fields"}, {"type": "edit_fields", "fields": fields})


async def confirm_conversion(state: ContactUsChatState, config: RunnableConfig) -> dict[str, Any]:
    conv = conversion_of(state)
    conv["awaiting_fingerprint"] = conversion_fingerprint(state)
    await _emitter(config).step("confirm_conversion", "waiting", f"Waiting for confirmation: {active_type(state).label}")
    return _waiting({"conversion": conv, "current_step": "confirm_conversion"}, {"type": "confirm_conversion"})


# --- execution -----------------------------------------------------------------------------


@tracked_step("create_account", "Creating the account", "Account creation failed.")
async def create_account(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    ctype = active_type(state)
    if "create_account" in (conv.get("completed") or []):
        return StepResult("Account already created", {"conversion": conv})
    # Recorded before the call: if the answer is lost, a retry checks Bootog first
    # (check_existing_account) instead of creating blindly.
    conv["create_attempted"] = True
    sent = conversion_fingerprint(state)  # exactly these details; a rejection never resends them
    email = (state.get("customer_payload") or {}).get("email")
    checked = conv.get("account") or {}
    # This turn's check of exactly this email found no account with the role.
    role_absent_before = bool(checked.get("verified") and not checked.get("has_role") and checked.get("email") == email)
    form = build_vendor_user_form(
        ctype, state.get("customer_payload") or {}, conv["role_id"], conversion_record(state) or {}, generate_password()
    )
    uploads = conversion_uploads(state)
    upload_fields, files = vendor_user_uploads(uploads)
    form.update(upload_fields)
    try:
        vendor_id = await AccountApi(config["configurable"]["bootog_client"]).add_vendor_user(form, files)
    except BootogApiError as exc:
        # `failed: true` / a 4xx is a definitive answer; a timeout, lost connection or 5xx
        # is not — the account may exist, so the next attempt checks Bootog before creating.
        unknown = exc.outcome_unknown
        kind = "unknown" if unknown else "access" if exc.status_code in (401, 403) else "rejected"
        reconciled = None
        # AddVendorUser never rolls back: any answer other than its duplicate refusal (sent
        # before it writes) or a refusal that never reached it may follow the role insert.
        may_have_written = exc.code not in ("EMAIL_EXISTS", CSRF_REJECTED) and kind != "access"
        if may_have_written and ctype.auto_create:
            # Reconcile now (Customer): BOOTOG may have written the role before failing.
            reconciled = await _role_after_failure(config, conv, ctype)
            if reconciled == "found" and role_absent_before:
                # The role was not there just before this call and is there now: this
                # attempt most likely wrote it before failing. It is recorded as THIS
                # conversion's account (never as an existing customer) so nothing — and no
                # other email — creates another, and the request is not marked CRM Completed.
                message = (
                    f"BOOTOG returned an error while creating the account, but {email} now has the "
                    f"{ctype.role_name} role, which it did not have just before: the account was most likely "
                    "created by this attempt and may be incomplete. CRM completion is not verified — the "
                    "request was not marked CRM Completed."
                )
                conv = _completed(conv, "create_account", created_vendor_id=conv.get("created_vendor_id"))
                conv["failed"] = {"step": "create_account", "kind": "created_with_error", "message": message, "code": exc.code}
                _log_creation(state, "created_with_error", exc)
                raise StepFailed(message, {"conversion": conv}) from exc
            if reconciled == "found":
                conv.update(blocked="existing_role", failed=None)
                _log_creation(state, "existing_role", exc)
                raise StepBlocked(
                    f"A {ctype.label} account already exists for this email. No duplicate account was created.",
                    {"conversion": conv},
                ) from exc
            if reconciled == "absent" and unknown:
                unknown, kind = False, "server_error"  # confirmed: no account to resume
            elif reconciled == "unverified" and not unknown:
                # A definitive "failed", but whether it wrote the role can't be checked:
                # treated as unknown, so the next attempt checks BOOTOG before sending.
                unknown, kind = True, "unknown"
        message = f"The account could not be created. {exc.message}"
        failure = _failure(
            conv, "create_account", message,
            outcome_unknown=unknown, kind=kind, code=exc.code, fingerprint=sent,
            reconciled=reconciled, role_absent_before=role_absent_before, email=email,
        )
        _log_creation(state, f"failed:{kind}", exc, reconciled)
        raise StepFailed(message, failure) from exc
    finally:
        form.clear()  # drop the password as soon as the request is done
    conv = _completed(conv, "create_account", created_vendor_id=vendor_id)
    _log_creation(state, "created")
    update: dict[str, Any] = {"conversion": conv}
    if uploads:
        # Sent with the account: only what the card shows is kept, never the image again.
        held = state["conversion_files"]
        update["conversion_files"] = {
            **held, "files": {k: {f: v for f, v in u.items() if f != "data"} for k, u in uploads.items()},
        }
    return StepResult(f"{ctype.label} account created", update)


def _log_creation(state: ContactUsChatState, outcome: str, exc: BootogApiError | None = None, reconciled: str | None = None) -> None:
    """Request id, stage and outcome for the backend team; never the email, form or password."""
    logger.info(
        "[CONTACT_US_AGENT] run=%s request=%s stage=create_account outcome=%s http=%s code=%s reconciled=%s",
        state.get("run_id"), conversion_of(state).get("request_id"), outcome,
        getattr(exc, "status_code", None), getattr(exc, "code", None), reconciled,
    )


async def lookup_accounts(config: RunnableConfig, email: str) -> list[dict[str, Any]]:
    """GET UserDetail/GetUserRolesByEmailId through the shared client (active roles only).
    BOOTOG matches the email case-sensitively, so the lower-case form is checked too.
    Raises on any failure or malformed answer — never returns "no account" for it."""
    accounts: list[dict[str, Any]] = []
    for candidate in dict.fromkeys([email, email.casefold()]):
        accounts += await get_accounts_by_email.ainvoke({"email": candidate}, config=config)
    return accounts


async def _role_after_failure(config: RunnableConfig, conv: dict[str, Any], ctype: ConversionType) -> str:
    """After an ambiguous creation answer: does the email now have the role?
    -> "found" | "absent" | "unverified" (the lookup itself failed: stay uncertain)."""
    email = (conv.get("account") or {}).get("email") or ""
    try:
        accounts = await lookup_accounts(config, email)
    except BootogApiError:
        return "unverified"
    found = any(is_type_role(r, ctype, conv.get("role_id")) for a in accounts for r in a["roles"])
    if found:
        conv["account"] = {
            **(conv.get("account") or {}),
            "verified": True, "exists": True, "has_role": True, "found_after_failed_attempt": True,
        }
        return "found"
    return "absent"


@tracked_step("verify_account", "Verifying the account", "Account verification failed.")
async def verify_account(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    ctype = active_type(state)
    email = (conv.get("account") or {}).get("email") or (state.get("customer_payload") or {}).get("email")
    accounts = await lookup_accounts(config, email)  # same case handling as the pre-check
    role_id = conv.get("role_id")
    match = next((a for a in accounts if any(is_type_role(r, ctype, role_id) for r in a["roles"])), None)
    if match is None:
        message = f"Bootog reported the account as created, but {email} doesn't have the {ctype.role_name} role yet."
        raise StepFailed(message, _failure(conv, "verify_account", message))
    vendor = conv.get("created_vendor_id") or next(
        (r["providerId"] for r in match["roles"] if is_type_role(r, ctype, role_id) and is_guid(r.get("providerId"))),
        None,
    )
    if ctype.needs_target_provider and not vendor:
        message = "The account exists, but Bootog returned no company id, so it can't be linked to the provider."
        raise StepFailed(message, _failure(conv, "verify_account", message))
    conv = _completed(conv, "verify_account", user_id=match.get("userId"), created_vendor_id=vendor)
    return StepResult("Account and role confirmed in BOOTOG", {"conversion": conv})


@tracked_step("link_provider_vendor", "Linking to the service provider", "Linking to the service provider failed.")
async def link_provider_vendor(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    provider = conv.get("target_provider") or {}
    try:
        await link_vendor_to_provider.ainvoke(
            {"vendor_id": conv["created_vendor_id"], "target_provider_id": provider["id"]}, config=config
        )
    except BootogApiError as exc:
        message = f"Linking to {provider.get('name')} failed. {exc.message}"
        raise StepFailed(message, _failure(conv, "link_provider_vendor", message)) from exc
    return StepResult(f"Linked to {provider.get('name')}", {"conversion": _completed(conv, "link_provider_vendor")})


@tracked_step("save_subcategories", "Saving service categories", "Saving service categories failed.")
async def save_subcategories(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    provider = conv.get("target_provider") or {}
    try:
        await save_contractor_subcategories.ainvoke(
            {"subcategory_ids": [s["id"] for s in conv.get("subcategories") or []], "target_provider_id": provider["id"]},
            config=config,
        )
    except BootogApiError as exc:
        message = f"Saving the service categories failed. {exc.message}"
        raise StepFailed(message, _failure(conv, "save_subcategories", message)) from exc
    return StepResult("Service categories saved", {"conversion": _completed(conv, "save_subcategories")})


@tracked_step("update_request_status", "Marking the request CRM Completed", "Updating the request status failed.")
async def update_request_status(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    if not conv.get("request_id"):  # not a step of a standalone partner; never sent without a request
        raise StepFailed("There is no Contact Us request to update.", {"conversion": conv})
    try:
        await mark_contact_request_converted.ainvoke({"request_id": conv["request_id"]}, config=config)
    except BootogApiError as exc:
        message = f"The request status could not be updated. {exc.message}"
        raise StepFailed(message, _failure(conv, "update_request_status", message)) from exc
    # bulk-update-file answers "Update completed successfully." whatever it changed (it
    # ignores the affected-row count): only the request read back proves CRM Completed.
    try:
        page = await get_contact_us_requests.ainvoke(
            {"request_id": conv["request_id"], "provider_id": state.get("provider_id"), "module_filter": state.get("module_filter")},
            config=config,
        )
    except BootogApiError as exc:
        message = (
            "BOOTOG accepted the status update, but the request could not be read back, so CRM Completed is "
            f"not verified. {exc.message}"
        )
        raise StepFailed(message, _failure(conv, "update_request_status", message, unverified=True)) from exc
    record = next((r for r in page["records"] if r.get("id") == conv["request_id"]), None)
    if record is None or record.get("status") != CRM_COMPLETED_STATUS:
        message = (
            "BOOTOG answered the status update as successful, but the request "
            + ("could not be found when read back" if record is None else "still does not show CRM Completed")
            + ", so the conversion is not confirmed as complete."
        )
        raise StepFailed(message, _failure(conv, "update_request_status", message, unverified=True))
    return StepResult(
        "Request marked CRM Completed (verified)",
        {
            "conversion": _completed(conv, "update_request_status", status=CRM_COMPLETED_STATUS),
            "contact_request": {**state["contact_request"], **record},
        },
    )


@tracked_step("refresh_requests", "Refreshing the request list", "Unable to refresh the request list.")
async def refresh_requests(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    try:
        page = await get_contact_us_requests.ainvoke(
            {"provider_id": state.get("provider_id"), "module_filter": state.get("module_filter")}, config=config
        )
    except BootogApiError:
        # The conversion itself is done; a stale list must not turn it into a failure.
        return StepResult("The request list could not be refreshed")
    return StepResult(
        f"{page['total']} open requests after the update",
        {
            "contact_requests": page["records"],
            "contact_requests_total": page["total"],
            "contact_requests_truncated": False,
            "request_filters": None,
            "refreshed": True,
        },
    )
