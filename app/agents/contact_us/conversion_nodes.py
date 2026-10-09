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
    verify_account           GET UserDetail/GetUserRolesByEmailId (account + role present)
    link_provider_vendor     POST Provider/SaveProviderVendor (idempotent)
    save_subcategories       POST ProviderContractorSubCategory/BulkInsert
    update_request_status    PUT ContactUs/bulk-update-file {Status: 11}
    refresh_requests         GET ContactUs (the list after the change)
    search_providers         "list/find providers" (no conversion needed)
    resolve_assigned_user    "assigned to X" -> assignee id (GetContactUsAssignedUser)

Execution steps record their outcome in `conversion` (completed / failed), also on
failure, so "retry" resumes at the failed step and never creates a second account.
Critical identifiers come only from API responses, never from the LLM."""

import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.contact_us.conversions import (
    ALL_TYPES,
    CONVERSION_TYPES,
    CUSTOMER,
    PARTNER_TYPES,
    ConversionType,
    OPERATION_TYPE,
    build_draft,
    build_vendor_user_form,
    conversion_type,
    fingerprint,
    generate_password,
)
from app.agents.contact_us.nodes import StepFailed, StepResult, tracked_step
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
    """The form the draft follows: the conversion's type, Customer by default."""
    return active_type(state) or CUSTOMER


def conversion_draft(state: ContactUsChatState) -> dict[str, Any] | None:
    record = state.get("contact_request")
    if not record:
        return None
    return build_draft(draft_type(state), record, state.get("user_customer_fields") or {})


def conversion_fingerprint(state: ContactUsChatState) -> str:
    conv = conversion_of(state)
    return fingerprint(
        conv.get("type"),
        conv.get("request_id"),
        conv.get("role_id"),
        (conv.get("target_provider") or {}).get("id"),
        sorted(s["id"] for s in conv.get("subcategories") or []),
        state.get("customer_payload") or {},
    )


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
    record = state["contact_request"]
    request_id = record.get("id")
    conv = conversion_of(state)
    turn = state.get("turn_conversion") or {}
    operation = state.get("operation")

    if conv.get("request_id") != request_id:
        conv = {"request_id": request_id}
    conv["blocked"] = None
    current = conversion_type(conv.get("type"))
    finished = bool(current) and set(current.steps()) <= set(conv.get("completed") or [])
    failed = conv.get("failed") or {}
    # An account may exist for an unfinished conversion: creation completed, or its answer
    # was lost (timeout / 5xx). Switching type then could create a second account.
    in_flight = not finished and (
        "create_account" in (conv.get("completed") or [])
        or (failed.get("step") == "create_account" and bool(failed.get("outcome_unknown")))
    )
    # A finished conversion's partner subtype is not carried into a new one.
    wanted = _wanted_type(state, conv, turn, operation, retain_partner=not finished)

    if wanted and (state.get("created_customers") or {}).get(converted_key(request_id, wanted)):
        conv["blocked"] = "already_converted"
        conv["blocked_type"] = wanted
        return StepResult(f"Already converted to {conversion_type(wanted).label}", {"conversion": conv})

    if in_flight and wanted != conv.get("type"):
        conv["blocked"] = "type_locked"
        return StepResult(f"Already started as {current.label}", {"conversion": conv})

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
    return StepResult(message, {"conversion": conv})


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
        question, options = "choose_partner_type", [t.label.removeprefix("Partner — ") for t in PARTNER_TYPES]
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


@tracked_step("resolve_target_provider", "Finding the service provider", "Unable to look up service providers.")
async def resolve_target_provider(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    turn = state.get("turn_conversion") or {}
    candidates = conv.get("provider_candidates") or []
    pos = turn.get("option_position") if _answered(state, "choose_provider") else None
    name = (turn.get("target_provider_name") or conv.get("pending_provider_name") or "").strip()
    conv["pending_provider_name"] = None

    chosen = None
    if pos and candidates and 1 <= pos <= len(candidates):
        chosen = candidates[pos - 1]
    elif name:
        found, truncated = await find_providers(config, name, None)
        exact = [p for p in found if _norm(p["name"]) == _norm(name)]
        if len(exact) == 1 or len(found) == 1:
            chosen = (exact or found)[0]
        else:
            options = (exact or found)[:_OPTION_LIMIT]
            conv["provider_candidates"] = [{"id": p["id"], "name": p["name"], "address": p.get("address")} for p in options]
            status = "ambiguous" if options else "not_found"
            question = {"type": "choose_provider", "status": status, "requested": name, "more": len(found) > len(options)}
            return StepResult(
                f"{len(found)} providers match “{name}” — asking which one" if options else f"No provider matches “{name}”",
                _waiting({"conversion": conv}, question),
                data={"matches": len(found), "truncated": truncated},
            )
    if chosen is None:
        # Nothing named yet: ask, offering the first providers as examples.
        first = await search_service_providers.ainvoke({"all_pages": False}, config=config)
        options = first["providers"][:_OPTION_LIMIT]
        conv["provider_candidates"] = [{"id": p["id"], "name": p["name"], "address": p.get("address")} for p in options]
        return StepResult(
            "Asking which service provider to use",
            _waiting({"conversion": conv}, {"type": "choose_provider", "status": "needed"}),
        )

    conv["target_provider"] = {"id": chosen["id"], "name": chosen["name"]}
    conv["provider_candidates"] = None
    conv["carried_over"] = {k: v for k, v in (conv.get("carried_over") or {}).items() if k != "Service provider"}
    if conversion_type(conv.get("type")) and conversion_type(conv["type"]).needs_subcategories:
        conv["subcategories"] = []  # sub-categories are provider-specific
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


@tracked_step("resolve_assigned_user", "Finding the assignee", "Unable to look up assigned users.")
async def resolve_assigned_user(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    filters = dict(state.get("request_filters") or {})
    wanted = _norm(filters.get("assigned_user_name"))
    cache = state.get("assigned_users_cache") or {}
    update: dict[str, Any] = {}
    if cache.get("module") == state.get("module_filter") and time.time() - cache.get("at", 0) < _ASSIGNEE_CACHE_SECONDS:
        users = cache.get("users") or []
    else:
        users = await get_contact_us_assigned_users.ainvoke({"module_filter": state.get("module_filter")}, config=config)
        update["assigned_users_cache"] = {"at": time.time(), "module": state.get("module_filter"), "users": users}

    exact = [u for u in users if _norm(u["name"]) == wanted]
    tokens = name_tokens(wanted)
    partial = [u for u in users if tokens and all(t in _norm(u["name"]) for t in tokens)]
    found = exact or partial
    if len(found) == 1:
        filters["assigned_user_ids"] = [found[0]["id"]]
        return StepResult(f"Requests assigned to {found[0]['name']}", {**update, "request_filters": filters})
    note = {
        "assignee": {
            "status": "ambiguous" if found else "not_found",
            "requested": filters.get("assigned_user_name"),
            "candidates": [u["name"] for u in (found or users)[:_OPTION_LIMIT]],
        }
    }
    return StepResult(
        f"No unique assignee matched “{filters.get('assigned_user_name')}”",
        {**update, "last_tool_result": note, "pending_question": {"type": "choose_assignee"}, "awaiting_user_input": True},
    )


# --- checks ---------------------------------------------------------------------------------


@tracked_step("check_existing_account", "Checking for an existing account", "Unable to check for an existing account.")
async def check_existing_account(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    ctype = active_type(state)
    email = (state.get("customer_payload") or {}).get("email") or ""
    accounts: list[dict[str, Any]] = []
    for candidate in dict.fromkeys([email, email.casefold()]):  # Bootog matches case-sensitively
        accounts += await get_accounts_by_email.ainvoke({"email": candidate}, config=config)

    roles = [r for a in accounts for r in a["roles"]]
    with_role = [r for r in roles if r["roleName"].casefold() == ctype.role_name.casefold()]
    conv["account"] = {
        "email": email,
        "exists": bool(accounts),
        "has_role": bool(with_role),
        "roles": sorted({r["roleName"] for r in roles})[:10],
    }
    lost_answer = (conv.get("failed") or {}).get("step") == "create_account" and (conv.get("failed") or {}).get(
        "outcome_unknown"
    )
    if with_role and lost_answer:
        # An earlier attempt whose answer was lost did create the account: adopt it.
        vendor = next((r["providerId"] for r in with_role if is_guid(r.get("providerId"))), None)
        user = next((a["userId"] for a in accounts if any(r in with_role for r in a["roles"])), None)
        conv = _completed(conv, "create_account", created_vendor_id=conv.get("created_vendor_id") or vendor, user_id=user)
        return StepResult("The earlier attempt did create the account — resuming", {"conversion": conv})
    if with_role:
        conv["blocked"] = "existing_role"
        return StepResult(f"{email} is already registered as {ctype.role_name}", {"conversion": conv})
    message = f"Existing login found (roles: {', '.join(conv['account']['roles']) or 'none'})" if accounts else "No existing account"
    return StepResult(message, {"conversion": conv})


@tracked_step("resolve_role", "Looking up the role", "Unable to look up the role.")
async def resolve_role(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    ctype = active_type(state)
    try:
        role_id = await get_default_role_id.ainvoke({"role_name": ctype.role_name}, config=config)
    except RoleNotFoundError as exc:
        raise StepFailed(exc.message) from exc
    conv.update(role_id=role_id, role_name=ctype.role_name)
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
    form = build_vendor_user_form(
        ctype, state.get("customer_payload") or {}, conv["role_id"], state["contact_request"], generate_password()
    )
    try:
        vendor_id = await AccountApi(config["configurable"]["bootog_client"]).add_vendor_user(form)
    except BootogApiError as exc:
        # `failed: true` / a 4xx is a definitive answer; a timeout, lost connection or 5xx
        # is not — the account may exist, so the next attempt checks Bootog before creating.
        unknown = exc.outcome_unknown
        kind = "unknown" if unknown else "access" if exc.status_code in (401, 403) else "rejected"
        message = f"The account could not be created. {exc.message}"
        failure = _failure(
            conv, "create_account", message,
            outcome_unknown=unknown, kind=kind, code=exc.code, fingerprint=conv.get("awaiting_fingerprint"),
        )
        raise StepFailed(message, failure) from exc
    finally:
        form.clear()  # drop the password as soon as the request is done
    conv = _completed(conv, "create_account", created_vendor_id=vendor_id)
    return StepResult(f"{ctype.label} account created", {"conversion": conv})


@tracked_step("verify_account", "Verifying the account", "Account verification failed.")
async def verify_account(state: ContactUsChatState, config: RunnableConfig) -> StepResult:
    conv = conversion_of(state)
    ctype = active_type(state)
    email = (conv.get("account") or {}).get("email") or (state.get("customer_payload") or {}).get("email")
    accounts = await get_accounts_by_email.ainvoke({"email": email}, config=config)
    match = next(
        (a for a in accounts if any(r["roleName"].casefold() == ctype.role_name.casefold() for r in a["roles"])), None
    )
    if match is None:
        message = f"Bootog reported the account as created, but {email} doesn't have the {ctype.role_name} role yet."
        raise StepFailed(message, _failure(conv, "verify_account", message))
    vendor = conv.get("created_vendor_id") or next(
        (r["providerId"] for r in match["roles"] if r["roleName"].casefold() == ctype.role_name.casefold() and is_guid(r.get("providerId"))),
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
    try:
        await mark_contact_request_converted.ainvoke({"request_id": conv["request_id"]}, config=config)
    except BootogApiError as exc:
        message = f"The request status could not be updated. {exc.message}"
        raise StepFailed(message, _failure(conv, "update_request_status", message)) from exc
    record = {**state["contact_request"], "status": CRM_COMPLETED_STATUS}  # what Bootog now holds
    return StepResult(
        "Request marked CRM Completed",
        {"conversion": _completed(conv, "update_request_status", status=CRM_COMPLETED_STATUS), "contact_request": record},
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
