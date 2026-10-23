"""Pure domain logic of the Scheduling Agent: no I/O, no LLM.

- Order dashboard: the status cards of GET Order/GetOrderHeaders grouped like the
  Orders page (Intake / Field / QA Review / TPQA / Client & Close). Counts and status ids
  always come from the API; only the grouping and display labels live here.
- Summaries of orders and producers (only what the chat shows and the workflow needs).
- Appointment slots from a producer's working hours, blocked days and booked
  appointments (same rules as the host's scheduling calendar).
- Payloads for POST Order/AssignOrSchedule and POST ClientContactLog, built exactly as
  the host's Assign/Schedule and Add Log popups build them."""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

# --- order dashboard -------------------------------------------------------------

# The Orders page shows the header cards under these groups and labels
# (assignments-view: "ASSIGNED TO PRODUCER" -> "Assigned", "RECEIVED BY IPA*" -> "TPQA*").
DASHBOARD_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Intake", [("NEW", "New"), ("CANCELLED INSPECTION", "Cancelled")]),
    ("Field", [("ASSIGNED TO PRODUCER", "Assigned"), ("SCHEDULED", "Scheduled"), ("REJECTED BY PRODUCER", "Rejected by producer")]),
    ("QA Review", [("SUBMITTED", "Submitted"), ("QA", "QA"), ("QA Approved", "QA approved"), ("QA Rejected", "QA rejected")]),
    ("TPQA", [("RECEIVED BY IPA", "TPQA"), ("RECEIVED BY IPA Approved", "TPQA approved"), ("RECEIVED BY IPA Rejected", "TPQA rejected")]),
    ("Client & Close", [("CLIENT RECEIVED", "Client received"), ("CLIENT REJECTED", "Client rejected"), ("CLOSED", "Closed"), ("ARCHIVE", "Archive")]),
]
_LABELS = {name.casefold(): label for _, cards in DASHBOARD_GROUPS for name, label in cards}
_ARCHIVE = "archive"


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").casefold())


def card_label(name: str) -> str:
    return _LABELS.get((name or "").casefold(), (name or "").title())


def dashboard(headers: list[dict[str, Any]]) -> dict[str, Any]:
    """Status cards grouped like the Orders page. Unknown cards go to "Other"."""
    by_name = {str(h.get("name") or "").casefold(): h for h in headers if h.get("name")}
    used: set[str] = set()
    groups: list[dict[str, Any]] = []
    for title, cards in DASHBOARD_GROUPS:
        items = []
        for name, label in cards:
            header = by_name.get(name.casefold())
            if header is None:
                continue
            used.add(name.casefold())
            items.append({"name": header["name"], "label": label, "count": _int(header.get("count"))})
        if items:
            groups.append({"title": title, "total": sum(i["count"] for i in items), "items": items})
    other = [
        {"name": h["name"], "label": card_label(h["name"]), "count": _int(h.get("count"))}
        for key, h in by_name.items() if key not in used
    ]
    if other:
        groups.append({"title": "Other", "total": sum(i["count"] for i in other), "items": other})
    return {"groups": groups}


def match_status_cards(headers: list[dict[str, Any]], wanted: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Header cards for the status names the user used ("new", "assigned", "TPQA approved",
    "qa rejected"). Returns (cards, names not recognised)."""
    index: dict[str, dict[str, Any]] = {}
    for h in headers:
        name = str(h.get("name") or "")
        if not name:
            continue
        for alias in {name, card_label(name)}:
            index[_norm(alias)] = h
    aliases = {"cancelled": "cancelledinspection", "canceled": "cancelledinspection", "assignedtoproducer": "assigned",
               "rejected": "rejectedbyproducer", "archived": "archive", "tpqareview": "tpqa", "qareview": "qa"}
    cards, unknown = [], []
    for raw in wanted:
        key = _norm(raw)
        key = aliases.get(key, key)
        card = index.get(key) or index.get(_norm(raw.rstrip("sS")))
        if card is None:
            unknown.append(raw)
        elif card not in cards:
            cards.append(card)
    return cards, unknown


def status_filters(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """GetOrderDetails parameters for status cards, as the Orders page sends them on a card
    click: service status ids, plus review statuses for the QA/TPQA cards; ARCHIVE is an
    order status. Cards with different review statuses can't be combined in one query."""
    filters: dict[str, Any] = {}
    service_ids, review, tpqa_review, order_ids = [], [], [], []
    for card in cards:
        status_id = card.get("statusId")
        if not status_id:
            continue
        if str(card.get("name") or "").casefold() == _ARCHIVE:
            order_ids.append(status_id)
            continue
        if status_id not in service_ids:
            service_ids.append(status_id)
        statuses = [s for s in card.get("reviewStatus") or [] if isinstance(s, int)]
        if statuses:
            (tpqa_review if card_label(card.get("name") or "").startswith("TPQA") else review).extend(statuses)
    if service_ids:
        filters["orderServiceStatusIds"] = service_ids
    if order_ids:
        filters["orderStatusIds"] = order_ids
    if review:
        filters["orderReviewStatuses"] = sorted(set(review))
    if tpqa_review:
        filters["orderTpQaReviewStatuses"] = sorted(set(tpqa_review))
    return filters


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# --- orders ----------------------------------------------------------------------

# Service statuses (OrderStatus names) from which the Assign/Schedule action applies.
# Assign: the Orders page only offers assignment for services in these statuses
# (assignments-view.component.ts assignOrderPopup: ['NEW', 'ASSIGNED TO PRODUCER']).
ASSIGNABLE_STATUSES = {"NEW", "ASSIGNED TO PRODUCER"}
SCHEDULABLE_STATUSES = {"NEW", "ASSIGNED TO PRODUCER", "REJECTED BY PRODUCER", "SCHEDULED"}
_CLOSED_ORDER_STATUSES = {"CLOSED", "CANCELLED INSPECTION", "ARCHIVE", "COMPLETED"}
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def is_uuid(value: Any) -> bool:
    return isinstance(value, str) and bool(UUID_RE.match(value.strip()))


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _join(*parts: Any, sep: str = ", ") -> str | None:
    return sep.join(p for p in (_text(x) for x in parts) if p) or None


def order_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Compact view of a GetOrderDetails row or a GetOrderDetailById body."""
    prop = record.get("property") if isinstance(record.get("property"), dict) else {}
    services = []
    for building in record.get("orderBuildingInformations") or []:
        for svc in (building or {}).get("orderServices") or []:
            if not isinstance(svc, dict):
                continue
            status = svc.get("orderStatusName") or (svc.get("orderStatus") or {}).get("name")
            services.append({
                "id": svc.get("id"),
                "name": svc.get("serviceTypeName"),
                "status": status,
                "statusId": svc.get("orderStatusId"),
                "subStatus": svc.get("orderSubStatusName"),
                "reviewStatus": svc.get("reviewStatus"),
                "assignedToId": svc.get("scheduleTo"),
                "assignedTo": _text(svc.get("scheduleToUserName")),
                "assignedPhone": _text(svc.get("scheduleToPhone")) if svc.get("scheduleTo") else None,
                "assignedEmail": _text(svc.get("scheduleToEmail")) if svc.get("scheduleTo") else None,
                "scheduleDate": (svc.get("scheduleDate") or None),
                "scheduleTime": svc.get("scheduleTime"),
                "scheduleEndTime": svc.get("scheduleEndTime"),
                "completionTime": svc.get("completionTime"),
            })
    customer = _join(record.get("contactFirstName"), record.get("contactLastName"), sep=" ") or _text(
        record.get("contactFullName") or record.get("customerName")
    )
    return {
        "id": record.get("id"),
        "jobNumber": _text(record.get("jobNumber")),
        "policyNumber": _text(record.get("policyNumber")),
        "service": _text(record.get("mainSeviceName") or record.get("mainServiceTypeName")),
        "serviceTypeId": record.get("mainServiceTypeId"),
        "programName": _text(record.get("programName")),
        "programId": record.get("programId"),
        "customer": customer,
        "contactPhone": _text(record.get("contactPhone") or record.get("customerPhone")),
        "contactEmail": _text(record.get("contactEmail") or record.get("customerEmail")),
        "address": _join(prop.get("address1"), prop.get("city"), _join(prop.get("state"), prop.get("zipCode"), sep=" ")),
        "county": _text(prop.get("county")),
        "state": _text(prop.get("state")),
        "country": _text(prop.get("country") or record.get("contactCountry")),
        "latitude": prop.get("latitude"),
        "longitude": prop.get("longitude"),
        "providerId": record.get("providerId"),
        "providerName": _text(record.get("providerName")),
        "orderStatus": _text(record.get("orderStatusName")),
        "createdDateTime": record.get("createdDateTime"),
        "isRushOrder": bool(record.get("isRushOrder")),
        "completionTime": record.get("completionTime"),
        "accessToPropertyNotes": _text(record.get("accessToPropertyNotes")),
        "noOfBuildings": record.get("noOfBuildings"),
        "services": services,
    }


def order_status_text(order: dict[str, Any]) -> str:
    statuses = sorted({s["status"] for s in order.get("services") or [] if s.get("status")})
    return ", ".join(statuses) or order.get("orderStatus") or "Unknown"


def assigned_producer(order: dict[str, Any]) -> dict[str, Any] | None:
    for svc in order.get("services") or []:
        if svc.get("assignedToId"):
            return {"id": svc["assignedToId"], "name": svc.get("assignedTo"), "phone": svc.get("assignedPhone"),
                    "email": svc.get("assignedEmail")}
    return None


def eligibility(order: dict[str, Any], action: str) -> str | None:
    """None when Assign/Schedule may run on the order, else the reason it can't."""
    if (order.get("orderStatus") or "").upper() in _CLOSED_ORDER_STATUSES:
        return f"the order is {order['orderStatus']}"
    allowed = ASSIGNABLE_STATUSES if action == "assign" else SCHEDULABLE_STATUSES
    services = order.get("services") or []
    if not services:
        return "the order has no services"
    open_services = [s for s in services if (s.get("status") or "").upper() in allowed]
    if not open_services:
        return f"its services are already {order_status_text(order)}"
    return None


def order_duration_minutes(order: dict[str, Any]) -> int:
    """Appointment length = the order's completion time (seconds), as the scheduler uses."""
    try:
        minutes = int(float(order.get("completionTime") or 0) // 60)
    except (TypeError, ValueError):
        minutes = 0
    return minutes if minutes > 0 else 60


def match_orders(orders: list[dict[str, Any]], reference: str) -> list[dict[str, Any]]:
    """Orders whose job/policy number equals the reference (case-insensitive)."""
    key = _norm(reference)
    return [o for o in orders if key and key in {_norm(o.get("jobNumber")), _norm(o.get("policyNumber"))}]


# --- producers -------------------------------------------------------------------


def producer_summary(user: dict[str, Any], group: str | None = None) -> dict[str, Any]:
    reviews = user.get("reviews") if isinstance(user.get("reviews"), dict) else {}
    try:
        distance = round(float(user.get("distance")), 1) if user.get("distance") not in (None, "") else None
    except (TypeError, ValueError):
        distance = None
    return {
        "id": user.get("id"),
        "name": _join(user.get("firstName"), user.get("lastName"), sep=" ") or "Unnamed producer",
        "email": _text(user.get("emailId")),
        "phone": _text(user.get("phoneNumber")),
        "address": _join(user.get("address1"), user.get("address2"), user.get("city"),
                         _join(user.get("state"), user.get("zipcode"), sep=" - ")),
        "headshot": _text(user.get("headshot")),
        "distanceMiles": distance,
        "rating": _text(str(reviews.get("averageRating"))) if reviews.get("averageRating") not in (None, "") else None,
        "ratingCount": reviews.get("averageRatingCount"),
        "openAssignments": user.get("noOfOpenAssignments"),
        "scheduledAssignments": user.get("noOfScheduledAssignments"),
        "groups": [group] if group else [],
    }


def merge_producers(groups: dict[str, list[dict[str, Any]]], labels: dict[str, str]) -> list[dict[str, Any]]:
    """One entry per producer (a producer can be in several groups), nearest first; the
    order's currently assigned producer is not special-cased here."""
    merged: dict[str, dict[str, Any]] = {}
    for key, users in groups.items():
        for user in users:
            summary = merged.get(user["id"])
            if summary is None:
                merged[user["id"]] = producer_summary(user, labels.get(key))
            elif labels.get(key) and labels[key] not in summary["groups"]:
                summary["groups"].append(labels[key])
    return sorted(merged.values(), key=lambda p: (p["distanceMiles"] is None, p["distanceMiles"] or 0, p["name"]))


def match_producers(producers: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    words = [w for w in re.findall(r"[^\W_]+", (name or "").casefold()) if w]
    if not words:
        return []
    hits = []
    for p in producers:
        have = re.findall(r"[^\W_]+", (p.get("name") or "").casefold())
        if all(any(h == w or (len(w) >= 3 and h.startswith(w)) for h in have) for w in words):
            hits.append(p)
    return hits


# --- producer ranking (Assign) -------------------------------------------------------

MATCHED_GROUP = "Matched"  # PRODUCER_GROUPS["matchedInspectors"]


def rank_producers(
    producers: list[dict[str, Any]],
    *,
    max_distance_miles: float,
    assigned_id: str | None = None,
    filters_sent: bool,
) -> dict[str, Any]:
    """Ranks the GetOrderUsers candidates for an assignment, using only what the API returns.

    What BOOTOG's "matched" group means (UserDetailService.GetOrderUsersAsync): the provider's
    active producers that pass the filters sent — program, credential ids, skill set ids. A
    producer matches with ANY ONE listed credential (approved document) and ANY ONE listed
    skill set. With no filters sent, every producer of the provider is "matched", so then the
    group says nothing and is not used as an eligibility signal.

    Excluded (reason shown to the user):
      - already the order's producer
      - no distance returned        -> "within N miles" can't be verified, never assumed
      - farther than the limit       (distance = great-circle statute miles from BOOTOG)
      - not in the matched group     (only when matching filters were sent)
    Recommended = the nearest eligible producer (the Assign/Schedule screen also orders
    producers by distance); ties go to fewer open assignments, then name."""
    ranked: list[dict[str, Any]] = []
    for p in producers:
        entry = {**p, "eligible": False, "excludedReason": None, "recommended": False,
                 "isCurrentProducer": bool(assigned_id and p.get("id") == assigned_id)}
        distance = p.get("distanceMiles")
        if entry["isCurrentProducer"]:
            entry["excludedReason"] = "Already assigned to this order"
        elif distance is None:
            entry["excludedReason"] = "Distance not returned by BOOTOG"
        elif distance > max_distance_miles:
            entry["excludedReason"] = f"{distance} mi — outside {max_distance_miles:g} mi"
        elif filters_sent and MATCHED_GROUP not in (p.get("groups") or []):
            entry["excludedReason"] = "Doesn't match the service's skill / credential filters"
        else:
            entry["eligible"] = True
        ranked.append(entry)

    def key(e: dict[str, Any]):
        open_count = e.get("openAssignments") if isinstance(e.get("openAssignments"), int) else 10**6
        return (not e["eligible"], e.get("distanceMiles") is None, e.get("distanceMiles") or 0, open_count, e.get("name") or "")

    ranked.sort(key=key)
    eligible = [e for e in ranked if e["eligible"]]
    if eligible:
        eligible[0]["recommended"] = True
    for i, e in enumerate(ranked, 1):
        e["rank"] = i
    within = [e for e in ranked if e.get("distanceMiles") is not None and e["distanceMiles"] <= max_distance_miles]
    rule = (
        "producers matching the service's skill / credential filters (any one of each) within the distance limit, nearest first"
        if filters_sent else
        "no skill / credential filters apply to this service, so: producers within the distance limit, nearest first"
    )
    return {
        "items": ranked,
        "eligibleCount": len(eligible),
        "withinDistanceCount": len(within),
        "recommended": eligible[0] if eligible else None,
        "maxDistanceMiles": max_distance_miles,
        "filtersSent": filters_sent,
        "rule": rule + " (ties: fewer open assignments)",
    }


def matching_filters(
    service_type: dict[str, Any] | None,
    program: dict[str, Any] | None,
    valid_skill_ids: set[str] | None,
    valid_credential_ids: set[str] | None,
) -> dict[str, list[str]]:
    """The skill set / credential ids the Assign/Schedule screen pre-fills for an order
    (scheduling-filter getFilterDefaults + applyProgramDefaults): from the service type, or
    from the program when it lists any; kept only when they are valid ids for the provider."""
    skills = [x.get("skillSetId") for x in (service_type or {}).get("serviceTypeSkillSets") or [] if isinstance(x, dict)]
    creds = [x.get("documentTypeId") for x in (service_type or {}).get("serviceTypeDocumentTypes") or [] if isinstance(x, dict)]
    program_skills = [x.get("skillSetId") for x in (program or {}).get("programSkillSets") or [] if isinstance(x, dict)]
    program_creds = [x.get("documentTypeId") for x in (program or {}).get("programDocumentTypes") or [] if isinstance(x, dict)]
    skills = program_skills or skills
    creds = program_creds or creds

    def valid(ids: list[Any], allowed: set[str] | None) -> list[str]:
        out = []
        for i in ids:
            if isinstance(i, str) and i and i not in out and (allowed is None or i in allowed):
                out.append(i)
        return out

    return {"skillsetIds": valid(skills, valid_skill_ids), "credentialIds": valid(creds, valid_credential_ids)}


def batch_mapping(
    count: int, groups: list[dict[str, Any]], current: dict[int, dict[str, Any]] | None = None
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Applies the user's order groups to a list of `count` orders.

    Returns ({position: {"producer_name": ...} | {"nearest": True}}, positions that don't
    exist). Positions are 1-based. Groups apply in order and override earlier choices (so
    "change the third order to Latha" updates a prepared batch); "remaining" groups fill
    every position still without a producer. A group without a producer maps nothing."""
    mapping = dict(current or {})
    invalid: list[int] = []

    def spec(group: dict[str, Any]) -> dict[str, Any] | None:
        if group.get("nearest"):
            return {"nearest": True}
        name = (group.get("producer_name") or "").strip()
        return {"producer_name": name[:100]} if name else None

    for group in groups:
        if group.get("remaining"):
            continue
        chosen = spec(group)
        positions = range(1, count + 1) if group.get("all_orders") else group.get("positions") or []
        for pos in positions:
            if not isinstance(pos, int) or not 1 <= pos <= count:
                invalid.append(pos)
            elif chosen:
                mapping[pos] = chosen
    for group in groups:
        chosen = spec(group)
        if group.get("remaining") and chosen:
            for pos in range(1, count + 1):
                mapping.setdefault(pos, chosen)
    return mapping, sorted(set(p for p in invalid if isinstance(p, int)))


def search_radii(start: float, step: float, maximum: float) -> list[float]:
    """Radii for an expanding producer search: start, start+step, ... up to maximum (always
    tried last). Invalid settings fall back to a single search at `start`."""
    if step <= 0 or maximum <= start:
        return [start]
    radii, radius = [], start
    while radius < maximum:
        radii.append(radius)
        radius += step
    return radii + [maximum]


def batch_selection(count: int, groups: list[dict[str, Any]], current: list[int] | None = None) -> list[int]:
    """The list positions a multi-order assignment covers. Only the orders the user picked
    ("the first 10" -> 1..10) are selected; "all these orders", "the rest" or no groups at all
    select the whole list. A change to a prepared batch adds the positions it names."""
    if not groups or any(g.get("all_orders") or g.get("remaining") for g in groups):
        return list(range(1, count + 1))
    named = {p for g in groups for p in g.get("positions") or [] if isinstance(p, int) and 1 <= p <= count}
    return sorted(set(current or []) | named)


def service_requirements(service_type: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the order's service type lists (GET ServiceType?filter=(id=...)), for display."""
    if not service_type:
        return None
    credentials = [
        c.get("credentialTypeName") for c in service_type.get("serviceTypeCredentialTypes") or []
        if isinstance(c, dict) and c.get("isMandatory") and c.get("credentialTypeName")
    ]
    skills = [s.get("skillSetName") for s in service_type.get("serviceTypeSkillSets") or [] if isinstance(s, dict) and s.get("skillSetName")]
    return {
        "service": _text(service_type.get("name")),
        "mandatoryCredentials": sorted(set(credentials)),
        "skillSets": sorted(set(skills)),
    }


def has_coordinates(order: dict[str, Any]) -> bool:
    lat, lng = order.get("latitude"), order.get("longitude")
    try:
        return lat is not None and lng is not None and not (float(lat) == 0.0 and float(lng) == 0.0)
    except (TypeError, ValueError):
        return False


def match_customer(orders: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Orders whose customer name contains every word the user gave ("Naveen" -> "Naveen b")."""
    words = [w for w in re.findall(r"[^\W_]+", (name or "").casefold()) if w]
    if not words:
        return []
    out = []
    for o in orders:
        have = re.findall(r"[^\W_]+", (o.get("customer") or "").casefold())
        if all(any(h == w or (len(w) >= 3 and h.startswith(w)) for h in have) for w in words):
            out.append(o)
    return out


# --- availability ----------------------------------------------------------------

SLOT_STEP_MINUTES = 60  # the scheduling calendar's slot grid


def parse_clock(value: Any) -> int | None:
    """"08:00 AM" / "8:30 pm" / "14:00" -> minutes after midnight."""
    if not isinstance(value, str):
        return None
    m = re.match(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*([AaPp][Mm])?\s*$", value)
    if not m:
        return None
    hour, minute, meridiem = int(m.group(1)), int(m.group(2)), (m.group(3) or "").upper()
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem == "PM" else 0)
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def format_clock(minutes: int) -> str:
    """Minutes after midnight -> "hh:mm AM" (the host's date-fns 'hh:mm a')."""
    hour, minute = divmod(minutes % (24 * 60), 60)
    return f"{(hour % 12) or 12:02d}:{minute:02d} {'PM' if hour >= 12 else 'AM'}"


def parse_day(value: Any) -> date | None:
    if isinstance(value, str):
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", value.strip())
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    return None


def js_weekday(day: date) -> int:
    """JavaScript getDay(): Sunday = 0 ... Saturday = 6 (UserWorkingHour.dayOfWeek)."""
    return (day.weekday() + 1) % 7


@dataclass(frozen=True)
class LocalClock:
    """The user's wall clock (the browser sends its UTC offset with every message)."""

    utc_offset_minutes: int = 0

    def now(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(minutes=self.utc_offset_minutes)

    def today(self) -> date:
        return self.now().date()

    def local_date_of(self, value: Any) -> date | None:
        """Calendar date (user's time) of a Bootog UTC timestamp."""
        if not isinstance(value, str):
            return None
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return parse_day(value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return (stamp.astimezone(timezone.utc) + timedelta(minutes=self.utc_offset_minutes)).date()

    def day_bounds_utc(self, first: date, last: date) -> tuple[str, str]:
        """ISO UTC instants of the user's local start of `first` and end of `last`."""
        offset = timedelta(minutes=self.utc_offset_minutes)
        start = datetime(first.year, first.month, first.day, tzinfo=timezone.utc) - offset
        end = datetime(last.year, last.month, last.day, 23, 59, 59, 999000, tzinfo=timezone.utc) - offset
        return _iso(start), _iso(end)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def _busy_ranges(day: date, booked: list[dict[str, Any]], blocks: list[dict[str, Any]], ignore_order_id: str | None):
    ranges: list[tuple[int, int, str]] = []
    for item in booked:
        if parse_day(item.get("scheduleDate")) != day:
            continue
        if ignore_order_id and item.get("orderId") == ignore_order_id:
            continue  # rescheduling: the order's own current slot is not a conflict
        start, end = parse_clock(item.get("scheduleTime")), parse_clock(item.get("scheduleEndTime"))
        if start is None:
            continue
        ranges.append((start, end if end and end > start else start + SLOT_STEP_MINUTES, "Booked"))
    for block in blocks:
        first, last = parse_day(block.get("fromDate")), parse_day(block.get("toDate")) or parse_day(block.get("fromDate"))
        if not first or not last or not first <= day <= last:
            continue
        start, end = _minutes(block.get("fromTime")), _minutes(block.get("toTime"))
        if start is None or end is None or end <= start:
            start, end = 0, 24 * 60  # no time range: the whole day is blocked
        ranges.append((start, end, "Blocked"))
    return ranges


def _minutes(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        if value.strip().isdigit():
            return int(value.strip())
        return parse_clock(value)
    return None


def available_slots(
    *,
    days: list[date],
    working_hours: list[dict[str, Any]],
    booked: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    duration_minutes: int,
    clock: LocalClock,
    ignore_order_id: str | None = None,
    per_day_limit: int = 8,
) -> list[dict[str, Any]]:
    """Open slots per day: inside the producer's working hours for that weekday, on the
    calendar's hourly grid, long enough for the order, not overlapping a booked
    appointment or a blocked period, and not in the past."""
    hours_by_day: dict[int, tuple[int, int]] = {}
    for wh in working_hours:
        try:
            dow, start, end = int(wh.get("dayOfWeek")), int(wh.get("fromTime")), int(wh.get("toTime"))
        except (TypeError, ValueError):
            continue
        if end > start:
            hours_by_day[dow] = (start, end)

    now = clock.now()
    out: list[dict[str, Any]] = []
    for day in days:
        hours = hours_by_day.get(js_weekday(day))
        entry: dict[str, Any] = {"date": day.isoformat(), "weekday": day.strftime("%A"), "slots": []}
        if hours is None:
            entry["note"] = "Not a working day"
            out.append(entry)
            continue
        busy = _busy_ranges(day, booked, blocks, ignore_order_id)
        start, end = hours
        cursor = start
        while cursor + duration_minutes <= end and len(entry["slots"]) < per_day_limit:
            slot_end = cursor + duration_minutes
            past = day < now.date() or (day == now.date() and cursor <= now.hour * 60 + now.minute)
            clash = any(b_start < slot_end and cursor < b_end for b_start, b_end, _ in busy)
            if not past and not clash:
                entry["slots"].append({"start": format_clock(cursor), "end": format_clock(slot_end)})
            cursor += SLOT_STEP_MINUTES
        if not entry["slots"]:
            entry["note"] = "Fully booked" if busy else "No open time left"
        out.append(entry)
    return out


def slot_is_open(availability: list[dict[str, Any]], day: str, start: str) -> dict[str, Any] | None:
    wanted = parse_clock(start)
    for entry in availability:
        if entry["date"] != day:
            continue
        for slot in entry["slots"]:
            if parse_clock(slot["start"]) == wanted:
                return slot
    return None


# --- write payloads ----------------------------------------------------------------


def _producer_fields(producer: dict[str, Any]) -> dict[str, Any]:
    return {
        "targetUserId": producer["id"],
        "scheduleToUserName": producer.get("name") or "",
        "scheduleToUserHeadshotUrl": producer.get("headshot") or "",
        "scheduleToEmail": producer.get("email") or "",
        "scheduleToPhone": producer.get("phone") or "",
        "scheduleToUserAddress": producer.get("address") or "",
    }


def assign_payload(order: dict[str, Any], producer: dict[str, Any]) -> dict[str, Any]:
    """POST Order/AssignOrSchedule without a date = assign (scheduling-view assignUser)."""
    return {
        **_producer_fields(producer),
        "orders": [{"orderId": order["id"], "scheduleTime": None, "scheduleEndTime": None}],
        "isPackage": False,
    }


def schedule_payload(
    order: dict[str, Any], producer: dict[str, Any], day: str, start: str, end: str, comments: str | None
) -> dict[str, Any]:
    """POST Order/AssignOrSchedule with a date and times (schedule-assignment-appointment
    submit). scheduleDate is the local calendar day at 00:00 'Z', as the host sends it."""
    return {
        **_producer_fields(producer),
        "orders": [{"orderId": order["id"], "scheduleTime": start, "scheduleEndTime": end}],
        "scheduleDate": f"{day}T00:00:00.000Z",
        "endTimeOverrideReason": "",
        "scheduleComments": comments or "",
        "inspectorComments": order.get("accessToPropertyNotes") or "",
        "isPackage": False,
    }


# Add Log enums mirrored from libs/models/enums (contact-type, caller-type, person-type,
# response-result, client-contact-log-visibility). These are the API's codes.
CONTACT_TYPES = {"phone_call": 1, "sms": 2, "email": 3, "email_and_sms": 4, "note": 5}
CONTACT_TYPE_LABELS = {1: "Phone Call", 2: "SMS", 3: "Email", 4: "Email & SMS", 5: "Note"}
CALLER_TYPES = {"policyholder": 1, "agent": 2, "agency_csr": 3, "property_manager": 4, "family_member": 5, "realtor": 6, "other": 7}
PERSON_TYPES = {"inbound": 1, "outbound": 2}
RESPONSE_RESULTS = {
    "answered": 1, "no_answer": 2, "left_voicemail": 3, "number_disconnected": 4, "answered_and_disconnected": 5,
    "reorder_left_voicemail": 6, "other": 7, "has_not_closed": 8, "insured_call_back": 9, "policy_holder_not_home": 10,
    "no_response_from_policy_holder": 11, "send_email": 12, "wrong_number": 13, "service_refused": 14,
    "condo_townhouse": 15, "closing_fell_through": 16,
}
RESPONSE_LABELS = {v: k.replace("_", " ").capitalize() for k, v in RESPONSE_RESULTS.items()}
CALLER_LABELS = {v: k.replace("_", " ").title() for k, v in CALLER_TYPES.items()}
VISIBLE_TO_ALL = [1, 2]  # Producer, Provider


def log_payload(order_id: str, log: dict[str, Any], clock: LocalClock) -> dict[str, Any]:
    """POST ClientContactLog body for one log (contact-log-popup validation() + FinalSubmitLog())."""
    now = clock.now()
    channel = log["channel"]
    payload: dict[str, Any] = {
        "orderId": order_id,
        "contactType": CONTACT_TYPES[channel],
        "dateOfContact": now.strftime("%m/%d/%Y"),
        "timeOfContact": f"{(now.hour % 12) or 12}:{now.minute:02d} {'PM' if now.hour >= 12 else 'AM'}",
        "visibleTo": VISIBLE_TO_ALL,
        "followUpNeeded": False,
        "languageKnown": [],
        "isVILinkSent": False,
        "isAssignOrScheduleLink": bool(log.get("about_schedule")),
        "isInvoiceAttached": False,
        "isPDFReportShared": False,
        "isHTMLReportShared": False,
        "clientContactLogAttachments": [],
        "emailContent": log.get("message") or "",
        "personType": PERSON_TYPES["outbound"],
        "callType": CALLER_TYPES["other"],
        "responseResult": RESPONSE_RESULTS["other"],
    }
    if log.get("user_name"):
        payload["userName"] = log["user_name"]
    if channel == "phone_call":
        payload.update({
            "contactName": log.get("contact_name"),
            "contactNumber": log.get("contact_number"),
            "personType": PERSON_TYPES[log.get("direction") or "outbound"],
            "callType": CALLER_TYPES[log.get("caller_type") or "other"],
            "responseResult": RESPONSE_RESULTS[log["response_result"]],
        })
    elif channel == "sms":
        payload.update({"smsTo": log.get("contact_number"), "contactName": log.get("contact_name"), "smsContent": log.get("message")})
    elif channel == "email":
        payload.update({
            "emailTo": log.get("email_to"), "contactName": log.get("contact_name"), "emailSubject": log.get("subject"),
            "responseResult": RESPONSE_RESULTS["send_email"],
        })
    return payload


def missing_log_fields(log: dict[str, Any]) -> list[str]:
    """What the Add Log form requires for the channel and is still missing."""
    channel = log.get("channel")
    if channel not in CONTACT_TYPES or channel == "email_and_sms":
        return ["channel (phone call, SMS, email or note)"]
    missing = []
    if not log.get("message"):
        missing.append("notes" if channel in ("phone_call", "note") else "message")
    if channel == "phone_call":
        if not log.get("contact_name"):
            missing.append("contact name")
        if not log.get("contact_number"):
            missing.append("contact number")
        if log.get("response_result") not in RESPONSE_RESULTS:
            missing.append("call result (answered, no answer, left voicemail, ...)")
    elif channel == "sms" and not log.get("contact_number"):
        missing.append("phone number")
    elif channel == "email":
        if not log.get("email_to"):
            missing.append("email address")
        if not log.get("subject"):
            missing.append("subject")
    return missing


def log_summary(record: dict[str, Any]) -> dict[str, Any]:
    ctype = record.get("contactType")
    return {
        "id": record.get("id"),
        "type": CONTACT_TYPE_LABELS.get(ctype, "Log"),
        "date": _join(record.get("dateOfContact"), record.get("timeOfContact"), sep=" "),
        "contact": _text(record.get("contactName")) or _text(record.get("emailTo")) or _text(record.get("smsTo")),
        "number": _text(record.get("contactNumber")) or _text(record.get("smsTo")),
        "subject": _text(record.get("emailSubject")),
        "message": _strip_html(record.get("emailContent") or record.get("smsContent") or record.get("note")),
        "result": RESPONSE_LABELS.get(record.get("responseResult")) if ctype == 1 else None,
        "by": _text(record.get("userName") or record.get("createdByUserName")),
        "createdDateTime": record.get("createdDateTime"),
    }


def _strip_html(value: Any, limit: int = 300) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"<[^>]+>", " ", value)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] or None
