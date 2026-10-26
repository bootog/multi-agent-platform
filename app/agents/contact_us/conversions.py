"""Contact Us request -> account conversions (Customer, B2B Client, Partners).

Everything here is verified against the host app and the Bootog services:

  Contact Us screen (master/.../contact-us.component.ts rowAction):
    "Customer" -> customer-detail-popup  -> AddVendorUser (role Customer, ProviderType 1,
                                            IsSkipTenant false)
    "B2B"      -> /user-accounts/create-b2b     (agency-popup, isCarrier)
                                         -> AddVendorUser (role InsuranceCarrier, ProviderType 4,
                                            IsSkipTenant true) -> SaveProviderVendor
    "Partners" -> /user-accounts/create-partner (agency-popup, company type picked by the user:
                                            InsuranceAgency 2 / RealEstateAgency 3 / Contractor 5,
                                            role InsuranceAgency / RealEstateCompany / Contractor)
                                         -> AddVendorUser -> SaveProviderVendor
                                            (+ ProviderContractorSubCategory/BulkInsert for Contractor)
  Then the request is marked ContactUsStatus.CRMCompleted (11).

The FormData keys and constant values mirror UserDetailService.createNewUser exactly.
Passwords are generated here, server-side, at submit time; Bootog emails it to the new
account (Auth0 forces a change at first login). It is never stored, logged or shown.

Adding a conversion type = one more ConversionType entry; the graph is generic."""

import base64
import binascii
import hashlib
import json
import re
import secrets
import string
from dataclasses import dataclass
from typing import Any

# --- fields -------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    source: str | None  # Contact Us record field it is prefilled from
    kind: str = "text"  # text | email | phone | url | integer | money | latitude | longitude


FIELD_SPECS: dict[str, FieldSpec] = {
    f.key: f
    for f in [
        FieldSpec("companyName", "Company Name", "companyName"),
        FieldSpec("firstName", "First Name", "firstName"),
        FieldSpec("lastName", "Last Name", "lastName"),
        FieldSpec("email", "Email", "emailId", "email"),
        FieldSpec("phone", "Phone Number", "phone", "phone"),
        FieldSpec("website", "Website", "website", "url"),
        FieldSpec("companyBio", "Company Bio", None),
        FieldSpec("yearsInBusiness", "Years in Business", None, "integer"),
        FieldSpec("annualRevenue", "Annual Revenue", None, "money"),
        FieldSpec("address1", "Street Address 1", "address1"),
        FieldSpec("address2", "Street Address 2", "address2"),
        FieldSpec("city", "City", "city"),
        FieldSpec("state", "State", "state", "state"),
        FieldSpec("zipcode", "Zipcode", "zipcode", "zipcode"),
        FieldSpec("county", "County", "county"),
        FieldSpec("country", "Country", "country", "country"),
        FieldSpec("latitude", "Latitude", "latitude", "latitude"),
        FieldSpec("longitude", "Longitude", "longitude", "longitude"),
    ]
}
FIELD_LABELS: dict[str, str] = {k: f.label for k, f in FIELD_SPECS.items()}

_PERSON = ("firstName", "lastName", "email", "phone")
_ADDRESS = ("address1", "address2", "city", "state", "zipcode", "county", "country", "latitude", "longitude")
_ADDRESS_REQUIRED = ("address1", "city", "state", "zipcode", "county", "country")

CUSTOMER_FORM = (*_PERSON, *_ADDRESS)
# The structured address form (coordinates are never typed in; they come from the request
# or from an address picked from Google suggestions on the form).
ADDRESS_FORM_FIELDS = ("address1", "address2", "city", "state", "zipcode", "county", "country")
COMPANY_FORM = ("companyName", *_PERSON, "website", "companyBio", "yearsInBusiness", "annualRevenue", *_ADDRESS)


# --- conversion types ------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionType:
    key: str
    label: str
    operation: str  # chat `operation` while this conversion is in progress
    role_name: str  # Roles/GetRoles name
    provider_type: int  # ProviderType enum
    skip_tenant: bool  # IsSkipTenant: true -> a new company (provider) is created
    needs_target_provider: bool  # SaveProviderVendor under a selected service provider
    needs_subcategories: bool
    fields: tuple[str, ...]
    required: frozenset[str]
    # Created as soon as the user's action completes valid details (Customer). Otherwise
    # an explicit confirmation of the summary is required (B2B Client, Partners).
    auto_create: bool = False
    # Optional images the form takes (UPLOAD_SPECS keys), sent with AddVendorUser.
    uploads: tuple[str, ...] = ()

    def steps(self) -> list[str]:
        """Execution steps after confirmation, in order."""
        out = ["create_account", "verify_account"]
        if self.needs_target_provider:
            out.append("link_provider_vendor")
        if self.needs_subcategories:
            out.append("save_subcategories")
        out.append("update_request_status")
        return out


CUSTOMER = ConversionType(
    "customer", "Customer", "create_customer", "Customer", 1, False, False, False,
    CUSTOMER_FORM, frozenset({*_PERSON, *_ADDRESS_REQUIRED}), auto_create=True,
)
B2B_CLIENT = ConversionType(
    "b2b_client", "B2B Client (Insurance Carrier)", "create_b2b_client", "InsuranceCarrier", 4, True, True, False,
    COMPANY_FORM, frozenset({"companyName", *_PERSON, *_ADDRESS_REQUIRED}),
)
_PARTNER_REQUIRED = frozenset({"companyName", *_PERSON, *_ADDRESS_REQUIRED})
_PARTNER_UPLOADS = ("logo", "headshot")  # the partner screen's Company Logo / User Headshot
INSURANCE_AGENCY = ConversionType(
    "insurance_agency", "Partner — Insurance Agency", "create_partner", "InsuranceAgency", 2, True, True, False,
    COMPANY_FORM, _PARTNER_REQUIRED, uploads=_PARTNER_UPLOADS,
)
REAL_ESTATE_AGENCY = ConversionType(
    "real_estate_agency", "Partner — Real Estate Agency", "create_partner", "RealEstateCompany", 3, True, True, False,
    COMPANY_FORM, _PARTNER_REQUIRED, uploads=_PARTNER_UPLOADS,
)
CONTRACTOR = ConversionType(
    "contractor", "Partner — Contractor", "create_partner", "Contractor", 5, True, True, True,
    COMPANY_FORM, _PARTNER_REQUIRED, uploads=_PARTNER_UPLOADS,
)
# The Partner form while its company type is still being chosen: the fields every partner
# type shares, for the card and for details given meanwhile. Not a conversion type (not in
# CONVERSION_TYPES, no role): nothing can be created from it.
PARTNER_FORM = ConversionType(
    "partner", "Partner", "create_partner", "", 0, True, False, False,
    COMPANY_FORM, _PARTNER_REQUIRED, uploads=_PARTNER_UPLOADS,
)

CONVERSION_TYPES: dict[str, ConversionType] = {
    t.key: t for t in (CUSTOMER, B2B_CLIENT, INSURANCE_AGENCY, REAL_ESTATE_AGENCY, CONTRACTOR)
}
PARTNER_TYPES: list[ConversionType] = [INSURANCE_AGENCY, REAL_ESTATE_AGENCY, CONTRACTOR]
ALL_TYPES: list[ConversionType] = [CUSTOMER, B2B_CLIENT, INSURANCE_AGENCY, REAL_ESTATE_AGENCY, CONTRACTOR]
# "convert_request": the user wants a conversion but hasn't said which type (asked).
CONVERSION_OPERATIONS = {"create_customer", "create_b2b_client", "create_partner", "convert_request"}
OPERATION_TYPE = {"create_customer": "customer", "create_b2b_client": "b2b_client"}  # partner: picked by the user

STEP_LABELS: dict[str, str] = {
    "create_account": "Account created",
    "verify_account": "Account verified",
    "link_provider_vendor": "Linked to the service provider",
    "save_subcategories": "Service categories saved",
    "update_request_status": "Request marked CRM Completed",
}


def conversion_type(key: str | None) -> ConversionType | None:
    return CONVERSION_TYPES.get(key or "")


def partner_type_name(ctype: ConversionType) -> str:
    """"Contractor" for "Partner — Contractor": the company type as the partner screen names it."""
    return ctype.label.removeprefix("Partner — ")


def all_field_keys() -> list[str]:
    return list(FIELD_SPECS)


# --- draft and validation ------------------------------------------------------------

# Patterns from the host app (agency-popup / customer-detail-popup validation).
_EMAIL = re.compile(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}$")
_URL = re.compile(r"^(https?://)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(:\d+)?(/\S*)?$")
_HTML = re.compile(r"<.*?>")  # Bootog's [NoHtml] rejects these with HTTP 400


# Answers that stand in for a value without being one ("Zipcode: Needs verification").
_PLACEHOLDER = re.compile(
    r"^(n/?a|none|nil|null|unknown|not known|tbd|tbc|to be (confirmed|determined|verified)|"
    r"needs? (verification|to be (checked|verified|confirmed))|(pending|awaiting) verification|"
    r"pending|not (available|provided|sure|applicable)|missing|empty|[?.\-_ ]+)$",
    re.IGNORECASE,
)


def is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and bool(_PLACEHOLDER.match(" ".join(value.split())))


def _clean(value: Any) -> str | None:
    """A usable value or None. Placeholders count as missing, never as data."""
    if value is None or isinstance(value, bool):
        return None
    text = " ".join(str(value).split())
    return None if not text or is_placeholder(text) else text


def build_draft(ctype: ConversionType, record: dict[str, Any], user_values: dict[str, str]) -> dict[str, str | None]:
    """Request data overlaid with what the user supplied (non-empty values win)."""
    draft: dict[str, str | None] = {}
    for key in ctype.fields:
        spec = FIELD_SPECS[key]
        value = _clean(record.get(spec.source)) if spec.source else None
        if spec.kind in ("latitude", "longitude") and value is not None and _number(value) in (None, 0.0):
            value = None  # 0/0 is "not geocoded", not a location
        draft[key] = value
    for key, value in (user_values or {}).items():
        if key in draft and _clean(value):
            draft[key] = _clean(value)
    return draft


def validate_draft(
    ctype: ConversionType, draft: dict[str, str | None]
) -> tuple[dict[str, str | None], list[str], dict[str, str]]:
    """-> (normalized draft, missing required labels, {label: why invalid})."""
    normalized = dict(draft)
    missing = [FIELD_LABELS[k] for k in ctype.fields if k in ctype.required and not draft.get(k)]
    invalid: dict[str, str] = {}
    for key in ctype.fields:
        value = draft.get(key)
        if not value:
            continue
        label, kind = FIELD_LABELS[key], FIELD_SPECS[key].kind
        if _HTML.search(value):
            invalid[label] = "must not contain HTML tags"
        elif kind == "email" and not _EMAIL.match(value):
            invalid[label] = "not a valid email address"
        elif kind == "phone":
            formatted = format_phone(value)
            if formatted is None:
                invalid[label] = "must be a 10-digit phone number"
            else:
                normalized[key] = formatted
        elif kind == "url" and not _URL.match(value):
            invalid[label] = "not a valid website address"
        elif kind == "integer":
            if not re.fullmatch(r"\d{1,3}", value.strip()):
                invalid[label] = "must be a whole number of years"
        elif kind == "money":
            amount = _number(value.replace("$", "").replace(",", ""))
            if amount is None or amount < 0:
                invalid[label] = "must be an amount in dollars"
            else:
                normalized[key] = str(int(amount)) if amount.is_integer() else f"{amount:.2f}"
        elif kind in ("latitude", "longitude"):
            number = _number(value)
            limit = 90 if kind == "latitude" else 180
            if number is None or not -limit <= number <= limit:
                invalid[label] = f"must be a number between -{limit} and {limit}"
        elif kind == "country":
            normalized[key] = normalize_country(value)
    us = normalized.get("country") == "US"
    if draft.get("state"):
        state = normalize_state(draft["state"]) if us else draft["state"]
        if state is None:
            invalid["State"] = "not a US state; use its name or 2-letter code"
        else:
            normalized["state"] = state
    if draft.get("zipcode"):
        zipcode = draft["zipcode"].strip()
        if us and not re.fullmatch(r"\d{5}(-\d{4})?", zipcode):
            invalid["Zipcode"] = "must be a 5-digit US ZIP code"
        elif not us and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 -]{1,9}", zipcode):
            invalid["Zipcode"] = "not a valid postal code"
    # A browser's saved-address autofill can put the state code into County ("IL").
    if us and (draft.get("county") or "").strip().rstrip(".").upper() in _US_STATES:
        invalid["County"] = "must be the county name, not a state code"
    has_lat, has_lng = bool(draft.get("latitude")), bool(draft.get("longitude"))
    if "latitude" in ctype.fields and has_lat != has_lng:
        invalid["Latitude" if not has_lat else "Longitude"] = "latitude and longitude must be given together"
    return normalized, missing, invalid


_US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico", "GU": "Guam",
    "VI": "U.S. Virgin Islands", "AS": "American Samoa", "MP": "Northern Mariana Islands",
}
_US_STATE_BY_NAME = {name.casefold(): code for code, name in _US_STATES.items()}
_US_NAMES = {"us", "usa", "u.s.", "u.s.a.", "united states", "united states of america", "america"}


def normalize_country(value: str) -> str:
    """US spellings ("United States (USA)", "USA", ...) -> "US", the short code the host
    app's address picker (Google Places short_name) sends; other countries unchanged."""
    text = " ".join(value.split())
    parts = [text, *re.findall(r"\(([^)]*)\)", text), re.sub(r"\s*\([^)]*\)", "", text)]
    return "US" if any(p.strip().casefold() in _US_NAMES for p in parts) else text


def normalize_state(value: str) -> str | None:
    """US state name / code / "Florida (FL)" -> "FL"; None when it is not a US state."""
    text = " ".join(value.split())
    candidates = [text, *re.findall(r"\(([^)]*)\)", text), re.sub(r"\s*\([^)]*\)", "", text)]
    for candidate in candidates:
        candidate = candidate.strip().rstrip(".")
        if candidate.upper() in _US_STATES:
            return candidate.upper()
        if candidate.casefold() in _US_STATE_BY_NAME:
            return _US_STATE_BY_NAME[candidate.casefold()]
    return None


def format_phone(value: str) -> str | None:
    """The host app's phone mask: (xxx) xxx-xxxx (10 digits; a leading US 1 is dropped)."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def _number(value: str | None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# --- payload ---------------------------------------------------------------------------


def build_vendor_user_form(
    ctype: ConversionType,
    draft: dict[str, str | None],
    role_id: str,
    record: dict[str, Any],
    password: str,
) -> dict[str, Any]:
    """AddVendorUser FormData, key for key as UserDetailService.createNewUser builds it.
    Optional values the form leaves empty are sent as "" (the host does the same);
    missing coordinates are sent empty (-> null), never invented."""
    company = ctype.skip_tenant
    return {
        "IsLocal": False,
        "fileUrl": "",
        # Customer popup sends 0; the company form sends its (possibly empty) input.
        "AnnualRevenue": (draft.get("annualRevenue") or "") if company else "0",
        "RoleId": role_id,
        "ProviderBio": (draft.get("companyBio") or "") if company else "",
        "Email": draft.get("email") or "",
        "Website": (draft.get("website") or "") if company else "",
        "ProviderName": (draft.get("companyName") or "") if company else "",
        "FirstName": draft.get("firstName") or "",
        "LastName": draft.get("lastName") or "",
        "PhoneNumber": draft.get("phone") or "",
        "IsSkipTenant": ctype.skip_tenant,
        "PasswordConfirm": password,
        "Password": password,
        "ProviderType": ctype.provider_type,
        "YearsInBusiness": (draft.get("yearsInBusiness") or "0") if company else "0",
        "ContactPerson": "",
        "Address1": draft.get("address1") or "",
        "Address2": draft.get("address2") or "",
        "Zipcode": draft.get("zipcode") or "",
        "County": draft.get("county") or "",
        "City": draft.get("city") or "",
        "State": draft.get("state") or "",
        "Country": draft.get("country") or "",
        "jobTitle": "",
        # The customer popup carries the request's subdivision; the company form never sets it.
        "SubDivision": (_clean(record.get("subDivision")) or "") if not company else "",
        "Latitude": draft.get("latitude") or "",
        "Longitude": draft.get("longitude") or "",
    }


# --- uploads ---------------------------------------------------------------------------
# Company Logo / User Headshot. The host app has no separate upload call: createNewUser
# attaches them to the AddVendorUser FormData itself ("LogoFile.FormFile" +
# "LogoFile.Encrypted", "HeadshotFile.FormFile" + "HeadshotFile.Encrypted"), and BOOTOG
# stores them; `fileUrl` stays "" either way. The host accepts .jpeg/.jpg/.png. The size
# cap is this agent's own: an image is held in the conversation until the account is
# created, then only its name and size are kept.


@dataclass(frozen=True)
class UploadSpec:
    key: str
    label: str
    form_name: str  # AddVendorUser multipart prefix


UPLOAD_SPECS: dict[str, UploadSpec] = {
    u.key: u for u in [UploadSpec("logo", "Company Logo", "LogoFile"), UploadSpec("headshot", "User Headshot", "HeadshotFile")]
}
UPLOAD_MAX_BYTES = 2 * 1024 * 1024
UPLOAD_EXTENSIONS = ("jpg", "jpeg", "png")
# The type is read from the content, never taken from the name or the browser.
_IMAGE_SIGNATURES = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG\r\n\x1a\n": "image/png"}


def read_upload(value: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """A submitted image ({name, data: base64}) -> (stored upload, None) or (None, why not)."""
    name = " ".join(str(value.get("name") or "").replace("\\", "/").rsplit("/", 1)[-1].split())[:120]
    if name.rsplit(".", 1)[-1].casefold() not in UPLOAD_EXTENSIONS or "." not in name:
        return None, "must be a .jpg, .jpeg or .png image"
    try:
        data = base64.b64decode(str(value.get("data") or ""), validate=True)
    except (binascii.Error, ValueError):
        return None, "could not be read"
    if not data:
        return None, "is empty"
    if len(data) > UPLOAD_MAX_BYTES:
        return None, f"must be {UPLOAD_MAX_BYTES // (1024 * 1024)} MB or smaller"
    content_type = next((t for sig, t in _IMAGE_SIGNATURES.items() if data.startswith(sig)), None)
    if content_type is None:
        return None, "is not a JPEG or PNG image"
    return {
        "name": name,
        "content_type": content_type,
        "size": len(data),
        "sha": hashlib.sha256(data).hexdigest()[:24],
        "data": base64.b64encode(data).decode(),
    }, None


def vendor_user_uploads(uploads: dict[str, dict[str, Any]]) -> tuple[dict[str, str], dict[str, tuple[str, bytes, str]]]:
    """-> (extra AddVendorUser fields, file parts), as createNewUser appends them. Uploads
    without content (already sent) are skipped."""
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes, str]] = {}
    for key, upload in uploads.items():
        spec = UPLOAD_SPECS.get(key)
        if spec is None or not upload.get("data"):
            continue
        files[f"{spec.form_name}.FormFile"] = (upload["name"], base64.b64decode(upload["data"]), upload["content_type"])
        fields[f"{spec.form_name}.Encrypted"] = "false"
    return fields, files


_SPECIALS = "@#$%&"


def generate_password(length: int = 16) -> str:
    """Cryptographically random; always has upper, lower, digit and special characters
    (a superset of what the host app generates, accepted by the Auth0 policy)."""
    rng = secrets.SystemRandom()
    pools = [string.ascii_uppercase, string.ascii_lowercase, string.digits, _SPECIALS]
    chars = [secrets.choice(pool) for pool in pools for _ in range(2)]
    everything = "".join(pools)
    chars += [secrets.choice(everything) for _ in range(max(length, 12) - len(chars))]
    rng.shuffle(chars)
    return "".join(chars)


def fingerprint(*parts: Any) -> str:
    """Stable digest of what the user confirmed; any change requires a new confirmation."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]
