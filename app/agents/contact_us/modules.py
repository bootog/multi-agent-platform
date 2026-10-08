"""Contact agents = the submenus of BOOTOG's "Contact" menu (MenuRole/GetMenus).

Each submenu's route segment (e.g. /dashboard/contact/Request-Demo -> "Request-Demo")
identifies the agent. All of them work on Contact Us records (GET /ContactUs); the
segment only decides which `moduleName` records belong to the agent.

The UI never takes its agent list from here: the menu API is the source of truth.
This module only knows how to scope GET /ContactUs for a given segment."""

import re

DEFAULT_AGENT_KEY = "Contact-Us"

# Same module clauses the host's Contact screen applies per route
# (master/.../contact-us/contact-us.component.ts, filterString()).
_KNOWN_MODULE_FILTERS: dict[str, str] = {
    "contact-us": "(moduleName=Contact-Us|moduleName=contact)",
    "request-demo": "(moduleName=Request Demo|moduleName=Request-Demo)",
    "request-proposal": "(moduleName=Request-Proposal)",
    "early-access": "(moduleName=Early Access|moduleName=Early-Access)",
    "request-training": "(moduleName=Resume-Submission:Training)",
    "resume-submission": (
        "(moduleName=Resume-Submission:Job|moduleName=Resume-Submission:Program|moduleName=Resume-Submission:Roster)"
    ),
}

# Route segments only: letters, digits, spaces, "-" and "_". Anything else could alter
# the filter expression, so it is rejected before it gets here (see schemas).
AGENT_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$"
_KEY = re.compile(AGENT_KEY_PATTERN)


def canonical_agent_key(agent_key: str | None) -> str:
    """Stable comparison form: "Request-Demo", "request demo" -> "request-demo"."""
    key = (agent_key or DEFAULT_AGENT_KEY).strip()
    if not _KEY.match(key):
        raise ValueError("invalid agent key")
    return re.sub(r"[\s_]+", "-", key).casefold()


def module_filter(agent_key: str | None) -> str:
    """The `moduleName` clause for an agent. A submenu added to the menu later gets the
    convention most Contact modules follow: its segment, with or without hyphens."""
    canonical = canonical_agent_key(agent_key)
    if canonical in _KNOWN_MODULE_FILTERS:
        return _KNOWN_MODULE_FILTERS[canonical]
    segment = (agent_key or DEFAULT_AGENT_KEY).strip()
    spaced = re.sub(r"[-_]+", " ", segment)
    return f"(moduleName={segment}|moduleName={spaced})" if spaced != segment else f"(moduleName={segment})"


def agent_name(agent_key: str | None, label: str | None) -> str:
    """Display name, e.g. "Request Demo Agent" (from the menu label when given)."""
    base = re.sub(r"[-_]+", " ", (label or agent_key or DEFAULT_AGENT_KEY)).strip()
    base = re.sub(r"\s+", " ", base)[:60]
    return base if base.casefold().endswith("agent") else f"{base} Agent"
