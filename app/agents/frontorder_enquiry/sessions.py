"""Conversations of the Frontend Order Enquiry Agent.

This reuses the platform's existing session mechanism instead of adding a new one:
- `ChatSessions` (the Contact and Scheduling agents' registry): which caller owns a
  conversation, one message at a time, and expiry of idle conversations (4 hours);
- an in-memory LangGraph checkpointer: one thread per conversation, holding the graph
  state between messages (the `scope`: provider, dates, kind of enquiries, page).

Only the final state of each message is checkpointed (graph.py invokes with
durability="exit"), and the `respond` node removes raw enquiry records first, so a
checkpoint holds ids, scope and the display result — never credentials or customer
contact details. Checkpoints are conversation memory, not an audit trail; runs are
audited in `agent_run_audit` (monitoring.py).

Porting this agent to another backend: replace these imports with that backend's own
session registry and checkpointer (any object with create/owner_of/begin/end and any
LangGraph checkpointer)."""

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.contact_us.agent_graph import ChatSessions, owner_fingerprint

AGENT_KEY = "frontorder-enquiry"

CHECKPOINTER = InMemorySaver()
sessions = ChatSessions(CHECKPOINTER)

__all__ = ["AGENT_KEY", "CHECKPOINTER", "owner_fingerprint", "sessions"]
