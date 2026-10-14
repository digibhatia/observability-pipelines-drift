"""
Nidan's four seeded incidents, carried over UNCHANGED (1-4), plus a fifth,
new to Satark, that exists specifically to give M5's alerting layer
something real to catch: a silently swapped model version.

WHY INCIDENT 5 IS DIFFERENT FROM 1-4:
Incidents 1-4 are each diagnosed from a SINGLE trace - you can see the whole
story in one trace_id. Incident 5 cannot be diagnosed that way on purpose:
a model swap doesn't break any one request, it shifts a DISTRIBUTION over
many requests (cost per call jumps, answer style drifts slightly, latency
changes). That's exactly the gap between Nidan's trace-level diagnosis and
Satark's production-evaluation layer - online_eval.py and drift.py exist
because "read one trace" stops being enough once the failure is statistical
rather than structural. alerting.py is what notices it without a human
reading every trace by hand.

HOW THIS IS MEANT TO BE RUN (same discipline as Nidan):
A facilitator calls seed_incident(n) for an n they choose, without saying
which. lab.incident.active stays true/false-only on every span - never which
incident number. For incidents 1-4, diagnose from a trace in Jaeger. For
incident 5, diagnose from the Grafana dashboard and /alerts - read enough
production traffic for the drift/eval signal to move, THEN check
reveal_incident(5).

THE FIVE INCIDENTS:
  1. retrieval_stale_lock     - (Nidan, unchanged) lookup_policy pinned to
                                  'stale' mode regardless of the request.
  2. tool_misrouting          - (Nidan, unchanged) calculate_monthly_repayment
                                  fires for every question; a trajectory
                                  failure invisible to an outcome-only check.
  3. delegation_loop          - (Nidan, unchanged) the fraud sub-agent never
                                  resolves, looping inside its own
                                  sub_agent_turn span.
  4. session_crosstalk        - (Nidan, unchanged) two session_ids served
                                  from the same underlying Session object.
  5. silent_model_swap        - (NEW) every chat_raw() call is silently
                                  served by gpt-4o instead of the requested
                                  gpt-4o-mini - no error, no log line, just a
                                  ~17x per-token cost jump and a quality/style
                                  shift that only shows up once you look at
                                  several requests together: the cost
                                  dashboard (M4) and the online-eval/drift
                                  signal (M5), never a single trace in
                                  isolation.
"""
from dataclasses import dataclass

CURRENT_INCIDENT = {"n": None}  # None = no incident seeded (normal operation)

_ANSWER_KEY = {
    1: {
        "name": "retrieval_stale_lock",
        "root_cause": (
            "lookup_policy's mode argument is being silently overridden to 'stale' "
            "regardless of what the caller passed, so every retrieval - even ones the "
            "agent intended to run normally - serves the outdated 24.9% EAR fixture "
            "instead of the live 19.9% figure."
        ),
        "remediation": (
            "Remove the override; lookup_policy should only use a non-normal mode when "
            "explicitly requested by the caller, never as a hidden default."
        ),
    },
    2: {
        "name": "tool_misrouting",
        "root_cause": (
            "The agent is calling calculate_monthly_repayment for questions that need "
            "lookup_policy, producing a well-formed, plausible-looking number that has "
            "nothing to do with the question asked - a trajectory failure invisible to "
            "any outcome-only check."
        ),
        "remediation": (
            "Tighten the tool descriptions in TOOL_SPECS, and rely on trajectory_eval.py's "
            "automatic trajectory check rather than a human reading every answer."
        ),
    },
    3: {
        "name": "delegation_loop",
        "root_cause": (
            "The fraud referral sub-agent never produces a tool-call-free turn, so it "
            "keeps calling lookup_policy indefinitely inside its own sub_agent_turn span - "
            "a runaway loop one level DOWN in the span hierarchy."
        ),
        "remediation": (
            "Give the sub-agent its own step cap, separate from the main agent's, and "
            "export a sub-agent-step-count attribute so this is visible without expanding "
            "every nested span by hand."
        ),
    },
    4: {
        "name": "session_crosstalk",
        "root_cause": (
            "memory_store.py's get_or_create() returned the wrong Session object - two "
            "different session_id values were sharing one Session instance, so a fact "
            "from customer A's conversation appeared in customer B's answer."
        ),
        "remediation": (
            "Add an assertion that get_or_create(session_id) always returns a Session "
            "whose .session_id matches the id requested, and keep session_id as a span "
            "attribute on every /chat trace."
        ),
    },
    5: {
        "name": "silent_model_swap",
        "root_cause": (
            "chat_raw() is being served by gpt-4o instead of the gpt-4o-mini every caller "
            "requested, with no error and no visible change to a single trace's shape - "
            "gen_ai.request.model still reads 'gpt-4o-mini' because that's what the CALLER "
            "asked for; only gen_ai.response.model and the per-call cost actually move. "
            "The effect only becomes visible as a step change in lab.cost_usd across many "
            "requests (the M4 cost panel) and as a small but real shift in the online-eval "
            "score distribution and output-embedding drift score (M5) - no single trace "
            "looks wrong in isolation, which is the whole point of this incident."
        ),
        "remediation": (
            "Alert on gen_ai.response.model disagreeing with gen_ai.request.model, not just "
            "on cost or eval score drifting - that is the one attribute that is still "
            "ground-truth even when the swap is otherwise invisible. Treat a version "
            "mismatch between request and response as its own alert class, independent of "
            "whatever quality or cost signal eventually also moves."
        ),
    },
}


def seed_incident(n: int):
    if n not in _ANSWER_KEY:
        raise ValueError(f"no such incident: {n}. Valid: {sorted(_ANSWER_KEY)}")
    CURRENT_INCIDENT["n"] = n


def clear_incident():
    CURRENT_INCIDENT["n"] = None


def effective_retrieval_mode(requested_mode: str) -> str:
    """Incident 1's hook - called from tools.py."""
    if CURRENT_INCIDENT["n"] == 1:
        return "stale"
    return requested_mode


def should_misroute_to_repayment_tool() -> bool:
    """Incident 2's hook - checked from app.py's main agent loop."""
    return CURRENT_INCIDENT["n"] == 2


def delegation_should_loop_forever() -> bool:
    """Incident 3's hook - checked from delegation.py."""
    return CURRENT_INCIDENT["n"] == 3


def session_crosstalk_active() -> bool:
    """Incident 4's hook - checked from memory_store.py's get_or_create()."""
    return CURRENT_INCIDENT["n"] == 4


def effective_chat_model(requested_model: str) -> str:
    """Incident 5's hook - called from llm_provider.py's chat_raw(). Returns
    the model that will ACTUALLY be called, which on incident 5 is not the
    model the caller asked for. gen_ai.request.model is set from
    `requested_model` (what the caller believes), gen_ai.response.model from
    the real API response (what actually ran) - so this incident is visible
    only by comparing those two attributes, or by watching cost/quality move
    in aggregate. Never surfaced as an error, a log line, or a single-trace
    anomaly."""
    if CURRENT_INCIDENT["n"] == 5 and requested_model == "gpt-4o-mini":
        return "gpt-4o"
    return requested_model


def reveal_incident(n: int) -> dict:
    """The answer key. Deliberately a SEPARATE function from seed_incident -
    call this only after you've written your own diagnosis down."""
    return _ANSWER_KEY[n]


def incident_is_active() -> bool:
    return CURRENT_INCIDENT["n"] is not None
