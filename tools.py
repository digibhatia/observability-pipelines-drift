"""lookup_policy (vector search) and calculate_monthly_repayment, each wrapped
in its own tool-call span. Carried over from Drishti; the retrieval modes are
reused as raw material for Nidan's seeded incidents (see incidents.py)."""
from otel_setup import get_tracer
from genai_conventions import (
    GEN_AI_OPERATION_NAME, GEN_AI_TOOL_NAME, LAB_RETRIEVAL_MODE, LAB_RETRIEVAL_HIT_COUNT,
)
from vectorstore import build_or_load_collection, query as vs_query
from context import pack_context

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        _collection = build_or_load_collection()
    return _collection


_STALE_SNAPSHOT = (
    "Arranged overdrafts on a Premier Current Account carry an Effective "
    "Annual Rate (EAR) of 24.9%. [This figure is a deliberately stale fixture "
    "for the 'stale' retrieval mode - the live document says 19.9%.]"
)


def lookup_policy(query: str, mode: str = "normal", n_results: int = 6) -> str:
    """mode: normal | stale | empty | oversized - failure classes triggered on demand.
    In Nidan, 'mode' can also be forced by incidents.py's seeded-incident state,
    without the caller explicitly asking for it - see incidents.CURRENT_INCIDENT."""
    import incidents  # local import avoids a circular import at module load time
    effective_mode = incidents.effective_retrieval_mode(mode)

    tracer = get_tracer("nidan.tool")
    with tracer.start_as_current_span("tool_call") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")
        span.set_attribute(GEN_AI_TOOL_NAME, "lookup_policy")
        span.set_attribute(LAB_RETRIEVAL_MODE, effective_mode)

        if effective_mode == "stale":
            span.set_attribute(LAB_RETRIEVAL_HIT_COUNT, 1)
            return _STALE_SNAPSHOT
        if effective_mode == "empty":
            span.set_attribute(LAB_RETRIEVAL_HIT_COUNT, 0)
            return pack_context([])
        if effective_mode == "oversized":
            collection = _get_collection()
            n = min(50, collection.count())
            hits = vs_query(collection, query, n_results=n)
            span.set_attribute(LAB_RETRIEVAL_HIT_COUNT, len(hits))
            return "\n---\n".join(h["text"] for h in hits)

        collection = _get_collection()
        hits = vs_query(collection, query, n_results=n_results)
        span.set_attribute(LAB_RETRIEVAL_HIT_COUNT, len(hits))
        return pack_context(hits)


def calculate_monthly_repayment(principal: float, annual_rate_pct: float, years: int) -> float:
    tracer = get_tracer("nidan.tool")
    with tracer.start_as_current_span("tool_call") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")
        span.set_attribute(GEN_AI_TOOL_NAME, "calculate_monthly_repayment")

        monthly_rate = (annual_rate_pct / 100) / 12
        n_payments = years * 12
        if monthly_rate == 0:
            return round(principal / n_payments, 2)
        payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / \
            ((1 + monthly_rate) ** n_payments - 1)
        return round(payment, 2)


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": "Search Lloyds Financial policy/product documents for a relevant passage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_monthly_repayment",
            "description": "Calculate the monthly repayment for a loan/mortgage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {"type": "number"},
                    "annual_rate_pct": {"type": "number"},
                    "years": {"type": "integer"},
                },
                "required": ["principal", "annual_rate_pct", "years"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "lookup_policy": lookup_policy,
    "calculate_monthly_repayment": calculate_monthly_repayment,
}
