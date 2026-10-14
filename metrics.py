"""
M4's actual instruments. otel_setup.py decided WHERE metrics go; this file
decides WHAT gets measured - the three things a duty engineer on an
incident call actually wants on one screen: quality, cost, latency. Every
instrument here is created once (module load) and recorded from exactly one
place: app.py's _run_traced_agent_turn(), right after a turn finishes, so
there is one call site to read instead of metrics scattered across every
module that happens to touch a request.

Why counters/histograms instead of just reading spans back out of Jaeger:
a trace backend is built to answer "what happened on THIS request" - it is
a poor tool for "what's the p95 latency over the last 15 minutes across
10,000 requests". That aggregation is exactly what a metrics pipeline (here:
OTel Collector -> Prometheus -> Grafana) is optimised for, and it is the
concrete answer to M4's own question: "the platform landscape and what each
is actually optimising for." Jaeger: one request, deeply. Prometheus/
Grafana: every request, shallowly, over time.
"""
from otel_setup import get_meter

_meter = get_meter("satark.metrics")

request_counter = _meter.create_counter(
    "satark.requests.total",
    description="Agent turns completed, labelled by outcome/trajectory verdict and model.",
)

latency_histogram = _meter.create_histogram(
    "satark.request.duration_ms",
    description="Wall-clock time for one agent turn (all model + tool calls inside it).",
    unit="ms",
)

cost_histogram = _meter.create_histogram(
    "satark.request.cost_usd",
    description="Total model-call cost for one agent turn.",
    unit="usd",
)

trajectory_failure_counter = _meter.create_counter(
    "satark.trajectory_failures.total",
    description="Turns where the outcome looked fine but the wrong tool (or tool order) was used.",
)

outcome_failure_counter = _meter.create_counter(
    "satark.outcome_failures.total",
    description="Turns whose final answer did not pass the outcome check.",
)

eval_score_histogram = _meter.create_histogram(
    "satark.eval.score",
    description="Sampled online-eval judge score (1-5) for live traffic (see online_eval.py).",
)

drift_score_histogram = _meter.create_histogram(
    "satark.drift.score",
    description="Distance from the reference baseline for the current drift window (see drift.py).",
)

guardrail_counter = _meter.create_counter(
    "satark.guardrail_triggers.total",
    description="Turns where a guardrail or refusal fired.",
)

feedback_counter = _meter.create_counter(
    "satark.feedback.total",
    description="Human feedback events received, labelled up/down.",
)


def record_agent_turn(*, model: str, duration_ms: float, cost_usd: float,
                       outcome_correct: bool = None, trajectory_correct: bool = None,
                       guardrail_triggered: bool = False):
    """The one call site every agent-turn path (app.py, replay.py-driven
    runs, delegation.py) should go through after a turn completes. Keeping
    it as one function - rather than six separate .add()/.record() calls
    sprinkled through app.py - is what makes it possible to add a seventh
    metric later without re-auditing every call site."""
    attrs = {"gen_ai.response.model": model}
    request_counter.add(1, attrs)
    latency_histogram.record(duration_ms, attrs)
    cost_histogram.record(cost_usd, attrs)

    if trajectory_correct is False:
        trajectory_failure_counter.add(1, attrs)
    if outcome_correct is False:
        outcome_failure_counter.add(1, attrs)
    if guardrail_triggered:
        guardrail_counter.add(1, attrs)


def record_eval_score(score: float, model: str):
    eval_score_histogram.record(score, {"gen_ai.response.model": model})


def record_drift_score(score: float, dimension: str):
    drift_score_histogram.record(score, {"satark.drift.dimension": dimension})


def record_feedback(rating: str):
    feedback_counter.add(1, {"satark.feedback.rating": rating})
