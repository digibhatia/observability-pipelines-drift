"""
M5, task "drift detection across inputs, outputs and embeddings."

THE IDEA: establish a baseline under known-good traffic, then watch how far
new traffic drifts from it. Two different kinds of drift are tracked here
because they catch different failures:

  1. EMBEDDING drift (inputs or outputs) - embed recent answers, compare
     their centroid to a frozen baseline centroid via cosine distance. This
     is what catches incident 5 (silent_model_swap): gpt-4o's answers are
     still correct and on-topic, but subtly different in phrasing/length/
     style from gpt-4o-mini's, which shows up as a small but real increase
     in distance from the baseline centroid - exactly the kind of shift a
     single trace can't show you, because one gpt-4o answer in isolation
     just looks like a slightly different correct answer.

  2. TOOL-MIX drift - the categorical distribution of which tools fire,
     compared against a baseline distribution via total variation distance.
     This catches a DIFFERENT failure than embedding drift would: a prompt
     or routing change that shifts the agent toward calling the wrong tool
     more often, even when each individual answer still reads fine
     (connects back to Nidan's incident 2, but as a trend instead of a
     single misrouted call).

Both are intentionally simple, closed-form statistics (centroid distance,
total variation distance) rather than a full statistical test - good
enough to demonstrate the DETECTION mechanism without pulling in a stats
dependency for a lab.
"""
import os
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional

from otel_setup import get_tracer
from genai_conventions import LAB_DRIFT_SCORE, LAB_DRIFT_FLAGGED, LAB_DRIFT_DIMENSION
import metrics

EMBED_WINDOW_SIZE = int(os.environ.get("SATARK_DRIFT_WINDOW", "15"))
BASELINE_SIZE = int(os.environ.get("SATARK_DRIFT_BASELINE_SIZE", "8"))
EMBED_DRIFT_THRESHOLD = float(os.environ.get("SATARK_DRIFT_THRESHOLD", "0.08"))
TOOL_MIX_DRIFT_THRESHOLD = float(os.environ.get("SATARK_TOOL_MIX_DRIFT_THRESHOLD", "0.35"))

_baseline_embeddings: list = []       # frozen once BASELINE_SIZE is reached
_recent_embeddings: deque = deque(maxlen=EMBED_WINDOW_SIZE)
_baseline_tool_counts: Counter = Counter()
_recent_tool_counts: deque = deque(maxlen=EMBED_WINDOW_SIZE)  # one Counter per turn's tool_sequence

_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from vectorstore import get_embedding_function
        _embed_fn = get_embedding_function()
    return _embed_fn


def _embed(text: str) -> Optional[list]:
    try:
        ef = _get_embed_fn()
        try:
            vecs = ef([text])
        except TypeError:
            vecs = ef(input=[text])
        return list(vecs[0])
    except Exception:
        # Drift detection is a secondary signal - an embedding-API hiccup
        # should never take down the main agent turn it's measuring.
        return None


def _cosine_distance(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cosine_sim = dot / (norm_a * norm_b)
    return 1.0 - cosine_sim


def _centroid(vectors: list) -> list:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def _total_variation_distance(baseline: Counter, recent: Counter) -> float:
    keys = set(baseline) | set(recent)
    b_total = sum(baseline.values()) or 1
    r_total = sum(recent.values()) or 1
    return 0.5 * sum(abs(baseline.get(k, 0) / b_total - recent.get(k, 0) / r_total) for k in keys)


@dataclass
class DriftResult:
    embedding_score: Optional[float] = None
    embedding_flagged: bool = False
    tool_mix_score: Optional[float] = None
    tool_mix_flagged: bool = False
    baseline_ready: bool = False


def observe(answer_text: str, tool_sequence: list) -> DriftResult:
    """Called once per agent turn from app.py, AFTER the answer and tool
    sequence are known. First BASELINE_SIZE calls build the frozen
    baseline; every call after that is scored against it."""
    tracer = get_tracer("satark.drift")
    with tracer.start_as_current_span("drift_check") as span:
        result = DriftResult()
        vec = _embed(answer_text) if answer_text else None

        if vec is not None and len(_baseline_embeddings) < BASELINE_SIZE:
            _baseline_embeddings.append(vec)
            _baseline_tool_counts.update(tool_sequence or ["(none)"])
            result.baseline_ready = len(_baseline_embeddings) >= BASELINE_SIZE
            span.set_attribute("satark.drift.baseline_building", True)
            return result

        result.baseline_ready = True

        if vec is not None and _baseline_embeddings:
            _recent_embeddings.append(vec)
            baseline_centroid = _centroid(_baseline_embeddings)
            recent_centroid = _centroid(list(_recent_embeddings))
            score = _cosine_distance(baseline_centroid, recent_centroid)
            flagged = score > EMBED_DRIFT_THRESHOLD
            result.embedding_score, result.embedding_flagged = round(score, 4), flagged
            metrics.record_drift_score(score, dimension="output")
            span.set_attribute(LAB_DRIFT_SCORE, score)
            span.set_attribute(LAB_DRIFT_FLAGGED, flagged)
            span.set_attribute(LAB_DRIFT_DIMENSION, "output")

        this_turn_tools = Counter(tool_sequence or ["(none)"])
        _recent_tool_counts.append(this_turn_tools)
        if _baseline_tool_counts:
            merged_recent = Counter()
            for c in _recent_tool_counts:
                merged_recent.update(c)
            tv_score = _total_variation_distance(_baseline_tool_counts, merged_recent)
            tv_flagged = tv_score > TOOL_MIX_DRIFT_THRESHOLD
            result.tool_mix_score, result.tool_mix_flagged = round(tv_score, 4), tv_flagged
            metrics.record_drift_score(tv_score, dimension="tool_mix")

        return result


def status() -> dict:
    """What a /drift-status endpoint or the Grafana dashboard's drift panel
    reads - the current state without forcing a new observation."""
    return {
        "baseline_ready": len(_baseline_embeddings) >= BASELINE_SIZE,
        "baseline_size": len(_baseline_embeddings),
        "baseline_target": BASELINE_SIZE,
        "recent_window_size": len(_recent_embeddings),
        "embed_drift_threshold": EMBED_DRIFT_THRESHOLD,
        "tool_mix_drift_threshold": TOOL_MIX_DRIFT_THRESHOLD,
        "baseline_tool_mix": dict(_baseline_tool_counts),
    }


def reset_baseline():
    """Facilitator-only, mirroring incidents.clear_incident()'s role - lets
    a fresh exercise start from a clean baseline instead of one polluted by
    the previous group's traffic."""
    _baseline_embeddings.clear()
    _recent_embeddings.clear()
    _baseline_tool_counts.clear()
    _recent_tool_counts.clear()
