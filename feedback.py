"""
M5, task "human feedback capture and annotation queues."

Two different audiences for the same underlying data:
  - record_feedback()  - a customer (or, in this lab, whoever calls
                          /feedback) attaches a thumbs up/down and an
                          optional note to a specific trace_id. This is the
                          cheapest, most direct quality signal there is -
                          no judge model, no sampling, just a real person
                          saying whether the answer was good.
  - annotation_queue()  - what a HUMAN REVIEWER sees: the turns most worth
                          a second look, combining explicit negative
                          feedback with online_eval.py's own low-scoring
                          sampled turns, so a reviewer isn't stuck choosing
                          between "only look at complaints" and "only look
                          at what the judge model already flagged."
"""
import time
from collections import deque

from otel_setup import get_tracer
from genai_conventions import LAB_FEEDBACK_RATING, LAB_FEEDBACK_HAS_NOTE
import metrics
import online_eval

_FEEDBACK = deque(maxlen=500)


def record_feedback(trace_id: str, rating: str, note: str = None, question: str = None, answer: str = None):
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")

    tracer = get_tracer("satark.feedback")
    with tracer.start_as_current_span("human_feedback") as span:
        span.set_attribute(LAB_FEEDBACK_RATING, rating)
        span.set_attribute(LAB_FEEDBACK_HAS_NOTE, bool(note))
        if trace_id:
            span.set_attribute("satark.feedback.trace_id", trace_id)

        metrics.record_feedback(rating)
        entry = {
            "trace_id": trace_id, "rating": rating, "note": note,
            "question": (question or "")[:200], "answer": (answer or "")[:300],
            "ts": time.time(),
        }
        _FEEDBACK.appendleft(entry)
        return entry


def recent_feedback(limit: int = 20) -> list:
    return list(_FEEDBACK)[:limit]


def annotation_queue(limit: int = 20) -> list:
    """Merges explicit thumbs-down feedback with online_eval.py's own
    low-scoring sampled turns into one reviewer-facing queue, newest first,
    de-duplicated by trace_id where one is available. This is the concrete
    answer to 'annotation queue' - a human reviewer works this list, not
    the raw trace stream."""
    queue = []
    seen_trace_ids = set()

    for entry in _FEEDBACK:
        if entry["rating"] == "down":
            queue.append({**entry, "reason": "negative human feedback"})
            if entry["trace_id"]:
                seen_trace_ids.add(entry["trace_id"])

    for judged in online_eval.recent_judged(limit=limit * 2):
        if judged["verdict"] == "fail" and judged["trace_id"] not in seen_trace_ids:
            queue.append({
                "trace_id": judged["trace_id"], "rating": None, "note": None,
                "question": judged["question"], "answer": judged["answer"],
                "ts": None, "reason": f"online-eval score {judged['score']}: {judged['rationale']}",
            })

    return queue[:limit]
