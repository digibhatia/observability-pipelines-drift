"""
M5, task 2: alerting that fires on quality drift - and the specific thing
it must catch is incident 5 (silent_model_swap, see incidents.py), which is
deliberately invisible to a single trace. This module is the piece that
turns online_eval.py's rolling score and drift.py's drift score into an
actual alert a duty engineer would see, instead of a number nobody is
watching.

FOUR ALERT RULES, each catching a DIFFERENT shape of the same underlying
problem a silent model swap can cause:
  - model_version_mismatch  - gen_ai.request.model != gen_ai.response.model
                               on a recorded turn. The most direct possible
                               signal, called out explicitly in incident 5's
                               remediation note, because it is still
                               ground-truth even when cost/quality signals
                               are ambiguous.
  - eval_score_drop         - online_eval.py's rolling_average_score() over
                               the last N judged turns falls below
                               SATARK_EVAL_SCORE_ALERT_THRESHOLD.
  - embedding_drift         - drift.py's embedding drift score crosses
                               SATARK_DRIFT_THRESHOLD.
  - cost_spike               - the mean per-call cost over a short window
                               jumps past SATARK_COST_SPIKE_MULTIPLIER times
                               the baseline mean - the cheapest signal to
                               compute and often the FASTEST one to fire,
                               since cost moves on literally the first
                               swapped call, before enough traffic has
                               accumulated for eval/drift windows to react.

Alerts are appended to an in-memory, capped list - good enough for a lab
exercise; a real system would push these to Slack/PagerDuty/Grafana
Alerting instead of a list a human has to poll.
"""
import os
import time
from collections import deque

import drift
import online_eval

EVAL_SCORE_ALERT_THRESHOLD = float(os.environ.get("SATARK_EVAL_SCORE_ALERT_THRESHOLD", "3.5"))
COST_SPIKE_MULTIPLIER = float(os.environ.get("SATARK_COST_SPIKE_MULTIPLIER", "3.0"))
COST_WINDOW_SIZE = int(os.environ.get("SATARK_COST_WINDOW", "10"))

_alerts = deque(maxlen=100)
_baseline_costs: deque = deque(maxlen=COST_WINDOW_SIZE)  # first N calls -> baseline mean
_recent_costs: deque = deque(maxlen=COST_WINDOW_SIZE)


def _fire(rule: str, severity: str, message: str, detail: dict = None):
    alert = {
        "ts": time.time(), "rule": rule, "severity": severity,
        "message": message, "detail": detail or {},
    }
    _alerts.appendleft(alert)
    return alert


def check_model_version_mismatch(requested_model: str, served_model: str, trace_id: str = None):
    if requested_model and served_model and requested_model != served_model:
        return _fire(
            "model_version_mismatch", "critical",
            f"Served model '{served_model}' does not match the requested model "
            f"'{requested_model}' - a silent model swap is the most likely cause.",
            {"requested_model": requested_model, "served_model": served_model, "trace_id": trace_id},
        )
    return None


def check_eval_score_drop(window: int = 20):
    avg = online_eval.rolling_average_score(window=window)
    if avg is not None and avg < EVAL_SCORE_ALERT_THRESHOLD:
        return _fire(
            "eval_score_drop", "warning",
            f"Rolling online-eval score ({avg:.2f}) has fallen below the "
            f"{EVAL_SCORE_ALERT_THRESHOLD} threshold over the last {window} judged turns.",
            {"rolling_avg_score": round(avg, 2), "window": window},
        )
    return None


def check_embedding_drift():
    st = drift.status()
    if not st["baseline_ready"]:
        return None
    # drift.observe() already records the score via metrics; alerting just
    # re-checks the most recent flagged state kept in drift.py's own window.
    return None  # the real-time flag is raised inline from observe_and_alert() below


def check_cost_spike(cost_usd: float):
    if len(_baseline_costs) < COST_WINDOW_SIZE:
        _baseline_costs.append(cost_usd)
        return None
    baseline_mean = sum(_baseline_costs) / len(_baseline_costs)
    _recent_costs.append(cost_usd)
    recent_mean = sum(_recent_costs) / len(_recent_costs)
    if baseline_mean > 0 and recent_mean > baseline_mean * COST_SPIKE_MULTIPLIER:
        return _fire(
            "cost_spike", "critical",
            f"Mean per-call cost (${recent_mean:.5f}) is {recent_mean / baseline_mean:.1f}x "
            f"the established baseline (${baseline_mean:.5f}) - check for a model-version swap "
            "before assuming traffic volume changed.",
            {"baseline_mean_usd": round(baseline_mean, 6), "recent_mean_usd": round(recent_mean, 6)},
        )
    return None


def observe_and_alert(*, requested_model: str, served_model: str, cost_usd: float,
                       answer_text: str, tool_sequence: list, trace_id: str = None):
    """The single call app.py makes after every agent turn - runs every
    rule above against this turn's numbers plus the rolling state the
    other modules keep. Returns the list of alerts newly fired this call
    (usually empty)."""
    fired = []

    a = check_model_version_mismatch(requested_model, served_model, trace_id)
    if a:
        fired.append(a)

    c = check_cost_spike(cost_usd)
    if c:
        fired.append(c)

    e = check_eval_score_drop()
    if e:
        fired.append(e)

    drift_result = drift.observe(answer_text, tool_sequence)
    if drift_result.embedding_flagged:
        fired.append(_fire(
            "embedding_drift", "warning",
            f"Output-embedding drift score ({drift_result.embedding_score}) crossed the "
            f"{drift.EMBED_DRIFT_THRESHOLD} threshold against the established baseline.",
            {"drift_score": drift_result.embedding_score},
        ))
    if drift_result.tool_mix_flagged:
        fired.append(_fire(
            "tool_mix_drift", "warning",
            f"Tool-usage distribution has drifted ({drift_result.tool_mix_score}) from the "
            f"established baseline mix.",
            {"tool_mix_score": drift_result.tool_mix_score},
        ))

    return fired


def recent_alerts(limit: int = 20) -> list:
    return list(_alerts)[:limit]


def clear_alerts():
    _alerts.clear()
    _baseline_costs.clear()
    _recent_costs.clear()
