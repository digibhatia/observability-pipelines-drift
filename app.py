"""FastAPI backend for Satark. Same reference agent as Nirikshan/Smriti/
Drishti/Nidan, now additionally wired for production observability (M4:
metrics, dashboards) and production evaluation (M5: online eval, drift,
alerting, human feedback) - the question those two modules ask is no
longer "what did ONE request do" (Nidan's job) but "is the AGENT, in
aggregate, still healthy", which is why almost every new call here runs
AFTER a turn completes rather than inside the turn loop itself."""
import json
import time

from fastapi import FastAPI, Query
from opentelemetry import trace

from otel_setup import get_tracer
from genai_conventions import (
    SESSION_ID, THREAD_ID, ENDUSER_ID, LAB_INCIDENT_ACTIVE,
    LAB_GUARDRAIL_TRIGGERED, LAB_GUARDRAIL_REASON, LAB_REFUSAL_DETECTED,
)
from llm_provider import chat_raw
from tools import TOOL_SPECS, TOOL_FUNCTIONS, calculate_monthly_repayment
from delegation import delegate_to_fraud_check
from memory_store import store
from replay import run_once, diff_runs, replay_n_times
from trajectory_eval import grade_run
import incidents
import metrics
import online_eval
import drift
import alerting
import feedback

app = FastAPI(title="Satark - Observability, Evaluation and Drift Reference Agent")

SYSTEM_PROMPT = (
    "You are a customer service assistant for Lloyds Financial (a training scenario). "
    "For ANY question about rates, limits, eligibility, fraud handling, complaints, or "
    "savings products, you MUST call lookup_policy before writing your answer - this "
    "applies even if the question sounds like it needs account-specific data. Use "
    "calculate_monthly_repayment for repayment arithmetic. Keep answers short."
)
MAX_AGENT_STEPS = 6

_REFUSAL_PHRASES = (
    "i cannot assist", "i can't assist", "i'm unable to help", "i am unable to help",
    "i'm sorry, but i can't", "i'm not able to provide", "as an ai, i cannot",
)


def _detect_guardrail(answer: str) -> tuple[bool, str]:
    """A deliberately simple phrase match standing in for a real guardrail
    service (Bedrock Guardrails, NeMo Guardrails, a moderation API) - good
    enough to demonstrate refusal/fallback telemetry as a quality signal
    without pulling in an external dependency for a lab. Real production
    code would call an actual guardrail/moderation layer here instead."""
    if not answer:
        return False, "empty_answer"
    lowered = answer.lower()
    for phrase in _REFUSAL_PHRASES:
        if phrase in lowered:
            return True, f"refusal_phrase:{phrase}"
    return False, ""


def _run_traced_agent_turn(question: str, session_id: str = None, user_id: str = "anonymous",
                            thread_id: str = None, history: list = None, skip_eval: bool = False):
    """Unchanged from Nidan's own agent loop, PLUS: every call's cost and
    duration are now accumulated (not just the last one), and after the
    turn completes, four new things happen in order - metrics are
    recorded, the guardrail check runs, the turn is (maybe) sampled for
    online eval, and alerting.observe_and_alert() runs every alert rule
    against this turn's numbers plus the rolling state the other modules
    keep. skip_eval=True is used by replay.py-driven paths that already
    call eval/grading explicitly, so a /replay with n=20 doesn't also fire
    20 extra judge calls and 20 extra alert checks."""
    tracer = get_tracer("satark.app")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend({"role": t["role"], "content": t["content"]} for t in history)
    messages.append({"role": "user", "content": question})

    t_start = time.perf_counter()
    with tracer.start_as_current_span("agent_turn") as root_span:
        root_span.set_attribute("lab.question", question[:200])
        root_span.set_attribute(LAB_INCIDENT_ACTIVE, incidents.incident_is_active())
        if session_id:
            root_span.set_attribute(SESSION_ID, session_id)
        if thread_id:
            root_span.set_attribute(THREAD_ID, thread_id)
        root_span.set_attribute(ENDUSER_ID, user_id)

        tool_sequence = []
        steps = 0
        final_answer = None
        total_cost_usd = 0.0
        requested_model = served_model = "gpt-4o-mini"

        for steps in range(1, MAX_AGENT_STEPS + 1):
            if incidents.should_misroute_to_repayment_tool() and steps == 1:
                result = calculate_monthly_repayment(principal=200000, annual_rate_pct=19.9, years=1)
                tool_sequence.append("calculate_monthly_repayment")
                final_answer = f"Based on a standard calculation, this comes to approximately GBP {result} per month."
                break

            turn = chat_raw(messages=messages, tools=TOOL_SPECS)
            total_cost_usd += turn.cost_usd
            requested_model, served_model = turn.requested_model, turn.served_model
            if not turn.tool_calls:
                final_answer = turn.content
                break
            messages.append({"role": "assistant", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in turn.tool_calls]})
            for tc in turn.tool_calls:
                tool_sequence.append(tc.name)
                args = json.loads(tc.arguments or "{}")
                fn = TOOL_FUNCTIONS[tc.name]
                result = fn(**args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        duration_ms = (time.perf_counter() - t_start) * 1000
        root_span.set_attribute("lab.steps_taken", steps)
        root_span.set_attribute("lab.tool_sequence", ",".join(tool_sequence) or "(none)")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        guardrail_triggered, guardrail_reason = _detect_guardrail(final_answer)
        root_span.set_attribute(LAB_GUARDRAIL_TRIGGERED, guardrail_triggered)
        root_span.set_attribute(LAB_REFUSAL_DETECTED, guardrail_triggered)
        if guardrail_reason:
            root_span.set_attribute(LAB_GUARDRAIL_REASON, guardrail_reason)

        metrics.record_agent_turn(
            model=served_model, duration_ms=duration_ms, cost_usd=total_cost_usd,
            guardrail_triggered=guardrail_triggered,
        )

        if not skip_eval:
            online_eval.maybe_grade(question, final_answer or "", trace_id=trace_id)
            alerting.observe_and_alert(
                requested_model=requested_model, served_model=served_model,
                cost_usd=total_cost_usd, answer_text=final_answer or "",
                tool_sequence=tool_sequence, trace_id=trace_id,
            )

        return {
            "answer": final_answer, "steps": steps, "tool_sequence": tool_sequence,
            "trace_id": trace_id, "cost_usd": round(total_cost_usd, 6),
            "duration_ms": round(duration_ms, 1), "requested_model": requested_model,
            "served_model": served_model, "guardrail_triggered": guardrail_triggered,
        }


@app.get("/")
def root():
    return {"service": "satark", "docs": "/docs"}


# ------------------------------------------------------------- carried over from Nidan
@app.get("/ask")
def ask(q: str = Query(...)):
    return _run_traced_agent_turn(q)


@app.post("/chat")
def chat(session_id: str = Query(...), message: str = Query(...),
          user_id: str = Query("anonymous"), thread_id: str = Query(None)):
    session = store.get_or_create(session_id, user_id=user_id)
    history = [{"role": t["role"], "content": t["content"]} for t in session.turns]
    session.append("user", message, thread_id=thread_id)
    result = _run_traced_agent_turn(message, session_id=session_id, user_id=user_id,
                                     thread_id=session.thread_id, history=history)
    session.append("assistant", result["answer"] or "")
    session.truncate()
    return {"session_id": session_id, "answer": result["answer"], "trace_id": result["trace_id"]}


@app.get("/memory/{session_id}")
def memory(session_id: str):
    return store.inspect(session_id)


@app.post("/memory/{session_id}/reset")
def memory_reset(session_id: str):
    store.reset(session_id)
    return {"session_id": session_id, "reset": True}


@app.get("/delegate")
def delegate(amount: float = Query(9500.0)):
    return delegate_to_fraud_check(amount)


@app.get("/replay")
def replay(q: str = Query(...), n: int = Query(5)):
    runs = replay_n_times(q, SYSTEM_PROMPT, n=n)
    return {
        "n_runs": n,
        "unique_answers": len({r.answer.strip() for r in runs}),
        "unique_tool_sequences": len({tuple(r.tool_sequence) for r in runs}),
        "runs": [{"answer": r.answer, "tool_sequence": r.tool_sequence, "steps": r.steps} for r in runs],
    }


@app.get("/diff")
def diff(q: str = Query(...), model_a: str = Query("gpt-4o-mini"), model_b: str = Query("gpt-4o-mini"),
         temperature_a: float = Query(None), temperature_b: float = Query(None)):
    run_a = run_once(q, SYSTEM_PROMPT, model=model_a, temperature=temperature_a)
    run_b = run_once(q, SYSTEM_PROMPT, model=model_b, temperature=temperature_b)
    return diff_runs(run_a, run_b, label_a="A", label_b="B")


@app.get("/grade")
def grade(q: str = Query(...), expected_tool: str = Query("lookup_policy"),
          must_contain_any: str = Query("19.9,4.10,4.55,90%,8000")):
    result = _run_traced_agent_turn(q, skip_eval=True)
    keywords = [k.strip() for k in must_contain_any.split(",")]
    verdict = grade_run(result["answer"] or "", result["tool_sequence"], expected_tool, keywords)
    return {
        "answer": result["answer"], "tool_sequence": result["tool_sequence"],
        "outcome_correct": verdict.outcome_correct, "trajectory_correct": verdict.trajectory_correct,
        "is_trajectory_failure": verdict.is_trajectory_failure, "notes": verdict.notes,
    }


# ------------------------------------------------------------- M4: observability pipeline
# Metrics themselves are emitted via metrics.py -> otel_setup.py's OTLP
# exporter -> the Collector's Prometheus exporter (see otel-collector-
# config.yaml) -> Grafana. There is deliberately no /metrics endpoint
# here duplicating that path - app.py stays the single place a metric is
# RECORDED (metrics.record_agent_turn, above), not a second place it is
# SERVED, so there is exactly one pipeline to reason about, not two.


# ------------------------------------------------------------- M5: evaluation and drift
@app.get("/eval/recent")
def eval_recent(limit: int = Query(20)):
    """Every sampled online-eval verdict, newest first - what you'd look
    at to sanity-check the judge itself, a step M5 calls out explicitly
    ('calibrating the grader before you trust it')."""
    return {"recent": online_eval.recent_judged(limit=limit),
            "rolling_average_score": online_eval.rolling_average_score()}


@app.get("/drift/status")
def drift_status():
    return drift.status()


@app.post("/drift/reset-baseline")
def drift_reset_baseline():
    """Facilitator-only - start a fresh baseline before a new exercise."""
    drift.reset_baseline()
    return {"reset": True}


@app.get("/alerts")
def alerts(limit: int = Query(20)):
    """What a duty engineer's screen actually needs - not a trace, not a
    raw metric, an ALERT: which rule fired, how severe, and the numbers
    behind it."""
    return {"alerts": alerting.recent_alerts(limit=limit)}


@app.post("/alerts/clear")
def alerts_clear():
    alerting.clear_alerts()
    return {"cleared": True}


@app.post("/feedback")
def submit_feedback(trace_id: str = Query(...), rating: str = Query(...),
                     note: str = Query(None), question: str = Query(None), answer: str = Query(None)):
    return feedback.record_feedback(trace_id, rating, note=note, question=question, answer=answer)


@app.get("/feedback/recent")
def feedback_recent(limit: int = Query(20)):
    return {"recent": feedback.recent_feedback(limit=limit)}


@app.get("/annotation-queue")
def annotation_queue(limit: int = Query(20)):
    """The reviewer-facing queue - negative feedback and low-scoring
    sampled turns, merged, newest first. This is what a human annotator
    actually works, not the raw trace or feedback stream."""
    return {"queue": feedback.annotation_queue(limit=limit)}


# ------------------------------------------------------------- incident pack (1-5)
@app.post("/seed-incident/{n}")
def seed_incident(n: int):
    """Facilitator-only in a real session. Visible here only so this lab's
    own mock-testing can confirm each incident actually changes behaviour -
    in a real exercise, do not show the response to anyone but the
    facilitator."""
    incidents.seed_incident(n)
    return {"seeded": True}


@app.post("/clear-incident")
def clear_incident():
    incidents.clear_incident()
    return {"cleared": True}


@app.get("/incident-status")
def incident_status():
    """What a learner is ALLOWED to see mid-exercise: only whether
    something is active, never which one."""
    return {"incident_active": incidents.incident_is_active()}


@app.get("/reveal-incident/{n}")
def reveal_incident(n: int):
    """The answer key - call only after writing your own diagnosis down."""
    return incidents.reveal_incident(n)
