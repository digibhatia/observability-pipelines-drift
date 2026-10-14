"""
M5, task 1: online evaluation - sampled model-graded scoring on LIVE
traffic, not a held-out test set run by hand (that's trajectory_eval.py's
job, carried over from Nidan, and it still runs on every /grade call).

OFFLINE vs ONLINE, the distinction M5 is built around:
  offline - you already have the "right" answer (trajectory_eval.py's
            keyword check); fast, cheap, deterministic, but only ever tells
            you about the questions you thought to write down in advance.
  online  - no reference answer exists; a second LLM call (the "judge")
            scores what real traffic actually produced, sampled rather than
            graded exhaustively because every graded turn is a second
            paid model call. This is the only way to find out how the
            agent is doing on the LONG TAIL of questions nobody wrote a
            keyword list for.

This module is deliberately NOT a separate endpoint you have to remember to
call - maybe_grade() is wired into app.py's _run_traced_agent_turn() so it
runs on real /ask and /chat traffic at whatever SATARK_EVAL_SAMPLE_RATE is
set to, the same way a production online-eval pipeline samples real
traffic rather than a curated test set.
"""
import os
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional

from otel_setup import get_tracer
from genai_conventions import (
    LAB_EVAL_SAMPLED, LAB_EVAL_SCORE, LAB_EVAL_VERDICT, LAB_EVAL_RATIONALE,
    safe_prompt_attribute,
)
import metrics

JUDGE_MODEL = "gpt-4o-mini"
PASS_THRESHOLD = 3  # score >= this on a 1-5 scale counts as "pass"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict QA grader for a banking customer-service assistant's answers. "
    "Given a customer question and the assistant's answer, score the answer from 1 (wrong "
    "or unhelpful) to 5 (accurate, complete, on-topic). Respond in EXACTLY this format, "
    "nothing else:\nSCORE: <1-5>\nREASON: <one short sentence>"
)

# Recent judged turns, newest first - what alerting.py reads to decide if
# quality has drifted, and what feedback.py's annotation queue surfaces for
# a human to look at. Capped so the lab doesn't grow memory unbounded.
_RECENT_JUDGED = deque(maxlen=200)


@dataclass
class EvalResult:
    sampled: bool
    score: Optional[int] = None
    verdict: Optional[str] = None
    rationale: Optional[str] = None


def _parse_judge_reply(text: str) -> tuple[Optional[int], str]:
    if not text:
        return None, "judge returned no content"
    score_match = re.search(r"SCORE:\s*(\d)", text)
    reason_match = re.search(r"REASON:\s*(.+)", text)
    score = int(score_match.group(1)) if score_match else None
    reason = reason_match.group(1).strip() if reason_match else text.strip()[:200]
    return score, reason


def maybe_grade(question: str, answer: str, trace_id: str = None,
                 sample_rate: float = None) -> EvalResult:
    """Called from app.py after every agent turn. Samples with probability
    sample_rate (default from SATARK_EVAL_SAMPLE_RATE, so this is tunable
    without a code change - the realistic lever a team actually pulls to
    trade judge-call cost against how quickly drift becomes visible)."""
    if sample_rate is None:
        sample_rate = float(os.environ.get("SATARK_EVAL_SAMPLE_RATE", "0.3"))

    tracer = get_tracer("satark.eval")
    with tracer.start_as_current_span("online_eval") as span:
        sampled = random.random() < sample_rate
        span.set_attribute(LAB_EVAL_SAMPLED, sampled)
        if not sampled:
            return EvalResult(sampled=False)

        from llm_provider import chat_raw  # local import avoids a circular import
        judge_messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION: {question}\nANSWER: {answer}"},
        ]
        turn = chat_raw(messages=judge_messages, model=JUDGE_MODEL, capture_prompt_attrs=False)
        score, rationale = _parse_judge_reply(turn.content)
        verdict = "pass" if (score is not None and score >= PASS_THRESHOLD) else "fail"

        span.set_attribute(LAB_EVAL_SCORE, score if score is not None else -1)
        span.set_attribute(LAB_EVAL_VERDICT, verdict)
        span.set_attribute(LAB_EVAL_RATIONALE, safe_prompt_attribute(rationale, max_len=200))

        if score is not None:
            metrics.record_eval_score(score, model=JUDGE_MODEL)

        _RECENT_JUDGED.appendleft({
            "trace_id": trace_id, "question": question[:200], "answer": answer[:300] if answer else "",
            "score": score, "verdict": verdict, "rationale": rationale,
        })
        return EvalResult(sampled=True, score=score, verdict=verdict, rationale=rationale)


def recent_judged(limit: int = 20) -> list:
    return list(_RECENT_JUDGED)[:limit]


def rolling_average_score(window: int = 20) -> Optional[float]:
    """What alerting.py actually watches: the mean of the last N judged
    scores. A single bad score is noise; a sustained drop across a window
    is the signal - the same reasoning drift.py applies to embeddings."""
    scored = [r["score"] for r in list(_RECENT_JUDGED)[:window] if r["score"] is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)
