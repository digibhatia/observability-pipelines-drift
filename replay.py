"""
Replay the same input multiple times, or under two different configurations,
and DIFF the results structurally - not just 'does the text look different'
(that's Smriti's nondeterminism.py), but a field-by-field diff of what each
run's trace actually did: which tools fired, in what order, what each cost,
and only then whether the final text differs.

This is the 'isolate a change' half of M3: when something changes (a prompt
edit, a different model, a different retrieval mode), replay_and_diff tells
you exactly which field moved, not just that the output 'feels different'.
"""
import difflib
from dataclasses import dataclass, field

from llm_provider import chat_raw
from tools import TOOL_SPECS, TOOL_FUNCTIONS
from otel_setup import get_tracer
import json

MAX_STEPS = 6


@dataclass
class RunResult:
    answer: str
    tool_sequence: list = field(default_factory=list)  # ordered list of tool names called
    total_cost_usd: float = 0.0
    steps: int = 0
    finish_reason: str = ""


def run_once(question: str, system_prompt: str, model: str = "gpt-4o-mini",
             temperature: float = None) -> RunResult:
    tracer = get_tracer("nidan.replay")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    with tracer.start_as_current_span("agent_turn") as span:
        tool_sequence = []
        steps = 0
        answer = None
        finish_reason = ""
        for steps in range(1, MAX_STEPS + 1):
            turn = chat_raw(messages=messages, tools=TOOL_SPECS, model=model, temperature=temperature)
            finish_reason = turn.finish_reason
            if not turn.tool_calls:
                answer = turn.content
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
        span.set_attribute("lab.replay.tool_sequence", ",".join(tool_sequence))
        span.set_attribute("lab.replay.steps", steps)
        return RunResult(answer=answer or "", tool_sequence=tool_sequence, steps=steps, finish_reason=finish_reason)


def diff_runs(run_a: RunResult, run_b: RunResult, label_a: str = "A", label_b: str = "B") -> dict:
    """A structured diff - each field compared independently, so you know
    EXACTLY what moved, not just that 'something' did."""
    text_ratio = difflib.SequenceMatcher(None, run_a.answer, run_b.answer).ratio()
    return {
        "tool_sequence_identical": run_a.tool_sequence == run_b.tool_sequence,
        "tool_sequence": {label_a: run_a.tool_sequence, label_b: run_b.tool_sequence},
        "steps": {label_a: run_a.steps, label_b: run_b.steps},
        "finish_reason": {label_a: run_a.finish_reason, label_b: run_b.finish_reason},
        "answer_similarity": round(text_ratio, 3),
        "answer_identical": run_a.answer.strip() == run_b.answer.strip(),
        "answers": {label_a: run_a.answer, label_b: run_b.answer},
    }


def replay_n_times(question: str, system_prompt: str, n: int = 5, **kwargs) -> list:
    return [run_once(question, system_prompt, **kwargs) for _ in range(n)]
