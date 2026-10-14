"""
Grades a run on TWO independent axes, deliberately kept separate:

    outcome    - is the final answer text acceptable?
    trajectory - did the agent get there the RIGHT way (the expected tool,
                 not an unnecessary detour or the wrong tool entirely)?

The whole point: these two verdicts can disagree. A run can pass outcome
and fail trajectory (Drishti's 'wrong_tool' failure class is exactly this -
calculate_monthly_repayment returns a real, correct-LOOKING number for a
question that needed lookup_policy instead). Grading outcome alone makes
that defect invisible. This module makes both verdicts explicit, every time,
so nobody has to remember to think about trajectory separately.
"""
from dataclasses import dataclass

from otel_setup import get_tracer
from genai_conventions import LAB_TRAJECTORY_TOOL, LAB_TRAJECTORY_EXPECTED, \
    LAB_TRAJECTORY_CORRECT, LAB_OUTCOME_CORRECT


@dataclass
class TrajectoryVerdict:
    outcome_correct: bool
    trajectory_correct: bool
    expected_tool: str
    actual_tool_sequence: list
    notes: str

    @property
    def is_trajectory_failure(self) -> bool:
        """The failure class this module exists to catch: outcome looks fine,
        trajectory does not."""
        return self.outcome_correct and not self.trajectory_correct


def grade_outcome(answer: str, must_contain_any: list) -> bool:
    """A simple keyword-presence check standing in for a full model-graded
    evaluator (see M5) - good enough to demonstrate the OUTCOME axis without
    needing a second LLM call in this lab."""
    if not answer:
        return False
    lowered = answer.lower()
    return any(kw.lower() in lowered for kw in must_contain_any)


def grade_trajectory(actual_tool_sequence: list, expected_tool: str) -> bool:
    """Did the expected tool appear in the sequence at all? A stricter
    version could check ORDER or EXCLUSIVITY (only the expected tool, no
    detours) - left as a direction for the assignment."""
    return expected_tool in actual_tool_sequence


def grade_run(answer: str, actual_tool_sequence: list, expected_tool: str,
              must_contain_any: list) -> TrajectoryVerdict:
    tracer = get_tracer("nidan.eval")
    with tracer.start_as_current_span("trajectory_eval") as span:
        outcome_ok = grade_outcome(answer, must_contain_any)
        trajectory_ok = grade_trajectory(actual_tool_sequence, expected_tool)

        span.set_attribute(LAB_OUTCOME_CORRECT, outcome_ok)
        span.set_attribute(LAB_TRAJECTORY_CORRECT, trajectory_ok)
        span.set_attribute(LAB_TRAJECTORY_EXPECTED, expected_tool)
        span.set_attribute(LAB_TRAJECTORY_TOOL, ",".join(actual_tool_sequence) or "(none)")

        if outcome_ok and not trajectory_ok:
            notes = ("TRAJECTORY FAILURE: the answer looks acceptable, but the expected tool "
                     f"'{expected_tool}' never appeared in the actual sequence {actual_tool_sequence}. "
                     "This would pass an outcome-only grader and should not.")
        elif not outcome_ok and trajectory_ok:
            notes = "The right tool was used, but the final answer still doesn't look acceptable."
        elif outcome_ok and trajectory_ok:
            notes = "Both outcome and trajectory check out."
        else:
            notes = "Both outcome and trajectory failed - a straightforward, fully-visible defect."

        verdict = TrajectoryVerdict(
            outcome_correct=outcome_ok,
            trajectory_correct=trajectory_ok,
            expected_tool=expected_tool,
            actual_tool_sequence=actual_tool_sequence,
            notes=notes,
        )
        span.set_attribute("lab.trajectory_eval.is_trajectory_failure", verdict.is_trajectory_failure)
        return verdict
