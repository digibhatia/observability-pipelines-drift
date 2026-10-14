"""
GenAI semantic-convention attribute names (OpenTelemetry's gen_ai.* namespace)
and the redaction wrapper for anything that touches a span.

Carried over from Nidan unchanged, PLUS a new block of attribute names for
Satark's two new axes: quality-under-production-traffic (online eval, drift,
guardrails) and the operational metrics that back the M4 dashboard. Same
reasoning as before - these ride the real convention where one exists
(gen_ai.*), and live in our own lab.* namespace only where the spec has no
slot for them yet.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""
from pii import redact

# --- request-side attributes --------------------------------------------
GEN_AI_SYSTEM = "gen_ai.system"                     # e.g. "openai"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"     # "chat", "embeddings", "execute_tool"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"

# --- response-side attributes --------------------------------------------
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# --- tool-call attributes --------------------------------------------
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# --- correlation attributes (carried over from Nidan unchanged) ----------
SESSION_ID = "session.id"
THREAD_ID = "nidan.thread.id"
ENDUSER_ID = "enduser.id"

# --- Nidan's lab.* attributes, carried over unchanged --------------------
LAB_COST_USD = "lab.cost_usd"
LAB_RETRIEVAL_MODE = "lab.retrieval.mode"
LAB_RETRIEVAL_HIT_COUNT = "lab.retrieval.hit_count"
LAB_DELEGATED_TO = "lab.delegated_to"
LAB_TRAJECTORY_TOOL = "lab.trajectory.tool_used"
LAB_TRAJECTORY_EXPECTED = "lab.trajectory.tool_expected"
LAB_TRAJECTORY_CORRECT = "lab.trajectory.correct"
LAB_OUTCOME_CORRECT = "lab.outcome.correct"
LAB_INCIDENT_ACTIVE = "lab.incident.active"

# --- NEW in Satark: online evaluation ------------------------------------
LAB_EVAL_SAMPLED = "satark.eval.sampled"              # bool - was this turn judged at all?
LAB_EVAL_SCORE = "satark.eval.score"                  # 1-5, the judge model's score
LAB_EVAL_VERDICT = "satark.eval.verdict"              # "pass" | "fail" - score thresholded
LAB_EVAL_RATIONALE = "satark.eval.rationale"          # judge model's one-line reasoning

# --- NEW in Satark: guardrail / refusal telemetry -------------------------
LAB_GUARDRAIL_TRIGGERED = "satark.guardrail.triggered"    # bool
LAB_GUARDRAIL_REASON = "satark.guardrail.reason"          # which rule fired
LAB_REFUSAL_DETECTED = "satark.refusal.detected"          # bool - model declined to answer
LAB_FALLBACK_USED = "satark.fallback.used"                # bool - a fallback path was taken

# --- NEW in Satark: drift ---------------------------------------------------
LAB_DRIFT_SCORE = "satark.drift.score"                 # distance from the reference centroid
LAB_DRIFT_FLAGGED = "satark.drift.flagged"             # bool - score crossed the threshold
LAB_DRIFT_DIMENSION = "satark.drift.dimension"         # "input" | "output" | "tool_mix"

# --- NEW in Satark: regression / model-version tracking -------------------
LAB_MODEL_VERSION_TAG = "satark.model_version_tag"     # the version label the CALLER believes
LAB_MODEL_VERSION_ACTUAL = "satark.model_version_actual"  # the model that actually served it

# --- NEW in Satark: human feedback -----------------------------------------
LAB_FEEDBACK_RATING = "satark.feedback.rating"         # "up" | "down"
LAB_FEEDBACK_HAS_NOTE = "satark.feedback.has_note"     # bool


# --- what NEVER goes on a span, even redacted -----------------------------
NEVER_ON_A_SPAN = [
    "the raw OPENAI_API_KEY or any other secret",
    "unredacted customer PII (emails, phone numbers, sort codes)",
    "the full conversation memory for a session, turn by turn",
    "anything a regulator would call 'customer data at rest'",
]


def safe_prompt_attribute(text: str, max_len: int = 300) -> str:
    """Redact PII, then truncate. This is what actually gets set on a span -
    never the raw prompt/response string."""
    redacted = redact(text).text
    if len(redacted) > max_len:
        redacted = redacted[:max_len] + "...[TRUNCATED]"
    return redacted
