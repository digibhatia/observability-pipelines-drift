"""OpenAI-only chat wrapper, instrumented with a model-call span per request.
Carried over from Nidan, with one addition: every call now runs through
incidents.effective_chat_model(), which is the hook incident 5 (silent
model swap) uses. gen_ai.request.model is set from what the CALLER asked
for; gen_ai.response.model comes from the real API response. On incident 5
those two attributes disagree - that disagreement, not anything else, is
the ground-truth signal the swap happened, which is why incidents.py's
remediation note says to alert on it directly rather than only on cost or
eval drift moving."""
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from otel_setup import get_tracer
from genai_conventions import (
    GEN_AI_SYSTEM, GEN_AI_OPERATION_NAME, GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE, GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS, GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS, LAB_COST_USD, LAB_MODEL_VERSION_TAG,
    LAB_MODEL_VERSION_ACTUAL, safe_prompt_attribute,
)
from pricing import estimate_cost_usd

# Load .env from the directory containing this file.
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE)

DEFAULT_MODEL = "gpt-4o-mini"
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OPENAI_API_KEY is not set. Check your .env file at: {ENV_FILE}"
            )
        _client = OpenAI(api_key=api_key)
    return _client


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class AgentTurn:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    requested_model: str = ""
    served_model: str = ""
    raw: object = None


def chat_raw(messages, tools=None, model: str = DEFAULT_MODEL,
             temperature: Optional[float] = None, capture_prompt_attrs: bool = True) -> AgentTurn:
    import incidents  # local import avoids a circular import at module load time

    tracer = get_tracer("satark.model")
    client = _get_client()
    requested_model = model
    actual_model = incidents.effective_chat_model(requested_model)

    with tracer.start_as_current_span("model_call") as span:
        span.set_attribute(GEN_AI_SYSTEM, "openai")
        span.set_attribute(GEN_AI_OPERATION_NAME, "chat")
        # Deliberately set from what the CALLER asked for, not what actually
        # runs - this is what makes incident 5 invisible on this attribute
        # alone, and why gen_ai.response.model below is the one that matters.
        span.set_attribute(GEN_AI_REQUEST_MODEL, requested_model)
        span.set_attribute(LAB_MODEL_VERSION_TAG, requested_model)
        if temperature is not None:
            span.set_attribute(GEN_AI_REQUEST_TEMPERATURE, temperature)

        if capture_prompt_attrs and messages:
            last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            span.set_attribute("gen_ai.prompt.last_user_message", safe_prompt_attribute(last_user))

        kwargs = {"model": actual_model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        response = client.chat.completions.create(**kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = estimate_cost_usd(response.model, input_tokens, output_tokens)

        span.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)
        span.set_attribute(LAB_MODEL_VERSION_ACTUAL, response.model)
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        span.set_attribute(LAB_COST_USD, cost)

        tool_calls = []
        for tc in (choice.message.tool_calls or []):
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments))

        return AgentTurn(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            duration_ms=duration_ms,
            cost_usd=cost,
            requested_model=requested_model,
            served_model=response.model,
            raw=response,
        )
