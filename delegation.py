"""
A deliberately simple 'delegated sub-agent' - the fraud-referral checker -
that the main agent calls into for one specific kind of question. The point
is not the sub-agent's intelligence (it's a fixed-rule check against the
fraud threshold in 01_mortgage_lending_sop.pdf), it's the SPAN SHAPE this
produces: a span nested two levels deep, inside the span that delegated to
it, inside the trace's root span.

    agent_turn                     (root - the customer's request)
      model_call                   (decides to delegate)
      sub_agent_turn                (NEW - the delegated sub-agent's own turn)
        model_call                 (the sub-agent's own reasoning)
        tool_call                  (the sub-agent's own tool use)
      model_call                   (main agent uses the sub-agent's verdict)

Compare this to Drishti, where every trace was flat: agent_turn with
model_call/tool_call children and nothing deeper. A real multi-agent system -
one agent delegating a sub-task to a specialist - looks like THIS instead.
"""
from otel_setup import get_tracer
from genai_conventions import LAB_DELEGATED_TO
from llm_provider import chat_raw
from tools import TOOL_SPECS, TOOL_FUNCTIONS
import incidents
import json

SUB_AGENT_MAX_STEPS = 4

FRAUD_SUB_AGENT_SYSTEM_PROMPT = (
    "You are the Fraud Referral Sub-Agent for Lloyds Financial (training scenario). "
    "You are delegated a single transaction amount and must decide, using lookup_policy "
    "to confirm the current threshold, whether it must be referred to the Fraud Operations "
    "queue. Answer with just 'REFER' or 'CLEAR' and one short sentence why."
)


def delegate_to_fraud_check(transaction_amount: float) -> dict:
    """Called BY the main agent (see app.py's /delegate endpoint and
    the delegated-case branch inside the main agent loop). Produces its own
    nested 'sub_agent_turn' span, itself containing model_call/tool_call
    spans - exactly the kind of depth a single flat agent never produces."""
    tracer = get_tracer("nidan.delegation")
    with tracer.start_as_current_span("sub_agent_turn") as span:
        span.set_attribute(LAB_DELEGATED_TO, "fraud_referral_sub_agent")
        span.set_attribute("lab.delegation.input_amount", transaction_amount)

        messages = [
            {"role": "system", "content": FRAUD_SUB_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transaction amount: GBP {transaction_amount:.2f}. Should this be referred?"},
        ]
        verdict_text = None
        sub_steps = 0
        force_loop = incidents.delegation_should_loop_forever()
        for sub_steps in range(1, SUB_AGENT_MAX_STEPS + 1):
            turn = chat_raw(messages=messages, tools=TOOL_SPECS)
            # Incident 3: ignore a tool-call-free turn and force another
            # lookup_policy round anyway - the sub-agent never resolves.
            if not turn.tool_calls and not force_loop:
                verdict_text = turn.content
                break
            messages.append({"role": "assistant", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in (turn.tool_calls or [])]})
            if not turn.tool_calls:
                # forced loop with nothing to call - re-query the same tool anyway
                messages.append({"role": "tool", "tool_call_id": "forced",
                                  "content": "(incident 3: re-querying unnecessarily)"})
                continue
            for tc in turn.tool_calls:
                args = json.loads(tc.arguments or "{}")
                fn = TOOL_FUNCTIONS[tc.name]
                result = fn(**args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        span.set_attribute("lab.delegation.sub_steps", sub_steps)
        span.set_attribute("lab.delegation.hit_sub_step_cap", sub_steps >= SUB_AGENT_MAX_STEPS)
        verdict = "REFER" if verdict_text and "REFER" in verdict_text.upper() else \
            ("UNRESOLVED" if verdict_text is None else "CLEAR")
        span.set_attribute("lab.delegation.verdict", verdict)
        return {"verdict": verdict, "detail": verdict_text, "sub_steps": sub_steps}
