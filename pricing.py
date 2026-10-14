"""Manual cost calculation - no API returns cost directly, so we maintain our
own table. Extended from Nidan with a second chat model: incident 5 (see
incidents.py) silently swaps gpt-4o-mini for gpt-4o on live traffic, and the
whole point of that incident is that it must still be prices-aware so the
cost jump actually shows up in the M4 dashboard and the M5 drift/regression
checks, not just in the model name attribute."""

PRICES_PER_MILLION_TOKENS_USD = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    prices = PRICES_PER_MILLION_TOKENS_USD.get(model)
    if prices is None:
        return 0.0
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
