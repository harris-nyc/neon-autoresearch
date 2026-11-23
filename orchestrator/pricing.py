"""List prices, USD per 1M tokens. Checked Nov 21, 2026 against provider docs; re-check before the run.

gemini-4-neon (Gemini API Standard / Vertex global): $3 in, $0.15 cached in, $15 out incl. thinking.
claude-opus-5-5: $4 in, $20 out, $0.20 cache read, cache write (5-min TTL) = 1.25x input = $5.
"""
PRICES = {
    "gemini-4-neon": {"input": 3.00, "cached": 0.15, "cache_write": 0.0, "output": 15.00},
    "claude-opus-5-5": {"input": 4.00, "cached": 0.20, "cache_write": 5.00, "output": 20.00},
    "mock": {"input": 0.0, "cached": 0.0, "cache_write": 0.0, "output": 0.0},
}


def cost(model, usage):
    """usage: tokens_in (uncached, excl. cache reads/writes), tokens_cached, tokens_cache_write,
    tokens_out_total (visible + thinking; both bill as output)."""
    p = PRICES[model]
    return (usage.get("tokens_in", 0) * p["input"]
            + usage.get("tokens_cached", 0) * p["cached"]
            + usage.get("tokens_cache_write", 0) * p["cache_write"]
            + usage.get("tokens_out_total", 0) * p["output"]) / 1e6
