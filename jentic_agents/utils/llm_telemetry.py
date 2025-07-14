"""Telemetry helper for tracking LLM usage and estimated cost.

This module exposes a simple singleton-like class `LLMTelemetry` that
aggregates token usage across *all* LLM calls in the current Python
process.  The LiteLLM wrapper as well as any other code that executes a
chat-completion should call `LLMTelemetry.add_usage(...)` with the model
name and the prompt / completion token counts returned by the provider.

Down-stream code (e.g. CLI / Discord outboxes) can fetch a human-readable
summary via `LLMTelemetry.get_cost_summary()` and include it in their
final output.
"""
from __future__ import annotations

from typing import Dict, Tuple

# === Hard-coded cost table ==================================================
# Values taken from OpenAI pricing as of 2024-05-13 (USD per 1K tokens).
# NOTE: Keep this table up-to-date as pricing changes!
_COST_TABLE: Dict[str, Tuple[float, float]] = {
    # model                      (prompt_cost, completion_cost)
    "gpt-4o": (0.01, 0.03),  # example
    "openai/gpt-4o": (0.01, 0.03),
    "gpt-4o-mini": (0.005, 0.015),  # made-up – adjust if needed
    "openai/gpt-4o-mini": (0.005, 0.015),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "openai/gpt-3.5-turbo": (0.0005, 0.0015),
    # Gemini pricing (approx May-2024, USD/1K tokens)
    "gemini-1.0-pro": (0.0005, 0.0015),
    "gemini-1.5-pro": (0.00025, 0.0005),
    "gemini-2.5-flash": (0.00025, 0.0005),
    "gemini/gemini-1.0-pro": (0.0005, 0.0015),
    "gemini/gemini-1.5-pro": (0.00025, 0.0005),
    "gemini/gemini-2.5-flash": (0.00025, 0.0005),
}


def _get_cost_per_1k_tokens(model: str) -> Tuple[float, float]:
    """Return (prompt_cost, completion_cost) USD for *model*. If the exact key is
    missing try to fall back to the base model name by stripping any provider
    prefix (e.g. ``openai/gpt-4o`` → ``gpt-4o``). Returns (0.0, 0.0) if still
    unknown.
    """
    if model in _COST_TABLE:
        return _COST_TABLE[model]

    # Attempt fallback without provider prefix
    if "/" in model:
        base = model.split("/", 1)[1]
        if base in _COST_TABLE:
            return _COST_TABLE[base]

    # Final fallback: log and assume low-end 3.5 Turbo pricing so cost is never 0.
    try:
        import logging
        logging.getLogger(__name__).warning(
            "LLMTelemetry: unknown model '%s' – falling back to $0.0005/$0.0015 per 1K tokens",
            model,
        )
    except Exception:
        pass
    return (0.0005, 0.0015)


class LLMTelemetry:
    """Global aggregator for LLM usage statistics."""

    _prompt_tokens: int = 0
    _completion_tokens: int = 0
    _cost_usd: float = 0.0

    @classmethod
    def reset(cls) -> None:
        """Reset all counters – mostly useful for unit tests."""
        cls._prompt_tokens = 0
        cls._completion_tokens = 0
        cls._cost_usd = 0.0

    # ---------------------------------------------------------------------
    # Recording helpers
    # ---------------------------------------------------------------------
    @classmethod
    def add_usage(cls, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record a single completion usage and update aggregate cost."""
        cls._prompt_tokens += prompt_tokens
        cls._completion_tokens += completion_tokens

        prompt_rate, completion_rate = _get_cost_per_1k_tokens(model)
        cls._cost_usd += (prompt_tokens / 1000) * prompt_rate
        cls._cost_usd += (completion_tokens / 1000) * completion_rate

    # ---------------------------------------------------------------------
    # Read helpers
    # ---------------------------------------------------------------------
    @classmethod
    def get_summary(cls) -> Dict[str, float | int]:
        """Return raw values for prompt_tokens, completion_tokens and cost."""
        return {
            "prompt_tokens": cls._prompt_tokens,
            "completion_tokens": cls._completion_tokens,
            "total_tokens": cls._prompt_tokens + cls._completion_tokens,
            "cost_usd": round(cls._cost_usd, 6),
        }

    @classmethod
    def get_cost_summary(cls) -> str:
        """Return a short human-readable cost line."""
        summary = cls.get_summary()
        tokens = summary["total_tokens"]
        cost = summary["cost_usd"]
        if tokens == 0:
            return "LLM cost: $0.000 (0 tokens)"
        return f"LLM cost: ${cost:.4f} ({tokens} tokens)" 