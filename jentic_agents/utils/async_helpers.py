from __future__ import annotations

"""Shared async/thread helpers for the Jentic Agents code-base.

This module centralises common utilities that previously lived in many
components, such as the "safe LLM call" wrapper.  By re-using a global
ThreadPoolExecutor we eliminate the overhead of constructing a new
executor for every LLM request and reduce thread churn.
"""

import asyncio
import concurrent.futures
from functools import partial
from typing import Any, List, Dict

# ---------------------------------------------------------------------------
# Global thread-pool ---------------------------------------------------------
# ---------------------------------------------------------------------------

# A single process-wide pool is cheaper than per-call pools and avoids the
# cost of spawning threads repeatedly.  8 workers is a reasonable default
# for I/O-bound work such as network calls to an LLM backend.
_THREAD_POOL: concurrent.futures.ThreadPoolExecutor | None = None


def _get_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _THREAD_POOL
    if _THREAD_POOL is None:
        _THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    return _THREAD_POOL


# ---------------------------------------------------------------------------
# Safe LLM call --------------------------------------------------------------
# ---------------------------------------------------------------------------

def safe_llm_call(
    llm: Any,
    messages: List[Dict[str, str]],
    *,
    timeout: int = 60,
    **kwargs,
) -> str:
    """Invoke *llm.chat* safely from both sync and async contexts.

    • If already inside an *asyncio* event-loop we delegate the synchronous
      `llm.chat` call to a shared thread-pool and await the result.
    • Outside an event-loop we call the function directly.
    • A *timeout* (seconds) prevents hanging calls in the thread-pool path.
    """

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # Async context – delegate the sync llm.chat to the shared pool and
            # block until the result is available.  We intentionally *block*
            # the coroutine that called us; the surrounding reasoner logic
            # already treats LLM calls as blocking operations.
            future = _get_pool().submit(llm.chat, messages, **kwargs)
            return future.result(timeout=timeout)
    except RuntimeError:
        # We are in a synchronous context (no running event-loop)
        pass

    # Synchronous path – simple direct call
    return llm.chat(messages, **kwargs)


__all__ = [
    "safe_llm_call",
]