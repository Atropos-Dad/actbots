"""LiteLLM wrapper for the Jentic Agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import asyncio
import concurrent.futures
import time
import logging
from .config import get_config_value


class BaseLLM(ABC):
    """Minimal synchronous chat‑LLM interface.

    • Accepts a list[dict] *messages* like the OpenAI Chat format.
    • Returns *content* (str) of the assistant reply.
    • Implementations SHOULD be stateless; auth + model name given at init.
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...
    
    async def chat_async(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Async version of chat that runs sync method in thread pool."""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.chat, messages, **kwargs)


class LiteLLMChatLLM(BaseLLM):
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        enable_cost_tracking: bool = True,
    ) -> None:
        import litellm

        if model is None:
            model = get_config_value("llm", "model", default="gpt-4o")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_cost_tracking = enable_cost_tracking
        self._client = litellm
        self._logger = logging.getLogger(__name__)
        
        # Cost tracking state
        self._total_calls = 0
        self._total_cost = 0.0
        self._total_tokens = 0

        # Simple in-process LRU cache for idempotent requests
        from collections import OrderedDict
        self._response_cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_size: int = 256  # configurable quick-win

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        start_time = time.time()
        
        try:
            # Build a deterministic cache key
            import json, hashlib
            cache_key_source = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            cache_key_str = json.dumps(cache_key_source, sort_keys=True, ensure_ascii=False)
            cache_key = hashlib.sha256(cache_key_str.encode("utf-8")).hexdigest()

            if cache_key in self._response_cache:
                # Cache hit – move to end for LRU order and return
                self._response_cache.move_to_end(cache_key)
                return self._response_cache[cache_key]

            # Cache miss – make the actual LLM call
            resp = self._client.completion(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            content = resp.choices[0].message.content or ""
            
            # Track cost if enabled
            if self.enable_cost_tracking:
                self._track_completion(resp, start_time)
            
            # Store in cache (LRU)
            self._response_cache[cache_key] = content
            if len(self._response_cache) > self._cache_size:
                # popitem(last=False) pops the oldest entry (LRU)
                self._response_cache.popitem(last=False)

            return content
            
        except Exception as e:
            if self.enable_cost_tracking:
                self._track_error(e, start_time)
            raise

    def _track_completion(self, response, start_time: float):
        """Track successful completion metrics."""
        duration = time.time() - start_time
        
        # Extract usage metrics
        usage = getattr(response, 'usage', None)
        prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
        completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
        total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0
        
        # Calculate cost using LiteLLM's completion_cost function
        cost = 0.0
        try:
            cost = self._client.completion_cost(completion_response=response)
        except Exception as e:
            self._logger.debug(f"Could not calculate cost: {e}")
        
        # Update tracking counters
        self._total_calls += 1
        self._total_cost += cost
        self._total_tokens += total_tokens
        
        # Log at debug level to avoid cluttering
        self._logger.debug(
            f"LLM Call - Model: {self.model}, Duration: {duration:.3f}s, "
            f"Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}), "
            f"Cost: ${cost:.6f}, Total Cost: ${self._total_cost:.6f}"
        )

    def _track_error(self, error: Exception, start_time: float):
        """Track failed completion metrics."""
        duration = time.time() - start_time
        self._total_calls += 1
        
        self._logger.error(
            f"LLM Call Failed - Model: {self.model}, Duration: {duration:.3f}s, "
            f"Error: {str(error)}, Total Calls: {self._total_calls}"
        )

    def get_cost_stats(self) -> Dict[str, float]:
        """Get current cost tracking statistics."""
        return {
            "total_calls": self._total_calls,
            "total_cost": self._total_cost,
            "total_tokens": self._total_tokens,
            "average_cost_per_call": self._total_cost / max(self._total_calls, 1),
            "average_tokens_per_call": self._total_tokens / max(self._total_calls, 1),
        }

    def reset_cost_tracking(self):
        """Reset cost tracking counters."""
        self._total_calls = 0
        self._total_cost = 0.0
        self._total_tokens = 0
