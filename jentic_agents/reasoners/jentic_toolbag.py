"""Reusable mix-ins providing Jentic-specific tool-handling logic.

This module isolates all interaction with the Jentic platform so that the
abstract ReWOO contract remains backend-agnostic.  A reasoner can opt-in to
Jentic capabilities simply by inheriting from :class:`JenticToolMixin`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import re

from jentic_agents.reasoners.models import Step, Tool
from jentic_agents.utils.logger import get_logger

# ---------------------------------------------------------------------------
# JSON fenced-block regex used by parameter parsing helpers
# ---------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")


class JenticToolBag:  # pylint: disable=too-few-public-methods
    """Mixin that adds Jentic tool discovery and execution helpers.

    Consuming classes **must** define the following private attributes *before*
    calling any mixin method:

    • ``self._jentic`` – :class:`~jentic_agents.platform.jentic_client.JenticClient`
    • ``self._call_llm(prompt: str) -> str`` – wrapper around an LLM request

    The mixin will lazily create ``self._tool_cache`` dict on first use if the
    parent class hasn't set one.
    """

    # ------------------------------------------------------------------
    # Attribute helpers
    # ------------------------------------------------------------------
    @property
    def _tool_cache(self) -> Dict[str, Any]:  # noqa: D401
        """Return a per-instance cache for loaded tool metadata."""
        cache = getattr(self, "__tool_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "__tool_cache", cache)
        return cache

    @property
    def _jentic_client(self):  # noqa: D401
        """Ensure the parent instance provides ``_jentic``."""
        if not hasattr(self, "_jentic"):
            raise AttributeError("JenticToolMixin requires 'self._jentic' attribute")
        return getattr(self, "_jentic")

    def _call_llm_proxy(self, prompt: str) -> str:  # noqa: D401
        """Proxy to the parent's ``_call_llm`` helper."""
        if not hasattr(self, "_call_llm"):
            raise AttributeError("JenticToolMixin requires '_call_llm' method on consuming class")
        return getattr(self, "_call_llm")(prompt)
    """Mixin that adds Jentic tool discovery and execution helpers.

    The concrete class is expected to expose:
    * ``self._jentic`` – an instance of :class:`~jentic_agents.platform.jentic_client.JenticClient`
    * ``self._logger`` – a standard logger
    * ``self._call_llm(prompt: str) -> str`` – wrapper around an LLM call
    * ``self._tool_cache`` – ``dict[str, Any]`` used for caching tool metadata
    """

    # ------------------------------------------------------------------
    # High-level public helpers (used by _execute_step)
    # ------------------------------------------------------------------
    def _select_tool(self, step: Step) -> str:  # noqa: D401
        """Search Jentic and ask the LLM to pick the best tool for *step*."""
        tools = self._search_tools(step)
        self._logger.info(
            "phase=SELECT_SEARCH run_id=%s step_text=%s hits=%s",
            getattr(self, "_run_id", "NA"),
            step.text,
            [f"{t.id}:{t.name}" for t in tools],
        )

        tools_json = json.dumps(
            [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "api_name": t.api_name,
                }
                for t in tools
            ],
            ensure_ascii=False,
        )

        # Lazy import to avoid circular deps
        from .rewoo_reasoner import _prompts as prompts  # type: ignore  # noqa: WPS433

        prompt = prompts.TOOL_SELECTION_PROMPT.format(step=step.text, tools_json=tools_json)
        reply = self._call_llm_proxy(prompt).strip()

        if self._is_valid_tool_reply(reply, tools):
            return reply

        # Retry once forcing a constrained answer
        retry_prompt = (
            "Previous response was invalid. Respond ONLY with a tool id from the list or 'none'.\n"
            f"List: {[t.id for t in tools]}"
        )
        reply = self._call_llm_proxy(retry_prompt).strip()
        if self._is_valid_tool_reply(reply, tools):
            return reply

        raise ValueError(
            f"Could not obtain valid tool id for step '{step.text}'. Last reply: {reply}")

    # ------------------------------------------------------------------
    # Lower-level helpers
    # ------------------------------------------------------------------
    def _search_tools(self, step: Step, top_k: int = 20) -> List[Tool]:
        """Return *top_k* potential tools for *step* using Jentic search."""
        hits = self._jentic_client.search(step.text, top_k=top_k)
        tools: List[Tool] = []
        for hit in hits:
            tools.append(
                Tool(
                    id=hit["id"],
                    name=hit.get("name", "unknown"),
                    description=hit.get("description", ""),
                    api_name=hit.get("api_name", "unknown"),
                    parameters={},
                )
            )
        return tools

    def _get_tool(self, tool_id: str) -> Dict[str, Any]:
        """Load and cache full tool execution info from Jentic API."""
        if tool_id in self._tool_cache:
            return self._tool_cache[tool_id]

        try:
            tool_execution_info = self._jentic_client.load(tool_id)
            self._tool_cache[tool_id] = tool_execution_info
            return tool_execution_info
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Could not load tool execution info Tool: %s Error: %s", tool_id, exc)
            return {}

    # --------------------- parameter generation ----------------------
    def _generate_params(
        self,
        step: Step,
        tool_id: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use the LLM to propose parameters for *tool_id*."""
        tool_execution_info = self._get_tool(tool_id)

        # Inputs may contain forced params from reflection retry logic
        forced_key = f"forced_params:{step.text}"
        forced = self._memory.retrieve(forced_key) if hasattr(self, "_memory") else None  # type: ignore[attr-defined]
        if forced:
            return forced  # pragma: no cover – reflection path

        from .rewoo_reasoner import _prompts as prompts  # type: ignore  # noqa: WPS433

        allowed_keys = ",".join(tool_execution_info.get("parameters", {}).keys())
        prompt = prompts.PARAMETER_GENERATION_PROMPT.format(
            step=step.text,
            tool_schema=json.dumps(tool_execution_info.get("parameters", {}), ensure_ascii=False),
            step_inputs=json.dumps(inputs, ensure_ascii=False),
            allowed_keys=allowed_keys,
        )
        raw = self._call_llm_proxy(prompt).strip()
        params = self._parse_json_or_retry(raw, prompt)

        # TEMP: Keep only parameters that the tool schema recognises to avoid 400s.
        params = {k: v for k, v in params.items() if k in tool_execution_info.get("parameters", {})}
        return params

    def _parse_json_or_retry(self, raw: str, original_prompt: str) -> Dict[str, Any]:
        """Best-effort JSON parse with a single retry on failure."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            retry_prompt = (
                "Your previous output was not valid JSON. Respond ONLY with a JSON object.\n"
                f"Original prompt was:\n{original_prompt}"
            )
            raw_retry = self._call_llm_proxy(retry_prompt).strip()
            return json.loads(raw_retry)  # may raise again – propagate upstream

    # ----------------------- util helpers ----------------------------
    @staticmethod
    def _is_valid_tool_reply(reply: str, tools: List[Tool]) -> bool:
        """Return *True* iff *reply* is a valid tool id from *tools* or 'none'."""
        if reply == "none":
            return True
        return any(t.id == reply for t in tools)
