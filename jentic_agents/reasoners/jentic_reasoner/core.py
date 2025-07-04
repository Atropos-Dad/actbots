"""Core orchestration logic for the JenticReasoner.

This skeleton follows the ReWOO (plan first, then bind tools) + Reflection
paradigm. Methods are intentionally left with minimal logic so that future
work can focus on incrementally implementing and testing each piece.
"""
from __future__ import annotations

from typing import Any, Deque, Dict, List, Optional, TypedDict
import json
from copy import deepcopy

from jentic_agents.reasoners.base_reasoner import BaseReasoner, ReasoningResult
from ._models import ReasonerState, Step, Tool
from ._parser import parse_bullet_plan
from . import _prompts as prompts  # noqa: WPS433 (importing internal module)

# Concrete interfaces from other Jentic Agents packages
from jentic_agents.platform.jentic_client import JenticClient  # type: ignore
from jentic_agents.memory.base_memory import BaseMemory
from jentic_agents.utils.llm import BaseLLM


class ParameterValidationError(ValueError):
    """Raised when generated parameters fail validation."""


class ToolExecutionError(RuntimeError):
    """Raised when executing a tool fails."""


class ReflectionDecision(TypedDict, total=False):
    action: str
    tool_id: str
    params: Dict[str, Any]
    step: str


class JenticReasoner:
    """Reasoner implementing ReWOO + Reflection on top of Jentic tools."""

    def __init__(
        self,
        jentic_client: JenticClient,
        memory: BaseMemory,
        llm: BaseLLM,
    ) -> None:
        self._jentic = jentic_client
        self._memory = memory
        # Cache loaded tool definitions to avoid redundant API calls
        self._tool_cache: Dict[str, Tool] = {}
        self._llm = llm

        # ------------------------------------------------------------------
        # Internal helper wrappers
        # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, **kwargs) -> str:
        """Send a single-turn user prompt to the LLM and return assistant content."""
        messages = [
            {"role": "user", "content": prompt},
        ]
        return self._llm.chat(messages, **kwargs).strip()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(self, goal: str, max_iterations: int = 10) -> ReasoningResult:  # noqa: D401
        """Execute the reasoning loop until completion or exhaustion."""
        state = ReasonerState(goal=goal)
        self._generate_plan(state)

        iterations = 0
        tool_calls: List[Dict[str, Any]] = []

        while state.plan and not state.is_complete and iterations < max_iterations:
            step = state.plan.popleft()
            iterations += 1
            try:
                result = self._execute_step(step, state)
                tool_calls.append(result)
            except Exception as exc:  # noqa: BLE001
                # Pass generic error type to maintain signature contract
                self._reflect_on_failure(step, state, exc, error_type="UnexpectedError")

        final_answer = self._synthesize_final_answer(state)
        success = state.is_complete and not state.plan
        return ReasoningResult(
            final_answer=final_answer,
            iterations=iterations,
            tool_calls=tool_calls,
            success=success,
            error_message=None if success else "Reasoner did not finish successfully.",
        )

    # ------------------------------------------------------------------
    # Private helpers – to be fully implemented in later steps
    # ------------------------------------------------------------------
    def _generate_plan(self, state: ReasonerState) -> None:
        """Generate initial plan from goal using the LLM."""
        prompt = prompts.PLAN_GENERATION_PROMPT.replace("{goal}", state.goal)
        plan_md = self._call_llm(prompt)
        state.plan = parse_bullet_plan(plan_md)

    def _execute_step(self, step: Step, state: ReasonerState) -> Dict[str, Any]:  # noqa: D401
        """Execute a single plan step with retry bookkeeping."""
        step.status = "running"
        try:
            tool_id = self._select_tool(step)
            if tool_id == "none":  # pure reasoning step
                step.status = "done"
                return {}

            params = self._generate_params(step, tool_id)
            try:
                result = self._jentic.execute(tool_id, params)
            except Exception as exc:  # noqa: BLE001
                raise ToolExecutionError(str(exc)) from exc

            step.status = "done"
            step.result = result
            return {"tool_id": tool_id, "params": params, "result": result}

        except ParameterValidationError as exc:
            self._reflect_on_failure(step, state, exc, error_type="ParameterValidationError")
        except ToolExecutionError as exc:
            self._reflect_on_failure(step, state, exc, error_type="ToolExecutionError")
        except Exception as exc:  # noqa: BLE001
            self._reflect_on_failure(step, state, exc, error_type="UnexpectedError")
        # In case reflection decided to give up return empty
        return {}

    def _select_tool(self, step: Step) -> str:  # noqa: D401
        """Search Jentic for relevant tools and ask the LLM to pick one."""
        tools = self._search_tools(step)
        tools_json = json.dumps([
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "api_name": t.api_name,
            }
            for t in tools
        ], ensure_ascii=False)

        prompt = prompts.TOOL_SELECTION_PROMPT.format(step=step.text, tools_json=tools_json)
        reply = self._call_llm(prompt).strip()

        if self._is_valid_tool_reply(reply, tools):
            return reply

        # Retry once with explicit instruction
        retry_prompt = (
            f"Previous response was invalid. Respond ONLY with a tool id from the list or 'none'.\n"
            f"List: {[t.id for t in tools]}"
        )
        reply = self._call_llm(retry_prompt).strip()
        if self._is_valid_tool_reply(reply, tools):
            return reply

        # Give up – mark step failed by raising
        raise ValueError(f"Could not obtain valid tool id for step '{step.text}'. Last reply: {reply}")

    # ------------------------------------------------------------------
    # Tool discovery helpers
    # ------------------------------------------------------------------
    def _search_tools(self, step: Step, top_k: int = 5) -> List[Tool]:
        """Search Jentic for tools relevant to the step text."""
        hits = self._jentic.search(step.text, top_k=top_k)
        tools: List[Tool] = []
        for hit in hits:
            tool_id = hit["id"]
            tools.append(self._get_tool(tool_id, hit))
        return tools

    def _get_tool(self, tool_id: str, summary: Optional[Dict[str, Any]] = None) -> Tool:
        """Return Tool metadata, loading full definition if necessary."""
        if tool_id in self._tool_cache:
            return self._tool_cache[tool_id]

        if summary is None:
            summary = {"id": tool_id, "name": "unknown", "description": ""}

        # Load detailed definition (parameters etc.)
        try:
            definition = self._jentic.load(tool_id)
            parameters = definition.get("parameters", {})
        except Exception:
            parameters = {}

        tool = Tool(
            id=tool_id,
            name=summary.get("name", definition.get("name", tool_id)),
            description=summary.get("description", ""),
            api_name=summary.get("api_name", definition.get("api_name", "unknown")),
            parameters=parameters,
        )
        self._tool_cache[tool_id] = tool
        return tool

    def _is_valid_tool_reply(self, reply: str, tools: List[Tool]) -> bool:
        """Check if the LLM reply is a valid tool id or 'none'."""
        if reply == "none":
            return True
        return any(t.id == reply for t in tools)

    # ------------------------------------------------------------------
    # Parameter generation
    # ------------------------------------------------------------------
    def _generate_params(self, step: Step, tool_id: str) -> Dict[str, Any]:  # noqa: D401
        """Generate and validate parameters for the selected tool via the LLM."""
        tool = self._get_tool(tool_id)  # ensure we have full schema
        prompt = prompts.PARAMETER_GENERATION_PROMPT.format(
            goal=step.text,  # using step text as sub-goal for param generation
            step=step.text,
            tool_schema=json.dumps(tool.parameters, ensure_ascii=False),
        )

        raw = self._call_llm(prompt).strip()
        params = self._parse_json_or_retry(raw, prompt)
        # Basic structural validation: ensure keys exist in schema if schema provided
        if tool.parameters:
            unknown_keys = [k for k in params.keys() if k not in tool.parameters]
            if unknown_keys:
                raise ParameterValidationError(f"Unknown parameter keys for tool {tool_id}: {unknown_keys}")
        return params

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------
    def _parse_json_or_retry(self, raw: str, original_prompt: str) -> Dict[str, Any]:
        """Attempt to parse JSON; retry once if it fails."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            retry_prompt = (
                f"Your previous output was not valid JSON. Respond ONLY with a JSON object.\n"
                f"Original prompt was:\n{original_prompt}"
            )
            raw_retry = self._call_llm(retry_prompt).strip()
            try:
                return json.loads(raw_retry)
            except json.JSONDecodeError as exc:
                raise ParameterValidationError("Failed to obtain valid JSON parameters") from exc

    def _reflect_on_failure(
        self,
        step: Step,
        state: ReasonerState,
        error: Exception,
        *,
        error_type: str,
    ) -> None:  # noqa: D401
        """Invoke reflection logic and possibly modify the plan."""
        step.status = "failed"
        step.error = str(error)

        if step.retry_count >= 2:
            state.history.append(f"Giving up on step after retries: {step.text}")
            return

        tool_schema = {}
        try:
            current_tool = self._get_tool(self._select_tool(step))
            tool_schema = current_tool.parameters
        except Exception:
            pass

        prompt = prompts.REFLECTION_PROMPT.format(
            goal=state.goal,
            step=step.text,
            error_type=error_type,
            error_message=str(error),
            tool_schema=json.dumps(tool_schema, ensure_ascii=False),
        )
        raw = self._call_llm(prompt).strip()
        decision = self._parse_json_or_retry(raw, prompt)

        action = decision.get("action")
        state.history.append(f"Reflection decision: {decision}")
        if action == "give_up":
            return

        new_step = deepcopy(step)
        new_step.retry_count += 1
        new_step.status = "pending"
        # Handle actions
        if action == "rephrase_step" and "step" in decision:
            new_step.text = str(decision["step"])
        elif action == "change_tool" and "tool_id" in decision:
            # store chosen tool_id in memory for later stages if needed
            self._memory.store(f"forced_tool:{new_step.text}", decision["tool_id"])
        elif action == "retry_params" and "params" in decision:
            # stash params so _generate_params can skip LLM call next time
            self._memory.store(f"forced_params:{new_step.text}", decision["params"])
        # push the modified step to the front of the deque for immediate retry
        state.plan.appendleft(new_step)

    def _synthesize_final_answer(self, state: ReasonerState) -> str:  # noqa: D401
        """Combine successful step results into a final answer."""
        prompt = prompts.FINAL_ANSWER_SYNTHESIS_PROMPT.format(history="\n".join(state.history))
        return self._call_llm(prompt)
