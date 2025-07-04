"""Core orchestration logic for the JenticReasoner.

This skeleton follows the ReWOO (plan first, then bind tools) + Reflection
paradigm. Methods are intentionally left with minimal logic so that future
work can focus on incrementally implementing and testing each piece.
"""
from __future__ import annotations

from typing import Any, Deque, Dict, List

from jentic_agents.reasoners.base_reasoner import BaseReasoner, ReasoningResult
from ._models import ReasonerState, Step
from ._parser import parse_bullet_plan
from . import _prompts as prompts  # noqa: WPS433 (importing internal module)

# Concrete interfaces from other Jentic Agents packages
from jentic_agents.platform.jentic_client import JenticClient  # type: ignore
from jentic_agents.memory.base_memory import BaseMemory
from jentic_agents.utils.llm import BaseLLM


class JenticReasoner(BaseReasoner):
    """Reasoner implementing ReWOO + Reflection on top of Jentic tools."""

    def __init__(
        self,
        jentic_client: JenticClient,
        memory: BaseMemory,
        llm: BaseLLM,
    ) -> None:
        self._jentic = jentic_client
        self._memory = memory
        self._llm = llm

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
                self._reflect_on_failure(step, state, exc)

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
        plan_md = self._llm.complete(prompt)
        state.plan = parse_bullet_plan(plan_md)

    def _execute_step(self, step: Step, state: ReasonerState) -> Dict[str, Any]:  # noqa: D401
        """Execute a single plan step and return raw tool call result."""
        step.status = "running"
        tool_id = self._select_tool(step)
        if tool_id == "none":  # pure reasoning step, nothing to execute
            step.status = "done"
            return {}
        params = self._generate_params(step, tool_id)
        result = self._jentic.execute(tool_id, params)
        step.status = "done"
        step.result = result
        return {"tool_id": tool_id, "params": params, "result": result}

    def _select_tool(self, step: Step) -> str:  # noqa: D401
        """Ask the LLM to choose the best tool for the step."""
        prompt = prompts.TOOL_SELECTION_PROMPT.format(step=step.text)
        return self._llm.complete(prompt).strip()

    def _generate_params(self, step: Step, tool_id: str) -> Dict[str, Any]:  # noqa: D401
        """Generate JSON parameters for the tool via the LLM."""
        prompt = prompts.PARAMETER_GENERATION_PROMPT.format(
            tool_id=tool_id,
            step=step.text,
            goal=step.text,
        )
        # NOTE: In production we would parse JSON. For now, return empty dict.
        _ = self._llm.complete(prompt)
        return {}

    def _reflect_on_failure(self, step: Step, state: ReasonerState, error: Exception) -> None:  # noqa: D401
        """Invoke reflection when a step fails."""
        step.status = "failed"
        step.error = str(error)
        if step.reflection_attempts >= 2:  # arbitrary retry limit
            return
        step.reflection_attempts += 1
        prompt = prompts.REFLECTION_PROMPT.format(step=step.text, error=str(error), goal=state.goal)
        reflection = self._llm.complete(prompt)
        # For now just append reflection to history. Later we can adjust plan.
        state.history.append(reflection)

    def _synthesize_final_answer(self, state: ReasonerState) -> str:  # noqa: D401
        """Combine successful step results into a final answer."""
        prompt = prompts.FINAL_ANSWER_SYNTHESIS_PROMPT.format(history="\n".join(state.history))
        return self._llm.complete(prompt)
