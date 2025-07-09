"""Core orchestration logic for the ReWOO Reasoner with JenticTools.

This skeleton follows the ReWOO (plan first, then bind tools) + Reflection
paradigm.
"""
from __future__ import annotations

from typing import Any, Dict
import json
from copy import deepcopy

from jentic_agents.reasoners.rewoo_reasoner.exceptions import MissingInputError, ToolExecutionError, ReasoningStepError
from jentic_agents.reasoners.rewoo_reasoner_contract import BaseReWOOReasoner
from jentic_agents.reasoners.jentic_toolbag import JenticToolBag
from jentic_agents.reasoners.models import ReasonerState, Step
from jentic_agents.reasoners.rewoo_reasoner._parser import parse_bullet_plan
import jentic_agents.reasoners.rewoo_reasoner._prompts as prompts  # noqa: WPS433 (importing internal module)

from jentic_agents.platform.jentic_client import JenticClient  # type: ignore
from jentic_agents.memory.base_memory import BaseMemory
from jentic_agents.utils.llm import BaseLLM
import re

class JenticReWOOReasoner(JenticToolBag, BaseReWOOReasoner):
    """Reasoner implementing ReWOO + Reflection on top of Jentic tools."""

    def __init__(
        self,
        *,
        jentic_client: JenticClient,
        memory: BaseMemory,
        llm: BaseLLM,
    ) -> None:
        super().__init__(jentic_client=jentic_client, memory=memory, llm=llm)


    def run(self, goal: str, max_iterations: int = 20):  # noqa: D401
        return super().run(goal, max_iterations)

    def _generate_plan(self, state: ReasonerState) -> None:
        """Generate initial plan from goal using the LLM."""
        prompt = prompts.PLAN_GENERATION_PROMPT.replace("{goal}", state.goal)
        plan_md = self._call_llm(prompt)
        self._logger.info(f"phase=PLAN_GENERATED plan={plan_md}")
        state.plan = parse_bullet_plan(plan_md)


    def _execute_step(self, step: Step, state: ReasonerState) -> Dict[str, Any]:  # noqa: D401
        """Execute a single plan step with retry bookkeeping."""
        step.status = "running"
        try:
            inputs = self._fetch_inputs(step)
        except MissingInputError as exc:
            self._reflect_on_failure(exc, step, state)
            return {}

        if step.step_type == Step.StepType.REASONING:
            result = self._execute_reasoning_step(step, inputs)
            step.status = "done"
            step.result = result
            self._store_step_output(step, state)
            return None

        tool_id = self._select_tool(step)
        params = self._generate_params(step, tool_id, inputs)
        try:
            result = self._jentic.execute(tool_id, params)
            self._logger.info("phase=EXECUTE_OK run_id=%s tool_id=%s", getattr(self, '_run_id', 'NA'), tool_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("phase=EXECUTE_FAIL run_id=%s tool_id=%s error=%s", getattr(self, '_run_id', 'NA'), tool_id, exc)
            raise ToolExecutionError(str(exc)) from exc

        step.status = "done"
        step.result = result['result'].output
        # Persist in-memory for downstream steps
        self._store_step_output(step, state)
        return {"tool_id": tool_id, "params": params, "result": result}

    def _reflect_on_failure(
        self,
        error: Exception,
        step: Step,
        state: ReasonerState,
    ) -> None:  # noqa: D401
        """Invoke reflection logic and possibly modify the plan."""
        step.status = "failed"
        step.error = str(error)

        if step.retry_count >= 2:
            state.history.append(f"Giving up on step after retries: {step.text}")
            return

        tool_schema = {}
        try:
            tool_execution_info = self._get_tool(self._select_tool(step))
            tool_schema = tool_execution_info['parameters']
        except Exception:
            pass

        error_type = error.__class__.__name__
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
        prompt = prompts.FINAL_ANSWER_SYNTHESIS_PROMPT.format(
        goal=state.goal,
        history="\n".join(state.history),
    )
        state.is_complete = True
        return self._call_llm(prompt)

    def _classify_step(self, step: Step, state: ReasonerState) -> Step.StepType:  # noqa: D401
        """Heuristic LLM classifier deciding TOOL vs REASONING."""
        mem_keys = getattr(self._memory, "keys", lambda: [])()
        keys_list = ", ".join(mem_keys)
        prompt = prompts.STEP_CLASSIFICATION_PROMPT.format(step_text=step.text, keys_list=keys_list)
        reply = self._call_llm(prompt).lower()
        print("Step Classified as :", reply)
        if "reason" in reply:
            return Step.StepType.REASONING
        return Step.StepType.TOOL

    def _execute_reasoning_step(self, step: Step, inputs: Dict[str, Any]) -> Any:  # noqa: D401
        """Execute a reasoning-only step via the LLM and return its output."""
        try:
            mem_snippet = json.dumps(inputs, ensure_ascii=False)
            prompt = prompts.REASONING_STEP_PROMPT.format(step_text=step.text, mem_snippet=mem_snippet)
            reply = self._call_llm(prompt).strip()
            _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")
            m = _JSON_FENCE_RE.search(reply)
            if m:
                reply = m.group(1).strip()
            return json.loads(reply)
        except Exception as exec:
            raise ReasoningStepError(str(exec))

    def _fetch_inputs(self, step: Step) -> Dict[str, Any]:
        """Retrieve all required inputs from memory or raise ``ge``."""
        inputs: Dict[str, Any] = {}
        for key in step.input_keys:
            try:
                inputs[key] = self._memory.retrieve(key)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                self._logger.warning("Missing required input key: %s", key)
                raise MissingInputError(key)
        return inputs

    def _store_step_output(self, step: Step, state: ReasonerState) -> None:
        """Persist a successful step's result under its `output_key`.

        Respect strict typing: only store if both key and result exist.
        The value is stored *as is*; if callers require serialisable data
        they must ensure the tool returns JSON-serialisable results.
        """
        if step.output_key and step.result is not None:
            # Unwrap OperationResult-like objects to their payload for JSON safety
            value_to_store = (
                step.result["result"].output if hasattr(step.result, "result") else step.result
            )
            try:
                self._memory.store(step.output_key, value_to_store)
                snippet = str(value_to_store).replace("\n", " ")
                state.history.append(f"stored {step.output_key}: {snippet}")
                self._logger.info(
                    "phase=MEM_STORE run_id=%s key=%s",
                    getattr(self, "_run_id", "NA"),
                    step.output_key,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Could not store result for key '%s': %s", step.output_key, exc)

    def _call_llm(self, prompt: str, **kwargs) -> str:
        """Send a single-turn user prompt to the LLM and return assistant content."""
        messages = [
            {"role": "user", "content": prompt},
        ]
        return self._llm.chat(messages, **kwargs).strip()