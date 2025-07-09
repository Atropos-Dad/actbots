"""Core orchestration logic for the JenticReasoner.

This skeleton follows the ReWOO (plan first, then bind tools) + Reflection
paradigm. Methods are intentionally left with minimal logic so that future
work can focus on incrementally implementing and testing each piece.
"""
from __future__ import annotations

from typing import Any, Deque, Dict, List, Optional, TypedDict
import json
from copy import deepcopy


from ..base_reasoner_v2 import BaseReasonerV2
from ..models import ReasonerState, Step, Tool
from ._parser import parse_bullet_plan
from . import _prompts as prompts  # noqa: WPS433 (importing internal module)

# Concrete interfaces from other Jentic Agents packages
from jentic_agents.platform.jentic_client import JenticClient  # type: ignore
from jentic_agents.memory.base_memory import BaseMemory
from jentic_agents.utils.llm import BaseLLM
import re

from ._prompts import STEP_CLASSIFICATION_PROMPT, REASONING_STEP_PROMPT


# ---------------------------------------------------------------------------
# Regex helpers used by reasoning-step executor
# ---------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")


class ParameterValidationError(ValueError):
    """Raised when generated parameters fail validation."""


class MissingInputError(KeyError):
    """Raised when a required memory key is absent."""

class ToolExecutionError(RuntimeError):
    """Raised when executing a tool fails."""


class ReflectionDecision(TypedDict, total=False):
    action: str
    tool_id: str
    params: Dict[str, Any]
    step: str


class JenticReasoner(BaseReasonerV2):
    """Reasoner implementing ReWOO + Reflection on top of Jentic tools."""

    def __init__(
        self,
        *,
        jentic_client: JenticClient,
        memory: BaseMemory,
        llm: BaseLLM,
    ) -> None:
        super().__init__(jentic_client=jentic_client, memory=memory, llm=llm)
        # Cache loaded tool definitions to avoid redundant API calls
        self._tool_cache: Dict[str, Any] = {}

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
    # Delegates orchestration to BaseReasonerV2
    def run(self, goal: str, max_iterations: int = 20):  # noqa: D401
        """Thin wrapper that calls the shared BaseReasonerV2 run()."""
        return super().run(goal, max_iterations)

    # ------------------------------------------------------------------
    # Private helpers – to be fully implemented in later steps
    # ------------------------------------------------------------------
    def _generate_plan(self, state: ReasonerState) -> None:
        """Generate initial plan from goal using the LLM."""
        prompt = prompts.PLAN_GENERATION_PROMPT.replace("{goal}", state.goal)
        plan_md = self._call_llm(prompt)
        self._logger.info(f"phase=PLAN_GENERATED plan={plan_md}")
        state.plan = parse_bullet_plan(plan_md)

    # ------------------------------------------------------------------
    # Memory persistence helpers
    # ------------------------------------------------------------------
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

    # TODO: check if this is necessary
    def _merge_inputs(self, base: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Overlay fetched inputs onto LLM-generated params (inputs win)."""
        merged = base.copy()
        merged.update(inputs)
        return merged

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

    # ------------------------------------------------------------------
    # Core execution logic
    # ------------------------------------------------------------------
    def _execute_reasoning_step(self, step: Step, inputs: Dict[str, Any]) -> Any:  # noqa: D401
        """Execute a reasoning-only step via the LLM and return its output."""
        try:
            # Make inputs printable
            mem_snippet = json.dumps(inputs, ensure_ascii=False)
            prompt = REASONING_STEP_PROMPT.format(step_text=step.text, mem_snippet=mem_snippet)
            reply = self._call_llm(prompt).strip()
            # Try to extract fenced JSON
            m = _JSON_FENCE_RE.search(reply)
            if m:
                reply = m.group(1).strip()
            # Attempt JSON parse; fall back to raw text

            return json.loads(reply)
        except Exception as e:
            return reply

    # ------------------------------------------------------------------
    # Existing execute step, adapted for StepType
    # ------------------------------------------------------------------
    def _execute_step(self, step: Step, state: ReasonerState) -> Dict[str, Any]:  # noqa: D401
        """Execute a single plan step with retry bookkeeping."""
        step.status = "running"
        try:
            # resolve inputs first
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

        # TOOL path
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

    def _select_tool(self, step: Step) -> str:  # noqa: D401
        """Search Jentic for relevant tools and ask the LLM to pick one."""
        tools = self._search_tools(step)
        self._logger.info("phase=SELECT_SEARCH run_id=%s step_text=%s hits=%s", getattr(self, '_run_id', 'NA'), step.text, [f"{t.id}:{t.name}" for t in tools])
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
    def _search_tools(self, step: Step, top_k: int = 20) -> List[Tool]:
        """Search Jentic for tools relevant to the step text."""
        hits = self._jentic.search(step.text, top_k=top_k)
        tools: List[Tool] = []
        for hit in hits:
            # Build lightweight Tool objects from search metadata only – avoids
            # expensive full-definition fetches for tools we may never use.
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

    def _get_tool(self, tool_id: str, summary: Optional[Dict[str, Any]] = None) -> dict[str, Any]:
        """Return Tool metadata, loading full definition if necessary."""
        if tool_id in self._tool_cache:
            return self._tool_cache[tool_id]

        # Load detailed definition (parameters etc.)
        try:
            tool_execution_info = self._jentic.load(tool_id)
            self._tool_cache[tool_id] = tool_execution_info
            return tool_execution_info
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Could not load tool execution info Tool: %s Error: %s", tool_id, exc)
            return {}


    def _is_valid_tool_reply(self, reply: str, tools: List[Tool]) -> bool:
        """Check if the LLM reply is a valid tool id or 'none'."""
        if reply == "none":
            return True
        return any(t.id == reply for t in tools)

        # ------------------------------------------------------------------
    # Step classification
    # ------------------------------------------------------------------
    def _classify_step(self, step: Step, state: ReasonerState) -> Step.StepType:  # noqa: D401
        """Heuristic LLM classifier deciding TOOL vs REASONING."""
        mem_keys = getattr(self._memory, "keys", lambda: [])()
        keys_list = ", ".join(mem_keys)
        prompt = STEP_CLASSIFICATION_PROMPT.format(step_text=step.text, keys_list=keys_list)
        reply = self._call_llm(prompt).lower()
        print("Step Classified as :", reply)
        if "reason" in reply:
            return Step.StepType.REASONING
        return Step.StepType.TOOL

    # ------------------------------------------------------------------
    # Parameter generation
    # ------------------------------------------------------------------
    def _generate_params(self, step: Step, tool_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
        """Generate and validate parameters for the selected tool via the LLM."""
        try:
            tool_execution_info = self._get_tool(tool_id)  # ensure we have full schema
            allowed_keys = [k for k in tool_execution_info['parameters'].keys()]
            step_inputs = json.dumps(inputs, ensure_ascii=False)
            prompt = prompts.PARAMETER_GENERATION_PROMPT.format(
                allowed_keys=",".join(allowed_keys),
                step=step.text ,
                step_inputs=step_inputs,
                tool_schema=json.dumps(tool_execution_info, ensure_ascii=False)
            )

            raw = self._call_llm(prompt).strip()
            params = self._parse_json_or_retry(raw, prompt)
            # merge concrete inputs
            params = self._merge_inputs(params, inputs)
            self._logger.info("phase=PARAMS_DONE run_id=%s tool_id=%s param_keys=%s", getattr(self, '_run_id', 'NA'), tool_id, list(params.keys()))
            # Basic structural validation: ensure keys exist in schema if schema provided
            # if tool_execution_info['parameters']:
            #     unknown_keys = [k for k in params.keys() if k not in tool_execution_info['parameters']]
            #     if unknown_keys:
            #         raise ParameterValidationError(f"Unknown parameter keys for tool {tool_id}: {unknown_keys}")

            # TEMP FIX
            params = {k: v for k, v in params.items()
                      if k in tool_execution_info["parameters"]}
            return params
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in parameter generation : %s", exc)
            raise ParameterValidationError(f"Failed to generate parameters for tool {tool_id}") from exc

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
