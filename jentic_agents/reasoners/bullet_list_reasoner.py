# bullet_plan_reasoner.py
"""BulletPlanReasoner — a *plan‑first, late‑bind* reasoning loop.

This class implements the **BulletPlan** strategy described in chat:

1. *Plan* — LLM produces a natural‑language indented Markdown bullet list
   of steps (potentially nested). No tools are named at this stage.
2. *Select* — at run‑time, for **each** step the reasoner
   • searches Jentic for suitable tools,
   • offers the top‑k candidates to the LLM,
   • receives an index of the chosen tool (or a request to refine the
     search query).
3. *Act* — loads the chosen tool spec, prompts the LLM for parameters
   (with memory enumeration), executes the tool and stores results.
4. *Observe / Evaluate / Reflect* — passes tool output back to LLM so it
   can mark the step complete, retry, or patch the plan.

The class extends *BaseReasoner* so it can be swapped into any
*BaseAgent* unchanged.

NOTE ▸ For brevity, this file depends on the following external pieces
(which already exist in the repo skeleton):

* `JenticClient` – thin wrapper around `jentic_sdk` with `.search()`,
  `.load()`, `.execute()`.
* `MemoryItem` dataclass and helper `prompt_memory_enumeration()` from the
  earlier discussion.
* A generic `call_llm(messages: list[dict], **kw)` helper that wraps the
  chosen OpenAI/Gemini client.

Where full implementations would be lengthy (e.g. robust Markdown plan
parser, reflection logic) the code inserts *TODO* comments so the
autonomous coding agent can fill them out.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# local utils
from ..utils.json_cleanser import strip_backtick_fences, cleanse
from .base_reasoner import BaseReasoner
from ..platform.jentic_client import JenticClient
from ..utils.llm import BaseLLM, LiteLLMChatLLM
from ..memory.agent_memory import AgentMemory
from ..utils.logger import get_logger
from .base_reasoner import StepType
from ..communication.hitl.base_intervention_hub import BaseInterventionHub, NoEscalation

logger = get_logger(__name__)

@dataclass
class Step:
    text: str
    indent: int = 0
    store_key: Optional[str] = None
    goal_context: Optional[str] = None
    status: str = "pending"
    result: Any = None
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    reflection_attempts: int = 0

@dataclass
class ReasonerState:
    goal: str
    plan: deque[Step] = field(default_factory=deque)
    history: List[str] = field(default_factory=list)
    goal_completed: bool = False
    failed: bool = False

BULLET_RE = re.compile(r"^(?P<indent>\s*)([-*]|\d+\.)\s+(?P<content>.+)$")

def parse_bullet_plan(markdown: str) -> deque[Step]:
    markdown_stripped = strip_backtick_fences(markdown)
    if markdown_stripped.startswith('[') and markdown_stripped.endswith(']'):
        try:
            logger.info("Parsing plan as JSON array")
            json_steps = json.loads(markdown_stripped)
            steps = []
            for step_data in json_steps:
                if isinstance(step_data, dict):
                    text = step_data.get('text', '')
                    step_type = step_data.get('step_type', '')
                    store_key = step_data.get('store_key')
                    goal_context = None
                    goal_match = re.search(r'\(\s*goal:\s*([^)]+)\s*\)', text)
                    if goal_match:
                        goal_context = goal_match.group(1).strip()
                        text = re.sub(r'\s*\(\s*goal:[^)]+\s*\)', '', text).strip()
                    step = Step(text=text, indent=0, store_key=store_key, goal_context=goal_context)
                    step.step_type = step_type
                    steps.append(step)
            logger.info(f"Parsed {len(steps)} steps from plan (JSON mode)")
            return deque(steps)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse as JSON: {e}, falling back to markdown parsing")

    steps: List[Step] = []
    for line_num, line in enumerate(markdown.splitlines(), 1):
        if not line.strip(): continue
        m = BULLET_RE.match(line)
        if not m:
            logger.debug(f"Line {line_num} doesn't match bullet pattern: {line}")
            continue
        indent_level = len(m.group("indent")) // 2
        content = m.group("content").strip()

        goal_context = None
        goal_match = re.search(r'\(\s*goal:\s*([^)]+)\s*\)', content)
        if goal_match:
            goal_context = goal_match.group(1).strip()
            content = re.sub(r'\s*\(\s*goal:[^)]+\s*\)', '', content).strip()

        store_key = None
        if "->" in content:
            content, directive = [part.strip() for part in content.split("->", 1)]
            if directive.startswith("store:"):
                store_key = directive.split(":", 1)[1].strip()

        steps.append(Step(text=content, indent=indent_level, store_key=store_key, goal_context=goal_context))

    leaf_steps: List[Step] = []
    for idx, step in enumerate(steps):
        next_indent = steps[idx + 1].indent if idx + 1 < len(steps) else step.indent
        if next_indent > step.indent:
            logger.debug(f"Skipping container step: '{step.text}'")
            continue
        leaf_steps.append(step)
    
    logger.info(f"Parsed {len(leaf_steps)} steps from plan (original {len(steps)})")
    return deque(leaf_steps)

class BulletPlanReasoner(BaseReasoner):
    @staticmethod
    def _load_prompt(prompt_name: str):
        current_dir = Path(__file__).parent.parent
        prompt_path = current_dir / "prompts" / f"{prompt_name}.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('{'): return json.loads(content)
                return content
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise

    def __init__(self, jentic: JenticClient, memory: AgentMemory, llm: Optional[BaseLLM] = None, model: str = "gpt-4o", max_iters: int = 20, search_top_k: int = 15, intervention_hub: Optional[BaseInterventionHub] = None):
        logger.info(f"Initializing BulletPlanReasoner with model={model}, max_iters={max_iters}, search_top_k={search_top_k}")
        super().__init__()
        self.jentic = jentic
        self.memory = memory
        self.llm = llm or LiteLLMChatLLM(model=model)
        self.max_iters = max_iters
        self.search_top_k = search_top_k
        self.escalation = intervention_hub or NoEscalation()
        logger.info("BulletPlanReasoner initialization complete")

    def _process_llm_response_for_escalation(self, response: str, context: str = "") -> str:
        response = response.strip()
        escalation_pattern = r'<escalate_to_human\s+reason="([^"]+)"\s+question="([^"]+)"\s*/>'
        match = re.search(escalation_pattern, response)
        if match:
            reason, question = match.group(1).strip(), match.group(2).strip()
            logger.info(f"🤖➡️👤 LLM requested escalation: {reason}")
            if self.escalation.is_available():
                try:
                    human_response = self.escalation.ask_human(question, context)
                    if human_response.strip():
                        logger.info(f"👤➡️🤖 Human provided response: {human_response}")
                        return human_response
                    logger.warning("👤 No response from human, continuing with original")
                except Exception as e:
                    logger.warning(f"Escalation failed: {e}")
            else:
                logger.warning("⚠️ Escalation requested but not available")
            return re.sub(escalation_pattern, '', response).strip()
        return response

    def _request_human_help(self, question: str, context: str = "") -> str:
        logger.info(f"🤖➡️👤 Direct escalation request: {question}")
        if self.escalation.is_available():
            try:
                response = self.escalation.ask_human(question, context)
                logger.info("👤➡️🤖 Human response received")
                return response
            except Exception as e:
                logger.warning(f"Direct escalation failed: {e}")
        else:
            logger.warning("⚠️ Direct escalation requested but not available")
        return ""

    def _init_state(self, goal: str, context: Dict[str, Any]) -> ReasonerState:
        return ReasonerState(goal=goal)

    def plan(self, state: ReasonerState):
        logger.info("=== PLAN PHASE ===")
        if not state.plan:
            logger.info("No existing plan, generating new plan")
            bullet_plan_template = self._load_prompt("bullet_plan")
            
            if isinstance(bullet_plan_template, dict):
                bullet_plan_template["inputs"]["goal"] = state.goal
                prompt = json.dumps(bullet_plan_template, ensure_ascii=False)
            else:
                prompt = bullet_plan_template.format(goal=state.goal)
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages=messages)
            
            context = f"Goal: {state.goal}\nPhase: Planning"
            processed_response = self._process_llm_response_for_escalation(response, context)
            
            if processed_response != response:
                logger.info("Planning was escalated, updating goal with human guidance")
                state.goal = processed_response
                return self.plan(state)
            
            plan_md = self._extract_fenced_code(processed_response)
            state.plan = parse_bullet_plan(plan_md)
            state.history.append(f"Plan generated ({len(state.plan)} steps)")
            logger.info(f"Generated plan with {len(state.plan)} steps.")

    def select_tool(self, plan_step: Step, state: ReasonerState) -> str:
        logger.info(f"=== SELECT TOOL for: {plan_step.text} ===")
        
        exec_match = re.match(r"execute\s+([\w\-_]+)", plan_step.text.strip(), re.IGNORECASE)
        if exec_match:
            mem_key = exec_match.group(1)
            if mem_key in self.memory.keys():
                stored = self.memory.retrieve(mem_key)
                if isinstance(stored, dict) and "id" in stored:
                    logger.info(f"Reusing tool_id from memory key '{mem_key}': {stored['id']}")
                    plan_step.tool_id = stored["id"]
                    return stored["id"]

        search_query = self._build_search_query(plan_step)
        hits = self.jentic.search(search_query, top_k=self.search_top_k)
        
        if not hits:
            logger.error(f"No tools found for query: '{search_query}'")
            if self.escalation.is_available():
                question = f"No tools were found for the step: \"{plan_step.text}\" with query \"{search_query}\". How should I proceed?"
                context = f"Step: {plan_step.text}\nSearch query: {search_query}\nGoal: {state.goal}"
                human_response = self._request_human_help(question, context)
                if "search:" in human_response.lower():
                    new_query = human_response.split(":", 1)[1].strip()
                    hits = self.jentic.search(new_query, top_k=self.search_top_k)
                    if not hits: raise RuntimeError("Human-guided search also found no tools")
                elif "skip" in human_response.lower():
                    raise RuntimeError("Human advised to skip step")
                else:
                    plan_step.text = human_response
                    return self.select_tool(plan_step, state)
            else:
                raise RuntimeError(f"No tools found for query '{search_query}' and no escalation available.")

        tool_id = self._select_tool_with_llm(plan_step, hits, state)
        plan_step.tool_id = tool_id
        return tool_id

    def _build_search_query(self, step: "Step") -> str:
        """Use the dedicated keyword-extraction prompt to build a search query.

        • Supports both JSON and legacy string prompt templates.
        • Falls back to the raw step text if the LLM call fails for any reason.
        """

        try:
            kw_template = self._load_prompt("keyword_extraction")
            if isinstance(kw_template, dict):
                kw_template["inputs"]["context_text"] = step.text
                prompt = json.dumps(kw_template, ensure_ascii=False)
            else:
                prompt = kw_template.format(context_text=step.text)

            reply = self.llm.chat([{"role": "user", "content": prompt}]).strip()
            if reply:
                logger.info("LLM keyword-extraction produced query: %s", reply)
                return reply.strip('"')
            logger.warning("LLM keyword-extraction returned empty string; using step text directly")
        except Exception as e:
            logger.error(f"Keyword-extraction prompt raised error: {e}; using step text directly")

        return step.text

    def _select_tool_with_llm(self, step: Step, hits: List[Dict[str, Any]], state: ReasonerState) -> str:
        numbered_lines = []
        for idx, h in enumerate(hits, 1):
            name = h.get("name") or h.get("id", "Unknown")
            api_name = h.get("api_name", "")
            desc = h.get("description", "")
            display = f"{name} ({api_name})" if api_name else name
            numbered_lines.append(f"{idx}. {display} — {desc}")
        candidate_block = "\n".join(numbered_lines)

        prompt_tpl = self._load_prompt("select_tool")
        prompt = self._render_prompt(
            prompt_tpl,
            goal=state.goal,
            plan_step=step.text,
            memory_keys=", ".join(self.memory.keys()),
            tool_candidates=candidate_block,
            context_analysis="",
            workflow_state="",
        )
        
        raw_reply = self.llm.chat(messages=[{"role": "user", "content": prompt}]).strip()
        
        match = re.search(r"(\d+)", raw_reply)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(hits): return hits[idx]["id"]

        return hits[0]["id"]

    def act(self, tool_id: str, state: ReasonerState, current_step: Step) -> Any:
        logger.info(f"=== ACTION for step: {current_step.text} ===")
        tool_info = self.jentic.load(tool_id)
        params = self._generate_params_with_ai(tool_id, tool_info, state, current_step)
        current_step.params = params
        logger.info(f"Executing tool {tool_id} with args: {params}")
        try:
            return self.jentic.execute(tool_id, params)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise

    def _generate_params_with_ai(self, tool_id: str, tool_info: Dict, state: ReasonerState, step: Step) -> Dict[str, Any]:
        param_prompt_template = self._load_prompt("param_generation")
        tool_schema = tool_info.get("parameters", {})
        memory_enum = self.memory.enumerate_for_prompt()

        prompt = self._render_prompt(
            param_prompt_template,
            goal=state.goal,
            memory=memory_enum,
            selected_operation=tool_id,
            schema=json.dumps(tool_schema, indent=2),
            step=step.text,
        )

        response = self.llm.chat(messages=[{"role": "user", "content": prompt}]).strip()
        
        context = f"Step: {step.text}\nTool: {tool_id}\nPhase: Parameter Generation"
        processed_response = self._process_llm_response_for_escalation(response, context)
        
        if processed_response != response:
            guidance_prompt = f"Based on human guidance: \"{processed_response}\"\n\nPlease generate JSON parameters for tool {tool_id} to accomplish: {step.text}"
            response = self.llm.chat(messages=[{"role": "user", "content": guidance_prompt}]).strip()

        if response.startswith("NEED_HUMAN_INPUT:"):
            missing_params_str = response.replace("NEED_HUMAN_INPUT:", "").strip()
            question = f"I need values for: {missing_params_str}"
            human_reply = self._request_human_help(question, context)
            # Re-run with human input
            step.text += f" (with human input: {human_reply})"
            return self._generate_params_with_ai(tool_id, tool_info, state, step)

        try:
            params = self._safe_json_loads(response)
            return self._resolve_placeholders(params)
        except ValueError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}. Raw: {response}")
            return {"error": str(e)}

    def observe(self, observation: Any, state: ReasonerState):
        logger.info("=== OBSERVATION ===")
        if not state.plan: return
        current_step = state.plan[0]
        current_step.result = observation
        
        success = True
        if isinstance(observation, dict) and "result" in observation:
            result_obj = observation.get("result")
            if hasattr(result_obj, "success") and not result_obj.success:
                success = False
                current_step.result = {"error": getattr(result_obj, "error", "Tool failed")}
        elif isinstance(observation, dict) and "error" in observation:
            success = False

        if success:
            current_step.status = "done"
            if current_step.store_key:
                self.memory.set(key=current_step.store_key, value=observation, description=f"Result from step '{current_step.text}'")
            state.history.append(f"{current_step.text} -> done")
            state.plan.popleft()
        else:
            current_step.status = "failed"
            state.failed = True
            state.history.append(f"{current_step.text} -> failed")

    def evaluate(self, state: ReasonerState) -> bool:
        is_complete = not state.plan and not state.failed
        logger.info(f"=== EVALUATION: Plan complete: {is_complete} ===")
        return is_complete

    def reflect(self, current_step: Step, err_msg: str, state: ReasonerState) -> bool:
        logger.info(f"=== REFLECTION on failed step: {current_step.text} | Error: {err_msg} ===")
        current_step.reflection_attempts += 1

        reflection_prompt = f"A step failed: '{current_step.text}' with error: '{err_msg}'. How to fix it? Options: TRY: <new step>, SKIP, or <escalate_to_human.../>"
        response = self.llm.chat(messages=[{"role": "user", "content": reflection_prompt}]).strip()
        
        context = f"Failed Step: {current_step.text}\nError: {err_msg}"
        processed_response = self._process_llm_response_for_escalation(response, context)
        
        if processed_response != response: # Escalation happened
            current_step.text = processed_response
            current_step.status = "pending"
            return True
        elif response.startswith("TRY:"):
            current_step.text = response.replace("TRY:", "").strip()
            current_step.status = "pending"
            return True
        elif response.startswith("SKIP"):
            state.plan.popleft()
            return True # Continue with next step
        return False

    def _classify_step(self, step: Step) -> StepType:
        if hasattr(step, "step_type") and step.step_type:
            try:
                return StepType(step.step_type.lower())
            except ValueError:
                logger.debug("Unknown step_type '%s'; falling back to heuristics", step.step_type)
        text_lower = step.text.lower()
        if any(v in text_lower for v in ["analyze", "extract", "identify", "summarize"]):
            return StepType.REASONING
        return StepType.TOOL_USING

    def _execute_reasoning_step(self, step: Step, state: ReasonerState) -> Any:
        logger.info(f"Executing reasoning step: '{step.text}'")
        mem_payload = {k: self.memory.retrieve(k) for k in self.memory.keys() if k in step.text}
        reasoning_prompt = self._render_prompt(
            self._load_prompt("reasoning_prompt"),
            step=step.text,
            mem=json.dumps(mem_payload, indent=2)
        )
        reply = self.llm.chat(messages=[{"role": "user", "content": reasoning_prompt}]).strip()
        return self._safe_json_loads(reply) if reply.startswith('{') else reply

    def run(self, goal: str, max_iterations: int = 20):
        logger.info(f"=== STARTING REASONING LOOP for goal: {goal} ===")
        from .base_reasoner import ReasoningResult  # local import to avoid circular

        state = self._init_state(goal, {})
        tool_calls: List[Dict[str, Any]] = []

        for iteration in range(max_iterations):
            logger.info(f"--- Iteration {iteration + 1}/{max_iterations} ---")
            if state.failed:
                logger.error("A step has failed and could not be recovered. Terminating.")
                break
            if not state.plan:
                self.plan(state)
                if not state.plan:
                    logger.info("Planning resulted in an empty plan. Goal considered complete.")
                    break
            
            current_step = state.plan[0]
            logger.info(f"Current step: {current_step.text}")
            
            try:
                step_type = self._classify_step(current_step)
                result = None
                if step_type == StepType.TOOL_USING:
                    tool_id = self.select_tool(current_step, state)
                    result = self.act(tool_id, state, current_step)
                    tool_calls.append({"tool_id": tool_id, "step": current_step.text, "result": result})
                elif step_type == StepType.REASONING:
                    result = self._execute_reasoning_step(current_step, state)
                
                self.observe(result, state)
            except Exception as e:
                logger.error(f"Step '{current_step.text}' failed with error: {e}")
                if not self.reflect(current_step, str(e), state):
                    logger.error("Reflection failed. Aborting.")
                    break
        
        logger.info("=== REASONING LOOP ENDED ===")

        final_answer = "Goal completed." if self.evaluate(state) else "Unable to complete goal."
        success = self.evaluate(state)
        result = ReasoningResult(
            final_answer=final_answer,
            iterations=len(tool_calls),
            tool_calls=tool_calls,
            success=success,
            error_message=None if success else "Plan incomplete or failed",
        )
        return result

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        text = strip_backtick_fences(text.strip())
        try:
            return json.loads(text or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\n{text}")

    def _resolve_placeholders(self, obj: Any) -> Any:
        return self.memory.resolve_placeholders(obj)

    @staticmethod
    def _extract_fenced_code(text: str) -> str:
        m = re.search(r"```(?:json|markdown)?\n([\s\S]+?)\n```", text)
        if m: return m.group(1).strip()
        logger.warning("No fenced code block found in LLM response, returning raw text.")
        return text

    def _render_prompt(self, prompt_template: Union[str, Dict[str, Any]], **kwargs) -> str:
        """Render a prompt template that may be either a plain string with Python-format
        placeholders *or* a JSON template following the same convention we use for
        other prompts (object with an "inputs" section containing placeholders).

        The function replaces any keys in *kwargs* either via ``str.format`` (for
        string templates) or by assigning directly into the ``inputs`` mapping
        (for JSON templates). The rendered prompt is always returned as a string
        ready to be sent to the language model.
        """
        if isinstance(prompt_template, dict):
            # JSON prompt – mutate a copy so we do not pollute global template cache
            tmpl_copy = json.loads(json.dumps(prompt_template))  # deep copy
            inputs = tmpl_copy.get("inputs", {})
            for k, v in kwargs.items():
                if k in inputs:
                    inputs[k] = v
            return json.dumps(tmpl_copy, ensure_ascii=False)
        # Plain string template
        return prompt_template.format(**kwargs)
