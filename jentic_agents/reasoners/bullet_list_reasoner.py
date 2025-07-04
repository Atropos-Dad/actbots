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
from typing import Any, Dict, List, Optional

from .base_reasoner import BaseReasoner
from ..platform.jentic_client import JenticClient  # local wrapper, not the raw SDK
from ..utils.llm import BaseLLM, LiteLLMChatLLM
from ..memory.scratch_pad import ScratchPadMemory
from ..utils.logger import get_logger
from .base_reasoner import StepType

# Initialize module logger using the shared logging utility
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum number of self-healing attempts for a single plan step.
MAX_REFLECTION_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Helper data models
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One bullet‑plan step.

    Only the *raw* natural‑language text is strictly required. Parsing of
    optional directives (e.g. `store_result_as:`) can be layered on via
    regex or a YAML code fence inside the bullet body.
    """

    text: str
    indent: int = 0  # 0 = top‑level, 1 = first sub‑bullet, …
    store_key: Optional[str] = None  # where to stash the result in memory
    goal_context: Optional[str] = None  # extracted goal context from parentheses
    status: str = "pending"  # pending | running | done | failed
    result: Any = None
    tool_id: Optional[str] = None  # chosen Jentic tool
    reflection_attempts: int = 0  # track how many times we've tried to fix this step


@dataclass
class ReasonerState:
    goal: str
    plan: deque[Step] = field(default_factory=deque)
    history: List[str] = field(default_factory=list)  # raw trace lines
    goal_completed: bool = False  # Track if the main goal has been achieved


# ---------------------------------------------------------------------------
# Markdown bullet‑list parsing helpers
# ---------------------------------------------------------------------------

BULLET_RE = re.compile(r"^(?P<indent>\s*)([-*]|\d+\.)\s+(?P<content>.+)$")


def parse_bullet_plan(markdown: str) -> deque[Step]:
    """Very lenient parser that turns an indented bullet list into Step objects."""
    logger.info(f"Parsing bullet plan from markdown:\n{markdown}")
    steps: List[Step] = []
    for line_num, line in enumerate(markdown.splitlines(), 1):
        if not line.strip():
            continue  # skip blanks
        m = BULLET_RE.match(line)
        if not m:
            logger.debug(f"Line {line_num} doesn't match bullet pattern: {line}")
            continue
        indent_spaces = len(m.group("indent"))
        indent_level = indent_spaces // 2  # assume two‑space indents
        content = m.group("content").strip()

        # Parse goal context from parentheses: "... ( goal: actual goal text )"
        goal_context = None
        goal_match = re.search(r'\(\s*goal:\s*([^)]+)\s*\)', content)
        if goal_match:
            goal_context = goal_match.group(1).strip()
            # Remove the goal context from the main content
            content = re.sub(r'\s*\(\s*goal:[^)]+\s*\)', '', content).strip()
            logger.debug(f"Extracted goal context: {goal_context}")

        # Simple directive detection:  "… -> store: weather"
        store_key = None
        if "->" in content:
            content, directive = [part.strip() for part in content.split("->", 1)]
            if directive.startswith("store:"):
                store_key = directive.split(":", 1)[1].strip()
                logger.debug(f"Found store directive: {store_key}")

        step = Step(text=content, indent=indent_level, store_key=store_key, goal_context=goal_context)
        steps.append(step)
        logger.debug(f"Parsed step: text='{step.text}', goal_context='{step.goal_context}', store_key='{step.store_key}'")

    # ------------------------------------------------------------------
    # Skip container/meta bullets so we only execute leaf actions.
    # A container is detected when the next bullet has a larger indent
    # level than the current one.
    leaf_steps: List[Step] = []
    for idx, step in enumerate(steps):
        next_indent = steps[idx + 1].indent if idx + 1 < len(steps) else step.indent
        if next_indent > step.indent:
            logger.debug(f"Skipping container step: '{step.text}'")
            continue  # don't enqueue parent/meta bullets
        leaf_steps.append(step)

    logger.info(
        f"Parsed {len(leaf_steps)} leaf steps from bullet plan (original {len(steps)})"
    )
    return deque(leaf_steps)


# ---------------------------------------------------------------------------
# BulletPlanReasoner implementation
# ---------------------------------------------------------------------------


class BulletPlanReasoner(BaseReasoner):
    """Concrete Reasoner that follows the BulletPlan strategy."""

    @staticmethod
    def _load_prompt(prompt_name: str) -> str:
        """Load a prompt from the prompts directory."""
        current_dir = Path(__file__).parent.parent
        prompt_path = current_dir / "prompts" / f"{prompt_name}.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise RuntimeError(f"Prompt file not found: {prompt_path}")

    def __init__(
        self,
        jentic: JenticClient,
        memory: ScratchPadMemory,
        llm: Optional[BaseLLM] = None,
        model: str = "gpt-4o",
        max_iters: int = 20,
        search_top_k: int = 15,
    ) -> None:
        logger.info(f"Initializing BulletPlanReasoner with model={model}, max_iters={max_iters}, search_top_k={search_top_k}")
        super().__init__()
        self.jentic = jentic
        self.memory = memory
        self.llm = llm or LiteLLMChatLLM(model=model)
        self.max_iters = max_iters
        self.search_top_k = search_top_k
        logger.info("BulletPlanReasoner initialization complete")

    # ------------------------------------------------------------------
    # BaseReasoner hook implementations
    # ------------------------------------------------------------------

    def _init_state(self, goal: str, context: Dict[str, Any]) -> ReasonerState:
        logger.info(f"Initializing state for goal: {goal}")
        logger.debug(f"Context: {context}")
        state = ReasonerState(goal=goal)
        logger.debug(f"Created initial state: {state}")
        return state

    # 1. PLAN -----------------------------------------------------------
    def plan(self, state: ReasonerState):
        logger.info("=== PLAN PHASE ===")
        if not state.plan:  # first call → create plan
            logger.info("No existing plan, generating new plan")
            bullet_plan_template = self._load_prompt("bullet_plan")
            prompt = bullet_plan_template.format(goal=state.goal)
            logger.debug(f"Planning prompt:\n{prompt}")
            
            messages = [{"role": "user", "content": prompt}]
            logger.info("Calling LLM for plan generation")
            response = self.llm.chat(messages=messages)
            logger.info(f"LLM planning response:\n{response}")
            
            logger.info("Extracting fenced code from response")
            plan_markdown = self._extract_fenced_code(response)
            logger.debug(f"Extracted plan markdown:\n{plan_markdown}")
            
            logger.info("Parsing bullet plan")
            state.plan = parse_bullet_plan(plan_markdown)
            state.history.append(f"Plan generated ({len(state.plan)} steps)")
            
            logger.info(f"Generated plan with {len(state.plan)} steps:")
            for i, step in enumerate(state.plan):
                logger.info(f"  Step {i+1}: {step.text}")
                if step.store_key:
                    logger.debug(f"    Store key: {step.store_key}")
        else:
            logger.info(f"Using existing plan with {len(state.plan)} remaining steps")
        
        if state.plan:
            current_step = state.plan[0]
            logger.info(f"Current step to execute: {current_step.text}")
            return current_step
        else:
            logger.warning("No steps in plan!")
            return None

    # 2. SELECT TOOL ----------------------------------------------------
    def select_tool(self, plan_step: Step, state: ReasonerState):
        logger.info("=== TOOL SELECTION PHASE ===")
        logger.info(f"Selecting tool for step: {plan_step.text}")
        logger.debug(f"Step goal_context: {plan_step.goal_context}")
        logger.debug(f"State goal: {state.goal}")
        
        if plan_step.tool_id:
            logger.info(f"Step already has tool_id: {plan_step.tool_id}")
            return plan_step.tool_id

        # Get AI to extract better search keywords from the goal
        search_query = self._extract_search_keywords_with_ai(plan_step, state)
        
        logger.info(f"Using generated search query: {search_query}")

        # Search Jentic by enhanced NL description
        logger.info(f"Searching Jentic for tools matching: {search_query}")
        hits = self.jentic.search(search_query, top_k=self.search_top_k)
        logger.info(f"Jentic search returned {len(hits)} results")
        
        if not hits:
            logger.error(f"No tools found for search query: {search_query}")
            raise RuntimeError(f"No tool found for step: {plan_step.text}")

        logger.info("Found tool candidates:")
        tool_lines_list = []
        for i, h in enumerate(hits):
            if isinstance(h, dict):
                name = h.get('name', h.get('id', 'Unknown'))
                api_name = h.get('api_name')
                description = h.get('description', '')
                hit_id = h.get('id', 'Unknown')
            else:
                name = getattr(h, 'name', 'Unknown')
                api_name = getattr(h, 'api_name', None)
                description = getattr(h, 'description', '')
                hit_id = getattr(h, 'id', 'Unknown')

            display_name = f"{name} ({api_name})" if api_name else name
            logger.info(f"  {i+1}. {display_name} (ID: {hit_id}) - {description}")
            tool_lines_list.append(f"{i+1}. {display_name} — {description}")
        
        tool_lines = "\n".join(tool_lines_list)
        
        # Include goal context in the selection prompt for better decision making
        goal_info = ""
        if plan_step.goal_context:
            goal_info = f"\nGoal context: {plan_step.goal_context}"
        elif state.goal:
            goal_info = f"\nOverall goal: {state.goal}"
        
        select_tool_template = self._load_prompt("select_tool")
        select_tool_prompt = select_tool_template.format(
            plan_step_text=plan_step.text,
            goal_info=goal_info,
            tool_lines=tool_lines
        )
        
        logger.debug(f"Tool selection prompt:\n{select_tool_prompt}")
        
        messages = [{"role": "user", "content": select_tool_prompt}]
        logger.info("Calling LLM for tool selection")
        reply = self.llm.chat(messages=messages).strip()
        logger.info(f"LLM tool selection response: '{reply}'")

        # Detect a "no suitable tool" reply signalled by leading 0 (e.g. "0", "0.", "0 -", etc.)
        if re.match(r"^\s*0\D?", reply):
            logger.warning("LLM couldn't find a suitable tool")
            raise RuntimeError("LLM couldn't find a suitable tool.")

        try:
            # Robustly extract the *first* integer that appears in the reply, e.g.
            # "3. inspect-request-data …" → 3
            # "Option 2: foo" → 2
            # "0" → 0
            # Handle verbose responses by looking for various patterns
            
            # First try to find a boxed answer (common in verbose responses)
            boxed_match = re.search(r'\$\\boxed\{(\d+)\}\$', reply)
            if boxed_match:
                idx = int(boxed_match.group(1)) - 1
                logger.debug(f"Found boxed answer, parsed tool index: {idx}")
            else:
                # Look for "Number: X" pattern (from our prompt)
                number_pattern = re.search(r'Number:\s*(\d+)', reply, re.IGNORECASE)
                if number_pattern:
                    idx = int(number_pattern.group(1)) - 1
                    logger.debug(f"Found 'Number:' pattern, parsed tool index: {idx}")
                else:
                    # Look for "final answer is X" or similar patterns
                    final_answer_match = re.search(r'(?:final answer is|answer is|therefore[,\s]+(?:tool\s+)?|the best match is)[:\s]*(\d+)', reply, re.IGNORECASE)
                    if final_answer_match:
                        idx = int(final_answer_match.group(1)) - 1
                        logger.debug(f"Found final answer pattern, parsed tool index: {idx}")
                    else:
                        # Fallback to finding the first integer
                        m = re.search(r"\d+", reply)
                        if not m:
                            raise ValueError("No leading integer found in LLM reply")
                        idx = int(m.group(0)) - 1
                        logger.debug(f"Used fallback method, parsed tool index: {idx}")
            
            if idx < 0 or idx >= len(hits):
                logger.error(f"Tool index {idx} out of range (0-{len(hits)-1})")
                raise IndexError(f"Tool index out of range")
                
            selected_hit = hits[idx]
            logger.debug(f"Selected hit: {selected_hit}")
            
            tool_id = selected_hit.get('id') if isinstance(selected_hit, dict) else getattr(selected_hit, 'id', None)
            tool_name = selected_hit.get('name', tool_id) if isinstance(selected_hit, dict) else getattr(selected_hit, 'name', tool_id)
            
            logger.info(f"Selected tool: {tool_name} (ID: {tool_id})")
            plan_step.tool_id = tool_id
            return tool_id
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing tool selection reply '{reply}': {e}")
            raise RuntimeError(f"Invalid tool index reply: {reply}")

    def _extract_search_keywords_with_ai(self, plan_step: Step, state: ReasonerState) -> str:
        """Use an LLM to rephrase a technical plan step into a high-quality,
        capability-focused search query for the Jentic tool marketplace."""

        # Combine step text with goal context for a richer prompt
        context_text = plan_step.text
        if plan_step.goal_context:
            context_text += f" (Context: This is part of a larger goal to '{plan_step.goal_context}')"

        keyword_extraction_template = self._load_prompt("keyword_extraction")
        keyword_prompt = keyword_extraction_template.format(context_text=context_text)

        logger.info("Calling LLM for keyword extraction")
        messages = [{"role": "user", "content": keyword_prompt}]
        keywords = self.llm.chat(messages=messages).strip()

        # Clean up the response, removing potential quotes
        keywords = keywords.strip('"\'')

        logger.info(f"AI extracted keywords: '{keywords}'")
        return keywords

    # 3. ACT ------------------------------------------------------------
    def act(self, tool_id: str, state: ReasonerState):
        logger.info("=== ACTION PHASE ===")
        logger.info(f"Executing action with tool_id: {tool_id}")
        
        logger.info("Loading tool information from Jentic")
        tool_info = self.jentic.load(tool_id)
        logger.debug(f"Tool info: {tool_info}")
        
        # Use tool info directly without filtering credentials
        # Jentic platform should handle credential injection automatically
        if isinstance(tool_info, dict):
            tool_schema = tool_info
        elif hasattr(tool_info, 'schema_summary') and isinstance(tool_info.schema_summary, dict):
            tool_schema = tool_info.schema_summary
        else:
            tool_schema = str(tool_info)
        logger.debug(f"Tool schema: {tool_schema}")

        logger.info("Enumerating memory for prompt")
        memory_enum = self.memory.enumerate_for_prompt()
        logger.debug(f"Memory enumeration: {memory_enum}")

        def _escape_braces(text: str) -> str:
            """Escape curly braces so str.format doesn't treat them as placeholders."""
            return text.replace('{', '{{').replace('}', '}}')

        # Convert tool schema to string for formatting if it's a dict
        tool_schema_str = str(tool_schema) if isinstance(tool_schema, dict) else tool_schema

        param_generation_template = self._load_prompt("param_generation")
        prompt = param_generation_template.format(
            tool_id=tool_id,
            tool_schema=_escape_braces(tool_schema_str),
            memory_enum=_escape_braces(memory_enum),
            goal=state.goal,
        )
        logger.debug(f"Parameter generation prompt:\n{prompt}")
        
        messages = [{"role": "user", "content": prompt}]
        logger.info("Calling LLM for parameter generation")
        args_json = self.llm.chat(messages=messages)
        logger.info(f"LLM parameter response:\n{args_json}")
        
        try:
            logger.info("Parsing JSON parameters")
            args: Dict[str, Any] = self._safe_json_loads(args_json)
            logger.debug(f"Parsed args: {args}")
        except ValueError as e:
            logger.error(f"Failed to parse JSON args: {e}")
            logger.error(f"Raw args_json: {args_json}")
            raise RuntimeError(f"LLM produced invalid JSON args: {e}\n{args_json}")

        # Host‑side memory placeholder substitution (simple impl)
        logger.info("Resolving memory placeholders")
        concrete_args = self._resolve_placeholders(args)
        logger.debug(f"Concrete args after placeholder resolution: {concrete_args}")

        logger.info(f"Executing tool {tool_id} with args: {concrete_args}")
        result = self.jentic.execute(tool_id, concrete_args)
        logger.info(f"Tool execution result: {result}")
        return result

    # 4. OBSERVE --------------------------------------------------------
    def observe(self, observation: Any, state: ReasonerState):
        logger.info("=== OBSERVATION PHASE ===")
        logger.info(f"Processing observation: {observation}")
        
        if not state.plan:
            logger.error("No current step to observe - plan is empty!")
            return state
            
        current_step = state.plan[0]
        logger.info(f"Updating step: {current_step.text}")
        
        # Unpack tool results to store only the meaningful, serializable output.
        # This prevents storing non-serializable objects like OperationResult in memory.
        value_to_store = observation
        if isinstance(observation, dict) and "result" in observation:
            result_obj = observation.get("result")
            if hasattr(result_obj, "output"):
                logger.debug("Unpacking tool result object to store its output.")
                value_to_store = result_obj.output

        current_step.result = value_to_store
        current_step.status = "done"
        logger.debug(f"Step status updated to: {current_step.status}")

        if current_step.store_key:
            logger.info(f"Storing result in memory with key: {current_step.store_key}")
            self.memory.set(
                key=current_step.store_key,
                value=value_to_store,
                description=f"Result from step '{current_step.text}'",
            )
            logger.debug(f"Memory updated with key '{current_step.store_key}'")

        history_entry = f"{current_step.text} -> done"
        state.history.append(history_entry)
        logger.debug(f"Added to history: {history_entry}")
        
        # Check if we got a successful API response that created something
        if self._check_successful_creation(observation):
            logger.info("Detected successful creation/completion. Marking goal as complete.")
            state.goal_completed = True
            state.plan.clear()  # Clear remaining steps
        else:
            logger.info("Removing completed step from plan")
            state.plan.popleft()  # advance to next step
            
        logger.info(f"Remaining steps in plan: {len(state.plan)}")
        
        return state

    def _check_successful_creation(self, observation: Any) -> bool:
        """Check if the observation shows we successfully created/completed something."""
        
        if isinstance(observation, dict):
            result = observation.get('result')
            if result and hasattr(result, 'success') and result.success:
                if hasattr(result, 'output') and result.output:
                    output = result.output
                    if isinstance(output, dict):
                        # Look for creation indicators: IDs, timestamps, URLs
                        creation_indicators = ['id', 'message_id', 'timestamp', 'url']
                        found = [key for key in creation_indicators if key in output]
                        if found:
                            logger.info(f"Found creation indicators: {found}")
                            return True
        
        return False

    # 5. EVALUATE -------------------------------------------------------
    def evaluate(self, state: ReasonerState) -> bool:
        logger.info("=== EVALUATION PHASE ===")
        is_complete = not state.plan
        logger.info(f"Plan complete: {is_complete} (remaining steps: {len(state.plan)})")
        
        if is_complete:
            logger.info("All steps completed successfully!")
        else:
            logger.info(f"Next step to execute: {state.plan[0].text if state.plan else 'None'}")
            
        return is_complete

    # 6. REFLECT (optional) --------------------------------------------
    def reflect(self, current_step: Step, err_msg: str) -> bool:
        logger.info("=== REFLECTION PHASE ===")
        logger.info(f"Reflecting on failed step: {current_step.text}")
        logger.info(f"Error message: {err_msg}")
        logger.info(f"Reflection attempts so far: {current_step.reflection_attempts}")
        
        # Limit reflection attempts to prevent infinite loops
        if current_step.reflection_attempts >= MAX_REFLECTION_ATTEMPTS:
            logger.warning(
                "Max reflection attempts (%s) reached for step, giving up",
                MAX_REFLECTION_ATTEMPTS,
            )
            return False
            
        current_step.reflection_attempts += 1
        
        # Generic fallback - extract key action words
        words = current_step.text.split()
        # Filter for meaningful words (nouns, verbs) and take the first few
        key_words = [w.strip('.,!?:;').lower() for w in words if len(w) > 3 and w.isalpha()][:4]
        
        if key_words:
            revised_step = " ".join(key_words)
        else:
            revised_step = "general purpose tool"
        
        # Ensure single line and reasonable length
        if revised_step:
            revised_step = revised_step.strip().replace('\n', ' ')[:80]
            
            logger.info(f"Simplified step from '{current_step.text}' to '{revised_step}'")
            current_step.text = revised_step
            current_step.status = "pending"
            current_step.tool_id = None
            return True
        else:
            logger.warning("Could not generate meaningful revision")
            return False

    # 7. STEP CLASSIFICATION --------------------------------------------
    def _classify_step(self, step: Step, state: ReasonerState) -> StepType:
        """Classify a plan step as TOOL_USING or REASONING via a lightweight LLM prompt.

        The prompt is intentionally minimal to control token cost and reduce
        hallucination risk.  If the LLM response is not recognised, we fall
        back to TOOL_USING to keep the agent progressing.
        """
        logger.info("Classifying step: '%s'", step.text)

        # Summarise memory keys only (avoid dumping large payloads).
        mem_keys: List[str] = []
        if hasattr(self.memory, "keys"):
            try:
                mem_keys = list(self.memory.keys())  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not list memory keys: %s", exc)
        context_summary = (
            "Memory keys: " + ", ".join(mem_keys) if mem_keys else "Memory is empty."
        )

        prompt = (
            "You are a classifier that decides whether a plan step needs an external "
            "API/tool (`tool-using`) or can be solved by internal reasoning over the "
            "already-available data (`reasoning`).\n\n"
            f"Context: {context_summary}\n"
            f"Step: '{step.text}'\n\n"
            "Reply with exactly 'tool-using' or 'reasoning'."
        )

        try:
            reply = (
                self.llm.chat(messages=[{"role": "user", "content": prompt}])
                .strip()
                .lower()
            )
            logger.debug("Classifier reply: %s", reply)
            if "reason" in reply:
                return StepType.REASONING
            if "tool" in reply:
                return StepType.TOOL_USING
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM classification error: %s", exc)

        # Default/fallback
        logger.info("Falling back to TOOL_USING classification")
        return StepType.TOOL_USING

    # 8. EXECUTE REASONING STEP ----------------------------------------
    def _execute_reasoning_step(self, step: Step, state: ReasonerState) -> Any:
        """Run an internal-reasoning step via the LLM.

        The prompt receives a *compact* JSON view of memory to keep context
        size under control.  The LLM should output ONLY the result (no
        explanatory text).  We make a best-effort attempt to parse JSON if it
        looks like JSON; otherwise we return the raw string.
        """
        logger.info("Executing reasoning step: '%s'", step.text)

        # Build a JSON payload of *relevant* memory keys.
        # 1. Explicitly include any key that appears in the step text (e.g. "search_results").
        # 2. For other keys, include only a truncated preview to keep token usage reasonable.
        mem_payload: Dict[str, Any] = {}
        referenced_keys = {k for k in getattr(self.memory, 'keys', lambda: [])() if k in step.text}

        try:
            all_keys = self.memory.keys()
            for k in all_keys:
                v = self.memory.retrieve(k)
                if k in referenced_keys:
                    # Include full value (may still be large JSON)
                    mem_payload[k] = v
                else:
                    # Provide short preview for context only
                    if isinstance(v, str):
                        mem_payload[k] = v[:200] + ("…" if len(v) > 200 else "")
                    else:
                        mem_payload[k] = v  # non-string values are usually small JSON anyway
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not build memory payload: %s", exc)

        reasoning_template = self._load_prompt("reasoning_prompt")
        reasoning_prompt = reasoning_template.format(
            step=step.text, 
            mem=json.dumps(mem_payload, indent=2)
        )

        try:
            reply = self.llm.chat(messages=[{"role": "user", "content": reasoning_prompt}]).strip()
            logger.debug("Reasoning LLM reply: %s", reply)

            # Attempt to parse JSON result if present. If successful, resolve
            # placeholders within the structure. Otherwise, resolve on the raw string.
            if reply.startswith("{") and reply.endswith("}"):
                try:
                    parsed_json = json.loads(reply)
                    return self._resolve_placeholders(parsed_json)
                except json.JSONDecodeError:
                    # Not valid JSON, fall through to treat as a raw string
                    pass
        
            return self._resolve_placeholders(reply)
        except Exception as exc:  # noqa: BLE001
            logger.error("Reasoning step failed: %s", exc)
            return f"Error during reasoning: {exc}"

    # ------------------------------------------------------------------
    # REQUIRED PUBLIC API (BaseReasoner)
    # ------------------------------------------------------------------

    def run(self, goal: str, max_iterations: int = 10):  # type: ignore[override]
        """Execute the reasoning loop until all plan steps are done or iteration cap reached."""
        logger.info("=== STARTING REASONING LOOP ===")
        logger.info(f"Goal: {goal}")
        logger.info(f"Max iterations: {max_iterations}")
        
        from .base_reasoner import ReasoningResult  # local import to avoid circular

        state = self._init_state(goal, {})
        tool_calls: List[Dict[str, Any]] = []

        iteration = 0
        while iteration < max_iterations:
            logger.info(f"=== ITERATION {iteration + 1}/{max_iterations} ===")
            
            # Check if goal is already marked as completed
            if state.goal_completed:
                logger.info("Goal marked as completed! Breaking from loop")
                break
            
            # Ensure we have at least one step planned.
            if not state.plan:
                logger.info("No plan exists, generating plan")
                self.plan(state)

            if self.evaluate(state):
                logger.info("Goal achieved! Breaking from loop")
                break  # goal achieved

            if not state.plan:
                logger.error("No steps in plan after planning phase!")
                break

            current_step = state.plan[0]
            logger.info(f"Executing step: {current_step.text}")

            step_type = self._classify_step(current_step, state)
            logger.info(f"Step classified as: {step_type.value}")

            try:
                if step_type is StepType.TOOL_USING:
                    tool_id = self.select_tool(current_step, state)
                    logger.info(f"Selected tool: {tool_id}")

                    result = self.act(tool_id, state)
                    logger.info(f"Action completed with result type: {type(result)}")

                    tool_calls.append({
                        "tool_id": tool_id,
                        "step": current_step.text,
                        "result": result,
                    })
                else:
                    result = self._execute_reasoning_step(current_step, state)
                    logger.info("Reasoning step output produced")

                self.observe(result, state)
                logger.info("Observation phase completed")
                
            except Exception as e:  # noqa: BLE001
                logger.error(f"Step execution failed: {e}")
                logger.exception("Full exception details:")
                
                err_msg = str(e)
                state.history.append(f"Step failed: {err_msg}")

                # Ask the LLM to repair / re-phrase the step
                logger.info("Attempting to reflect and revise step")
                if not self.reflect(current_step, err_msg):
                    # If reflection returns False we remove the step to avoid loops
                    logger.warning("Reflection failed, marking step as failed and removing")
                    current_step.status = "failed"
                    state.plan.popleft()
                else:
                    logger.info("Step revised, will retry on next iteration")

            iteration += 1
            logger.info(f"Iteration {iteration} completed")

        logger.info("=== REASONING LOOP COMPLETE ===")
        success = state.goal_completed or self.evaluate(state)
        logger.info(f"Final success status: {success} (goal_completed: {state.goal_completed})")
        logger.info(f"Total tool calls made: {len(tool_calls)}")
        logger.info(f"Final history: {state.history}")

        final_answer = "Goal completed." if success else "Unable to complete goal within iteration limit."
        logger.info(f"Final answer: {final_answer}")

        result = ReasoningResult(
            final_answer=final_answer,
            iterations=len(tool_calls),
            tool_calls=tool_calls,
            success=success,
            error_message=None if success else "Max iterations reached or failure during steps",
        )
        logger.info(f"Returning result: {result}")
        return result

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_fenced_code(text: str) -> str:
        """Return the first triple‑backtick‑fenced block, else raise."""
        logger.debug("Extracting fenced code from text")
        m = re.search(r"```[\s\S]+?```", text)
        if not m:
            logger.error("No fenced plan in LLM response")
            raise RuntimeError("No fenced plan in LLM response")
        fenced = m.group(0)
        logger.debug(f"Found fenced block: {fenced}")
        
        # Remove opening and closing fences (```)
        inner = fenced.strip("`")  # remove all backticks at ends
        # After stripping, drop any leading language hint (e.g. ```markdown)
        if "\n" in inner:
            inner = inner.split("\n", 1)[1]  # drop first line (language) if present
        # Remove trailing fence that may remain after stripping leading backticks
        if inner.endswith("```"):
            inner = inner[:-3]
        result = inner.strip()
        logger.debug(f"Extracted inner content: {result}")
        return result

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        """Parse JSON even if the LLM wrapped it in a Markdown fence."""
        logger.debug(f"Parsing JSON from text: {text}")
        text = text.strip()
        
        # Check if text is wrapped in markdown code fences
        if text.startswith("```") and "```" in text[3:]:
            # Extract content between markdown fences
            pattern = r"```(?:json)?\s*([\s\S]+?)\s*```"
            match = re.search(pattern, text)
            if match:
                text = match.group(1).strip()
                logger.debug(f"Removed markdown fences from JSON")
        
        try:
            result = json.loads(text or "{}")
            logger.debug(f"Parsed JSON result: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Failed to parse JSON: {e}\n{text}")

    def _resolve_placeholders(self, obj: Any) -> Any:
        """Delegate placeholder resolution to ScratchPadMemory."""
        logger.debug(f"Resolving placeholders in: {obj}")
        try:
            result = self.memory.resolve_placeholders(obj)
            logger.debug(f"Placeholder resolution result: {result}")
            return result
        except KeyError as e:
            logger.warning(f"Memory placeholder resolution failed: {e}")
            logger.warning("Continuing with unresolved placeholders - this may cause tool execution to fail")
            # Return the original object with unresolved placeholders
            return obj

    
