"""Reflection and failure analysis logic for BulletPlanReasoner."""

import json
import re
from typing import Optional, Tuple

from ...utils.config import get_bullet_plan_config_value
from ...utils.logger import get_logger
from ...utils.async_helpers import safe_llm_call as _global_safe_llm_call
from ...utils.prompt_loader import load_prompt
from .reasoner_state import Step, ReasonerState

logger = get_logger(__name__)


class ReflectionEngine:
    """Handles failure analysis and step revision for BulletPlanReasoner."""

    def __init__(self, memory, llm, intervention_hub=None):
        self.memory = memory
        self.llm = llm
        self.intervention_hub = intervention_hub
        self.max_reflection_attempts = get_bullet_plan_config_value("max_reflection_attempts", 3)
        self._last_escalation_question = None

    def reflect_on_failure(
        self, failed_step: Step, error_msg: str, state: ReasonerState
    ) -> Tuple[bool, Optional[Step]]:
        """
        Reflect on a failed step and attempt to revise it.
        Returns (success, revised_step).
        """
        logger.info(
            f"Reflecting on failed step: {failed_step.text} | "
            f"Error: {error_msg} | "
            f"Attempts: {failed_step.reflection_attempts}"
        )

        # Check if we've exceeded max attempts
        if failed_step.reflection_attempts >= self.max_reflection_attempts:
            logger.warning(
                f"Max reflection attempts ({self.max_reflection_attempts}) reached for step, giving up"
            )
            return False, None

        # Perform LLM-based reflection
        revised_step_text = self._perform_llm_reflection(failed_step, error_msg, state)
        
        if not revised_step_text:
            logger.warning("LLM reflection did not provide a revised step.")
            return False, None

        # Check for unrecoverable failures
        if "AUTH_FAILURE" in revised_step_text:
            logger.error("Reflection indicates an unrecoverable authentication failure.")
            return False, None

        # Create revised step
        revised_step = Step(
            text=revised_step_text,
            indent=failed_step.indent,
            store_key=failed_step.store_key,
            goal_context=failed_step.goal_context,
            status=failed_step.status,  # Reset to pending
            reflection_attempts=failed_step.reflection_attempts + 1,
            load_from_store=failed_step.load_from_store,
            step_type=failed_step.step_type,
        )

        logger.info(f"LLM revised step from '{failed_step.text}' to '{revised_step_text}'")
        return True, revised_step

    def _perform_llm_reflection(
        self, failed_step: Step, error_msg: str, state: ReasonerState
    ) -> Optional[str]:
        """Perform LLM-based reflection on the failed step."""
        # Build reflection prompt
        reflection_prompt = self._build_reflection_prompt(failed_step, error_msg, state)
        
        # Add human guidance context if available
        context_aware_prompt = self._add_human_guidance_to_prompt(reflection_prompt)
        
        # Get LLM response
        revised_step = self._safe_llm_call([{"role": "user", "content": context_aware_prompt}]).strip()

        # Process for escalation during reflection
        context = (
            f"Step: {failed_step.text}\n"
            f"Phase: Reflection\n"
            f"Error: {error_msg}\n"
            f"Goal: {state.goal}"
        )
        processed_step = self._process_llm_response_for_escalation(revised_step, context)

        if processed_step != revised_step:
            # Human provided guidance during reflection
            logger.info("Reflection escalated to human, using human guidance")
            
            # Preserve original step context and incorporate human guidance
            original_step = failed_step.text
            if self._last_escalation_question:
                processed_step = f"{original_step} (human answered '{self._last_escalation_question}' with: {processed_step})"
                self._last_escalation_question = None  # Clear after use
            else:
                processed_step = f"{original_step} (using human guidance: {processed_step})"
            
            return processed_step

        return processed_step if processed_step else None

    def _build_reflection_prompt(
        self, failed_step: Step, error_msg: str, state: ReasonerState
    ) -> str:
        """Build the reflection prompt for LLM."""
        reflection_template = load_prompt("reflection_prompt")
        
        if isinstance(reflection_template, dict):
            reflection_template["inputs"]["goal"] = state.goal
            reflection_template["inputs"]["failed_step_text"] = failed_step.text
            reflection_template["inputs"]["error_message"] = error_msg
            reflection_template["inputs"]["history"] = "\n".join(state.history)
            
            # Add tool information if available
            tool_schema = json.dumps(failed_step.params or {}, indent=2)
            failed_args = json.dumps(getattr(failed_step, "args", {}), indent=2)
            reflection_template["inputs"]["tool_schema"] = tool_schema
            reflection_template["inputs"]["failed_args"] = failed_args
            
            prompt = json.dumps(reflection_template, ensure_ascii=False)
        else:
            prompt = reflection_template.format(
                goal=state.goal,
                failed_step_text=failed_step.text,
                error_message=error_msg,
                history="\n".join(state.history),
            )
        
        return prompt

    def _process_llm_response_for_escalation(self, response: str, context: str = "") -> str:
        """Handle XML escalation requests in LLM responses."""
        response = response.strip()
        
        escalation_pattern = r'<escalate_to_human\s+reason="([^"]+)"\s+question="([^"]+)"\s*/>'
        match = re.search(escalation_pattern, response)
        
        if match:
            reason = match.group(1).strip()
            question = match.group(2).strip()
            logger.info(f"🤖➡️👤 LLM requested escalation: {reason}")
            
            self._last_escalation_question = question
            
            if self.intervention_hub and self.intervention_hub.is_available():
                try:
                    human_response = self.intervention_hub.ask_human(question, context)
                    if human_response.strip():
                        logger.info(f"👤➡️🤖 Human provided response: {human_response}")
                        
                        # Store guidance in memory
                        guidance_key = f"human_guidance_{len(self.memory.keys())}"
                        self.memory.set(
                            key=guidance_key,
                            value=human_response,
                            description=f"Human guidance for: {question}",
                        )
                        self.memory.set(
                            key="human_guidance_latest",
                            value=human_response,
                            description=f"Latest human guidance: {question}",
                        )
                        logger.info(f"Stored human guidance in memory: {guidance_key}")
                        
                        return human_response
                    else:
                        logger.warning("👤 No response from human, continuing with original")
                except Exception as e:
                    logger.warning(f"Escalation failed: {e}")
            else:
                logger.warning("⚠️ Escalation requested but not available")
            
            # Remove escalation tag from response
            return re.sub(escalation_pattern, "", response).strip()
        
        return response

    def _safe_llm_call(self, messages, **kwargs) -> str:
        """Thin wrapper around the shared *safe_llm_call* utility."""
        return _global_safe_llm_call(self.llm, messages, **kwargs)

    def _add_human_guidance_to_prompt(self, base_prompt: str) -> str:
        """Add recent human guidance from memory to prompts."""
        try:
            latest_guidance = self.memory.retrieve("human_guidance_latest")
            if latest_guidance and latest_guidance.strip():
                guidance_section = f"\n\nRECENT HUMAN GUIDANCE: {latest_guidance}\n"
                return base_prompt + guidance_section
        except KeyError:
            pass
        return base_prompt 