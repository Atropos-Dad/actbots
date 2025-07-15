"""Step execution logic for BulletPlanReasoner."""

import json
from typing import Any, Dict

from ...utils.logger import get_logger
from ...utils.async_helpers import safe_llm_call as _global_safe_llm_call
from ...utils.prompt_loader import load_prompt
from ...utils.parsing_helpers import make_json_serializable
from .reasoner_state import Step, ReasonerState

logger = get_logger(__name__)


class StepExecutor:
    """Handles execution of both tool-based and reasoning steps."""

    def __init__(self, jentic_client, memory, llm, intervention_hub=None):
        self.jentic_client = jentic_client
        self.memory = memory
        self.llm = llm
        self.intervention_hub = intervention_hub
        self._last_escalation_question = None

    def execute_tool_step(
        self, tool_id: str, params: Dict[str, Any], step: Step
    ) -> Any:
        """Execute a tool-based step."""
        logger.info(f"Executing tool step: {tool_id}")
        logger.debug(f"Tool parameters: {params}")

        # Resolve tool ID and load schema
        resolved_tool_id = self._resolve_tool_id_from_memory(tool_id)
        
        # Substitute memory placeholders in parameters
        concrete_args = self.memory.resolve_placeholders(params)
        logger.debug(f"Concrete args after placeholder resolution: {concrete_args}")

        # Execute the tool
        logger.info(f"Executing tool {resolved_tool_id} with parameters.")
        result = self._execute_tool_safely(resolved_tool_id, concrete_args)

        success = self._determine_tool_execution_success(result)
        logger.info(f"Tool execution completed. Success: {success}")
        return result

    def execute_reasoning_step(self, step: Step, state: ReasonerState) -> Any:
        """Execute an internal reasoning step via LLM."""
        logger.info(f"Executing reasoning step: '{step.text}'")

        # Build memory payload with referenced keys
        mem_payload = self._build_memory_payload(step)

        # Prepare reasoning prompt
        reasoning_prompt = self._build_reasoning_prompt(step, mem_payload)

        try:
            # Add human guidance context if available
            context_aware_prompt = self._add_human_guidance_to_prompt(reasoning_prompt)
            reply = self._safe_llm_call([{"role": "user", "content": context_aware_prompt}]).strip()
            logger.debug("Reasoning LLM reply: %s", reply)

            # Process for escalation
            context = f"Step: {step.text}\nPhase: Reasoning\nGoal: {state.goal}"
            processed_reply = self._process_llm_response_for_escalation(reply, context)

            if processed_reply != reply:
                # Human provided guidance, use it as the reasoning result
                logger.info("Reasoning step escalated, using human guidance as result")
                return self.memory.resolve_placeholders(processed_reply)

            # Parse JSON result if present, otherwise treat as string
            if processed_reply.startswith("{") and processed_reply.endswith("}"):
                try:
                    parsed_json = json.loads(processed_reply)
                    return self.memory.resolve_placeholders(parsed_json)
                except json.JSONDecodeError:
                    # Not valid JSON, fall through to treat as raw string
                    pass

            return self.memory.resolve_placeholders(processed_reply)
            
        except Exception as exc:
            logger.error("Reasoning step failed: %s", exc)
            return f"Error during reasoning: {exc}"

    def _resolve_tool_id_from_memory(self, tool_id: str) -> str:
        """Resolve placeholders in a stored tool ID reference."""
        resolved = self.memory.resolve_placeholders({"id": tool_id})
        return resolved.get("id", tool_id)

    def _execute_tool_safely(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with error handling."""
        try:
            result = self.jentic_client.execute(tool_id, params)
            logger.info(f"Tool execution completed: {tool_id}")
            return result
        except Exception as e:
            error_msg = f"Tool execution failed for {tool_id}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": str(e),
                "tool_id": tool_id,
                "params": params
            }

    def _determine_tool_execution_success(self, result: Any) -> bool:
        """Determine if tool execution was successful."""
        if isinstance(result, dict):
            inner = result.get("result", result)
            if isinstance(inner, dict):
                return inner.get("success", False)
            return bool(inner)
        return getattr(result, "success", False)

    def _build_memory_payload(self, step: Step) -> Dict[str, Any]:
        """Build memory payload for reasoning step."""
        mem_payload = {}
        referenced_keys = {
            k for k in self.memory.keys() if k in step.text
        }

        try:
            all_keys = self.memory.keys()
            for k in all_keys:
                v = self.memory.retrieve(k)
                # Ensure value is JSON-serializable
                v = make_json_serializable(v)
                
                if k in referenced_keys:
                    # Include full value for referenced keys
                    mem_payload[k] = v
                else:
                    # Provide short preview for context only
                    if isinstance(v, str):
                        mem_payload[k] = v[:200] + ("…" if len(v) > 200 else "")
                    else:
                        mem_payload[k] = v  # Non-string values are usually small
        except Exception as exc:
            logger.debug("Could not build memory payload: %s", exc)

        return mem_payload

    def _build_reasoning_prompt(self, step: Step, mem_payload: Dict[str, Any]) -> str:
        """Build the reasoning prompt for LLM."""
        reasoning_template = load_prompt("reasoning_prompt")
        
        if isinstance(reasoning_template, dict):
            reasoning_template["inputs"]["step"] = step.text
            reasoning_template["inputs"]["memory"] = json.dumps(mem_payload, indent=2)
            reasoning_prompt = json.dumps(reasoning_template, ensure_ascii=False)
        else:
            reasoning_prompt = reasoning_template.format(
                step=step.text, 
                mem=json.dumps(mem_payload, indent=2)
            )
        
        return reasoning_prompt

    def _safe_llm_call(self, messages, **kwargs) -> str:  # type: ignore[override]
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

    def _process_llm_response_for_escalation(self, response: str, context: str = "") -> str:
        """Process LLM response for potential human escalation requests."""
        import re
        
        response = response.strip()
        
        escalation_pattern = (
            r'<escalate_to_human\s+reason="([^"]+)"\s+question="([^"]+)"\s*/>'
        )
        match = re.search(escalation_pattern, response)
        
        if match:
            reason = match.group(1).strip()
            question = match.group(2).strip()
            logger.info(f"🤖➡️👤 LLM requested escalation: {reason}")
            
            # Store the question for potential later use
            self._last_escalation_question = question
            
            if hasattr(self, 'intervention_hub') and self.intervention_hub and self.intervention_hub.is_available():
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
            response = re.sub(escalation_pattern, "", response).strip()
        
        return response 