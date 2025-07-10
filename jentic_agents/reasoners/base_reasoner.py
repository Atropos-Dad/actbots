"""
Abstract base class for reasoning loops that implement plan → select_tool → act → observe → evaluate → reflect.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import enum
import re
import logging
import json
from textwrap import dedent

from ..platform.jentic_client import JenticClient
from ..memory.base_memory import BaseMemory
from ..utils.llm import BaseLLM, LiteLLMChatLLM
from ..communication.hitl.base_intervention_hub import BaseInterventionHub, NoEscalation
from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class StepType(enum.Enum):
    """Category of a plan step used by reasoners to decide execution path."""

    TOOL_USING = "tool-using"
    REASONING = "reasoning"


class ReasoningResult(BaseModel):
    """Result object returned by reasoner.run()"""

    final_answer: str
    iterations: int
    tool_calls: List[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None


class BaseReasoner(ABC):
    """
    Abstract base class defining the reasoning loop contract.

    Implements the ReAct pattern: plan → select_tool → act → observe → evaluate → reflect.
    Subclasses implement the specific reasoning logic while maintaining a consistent interface.
    
    Provides shared infrastructure for all reasoners while keeping the interface clean.
    """

    def __init__(
        self,
        jentic_client: JenticClient,
        memory: BaseMemory,
        llm: Optional[BaseLLM] = None,
        model: Optional[str] = None,
        intervention_hub: Optional[BaseInterventionHub] = None,
        max_iterations: int = 20,
        **kwargs
    ):
        """
        Universal initialization pattern used by all reasoners.
        """
        # Core dependencies
        self.jentic_client = jentic_client
        self.jentic = jentic_client  # Alias for compatibility
        self.memory = memory
        
        # LLM initialization
        config = get_config()
        default_model = config.get("llm", {}).get("model", "gpt-4o")
        self.llm = llm or LiteLLMChatLLM(model=model or default_model)
        self.model = model or default_model
        
        # Intervention hub
        self.intervention_hub = intervention_hub or NoEscalation()
        self.escalation = self.intervention_hub  # Alias for compatibility
        
        # State tracking
        self.tool_calls: List[Dict[str, Any]] = []
        self.iteration_count: int = 0
        self.max_iterations = max_iterations
        self._last_escalation_question: Optional[str] = None
        
        logger.info(f"Initialized {self.__class__.__name__} with max_iterations={max_iterations}")

    @abstractmethod
    def run(self, goal: str, max_iterations: int = 10) -> ReasoningResult:
        """
        Execute the reasoning loop for a given goal.

        Args:
            goal: The objective or question to reason about
            max_iterations: Maximum number of reasoning iterations

        Returns:
            ReasoningResult with final answer and execution metadata
        """
        pass

    @abstractmethod
    def plan(self, goal: str, context: Dict[str, Any]) -> str:
        """
        Generate a plan for achieving the goal.

        Args:
            goal: The objective to plan for
            context: Current reasoning context and history

        Returns:
            A plan description string
        """
        pass

    @abstractmethod
    def select_tool(
        self, plan: str, available_tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Select the most appropriate tool for executing the current plan.

        Args:
            plan: The current plan description
            available_tools: List of available tools/workflows

        Returns:
            Selected tool metadata, or None if no tool is needed
        """
        pass

    @abstractmethod
    def act(self, tool: Dict[str, Any], plan: str) -> Dict[str, Any]:
        """
        Execute an action using the selected tool.

        Args:
            tool: Tool metadata and definition
            plan: Current plan description

        Returns:
            Action parameters to pass to the tool
        """
        pass

    @abstractmethod
    def observe(self, action_result: Dict[str, Any]) -> str:
        """
        Process and interpret the result of an action.

        Args:
            action_result: Result returned from tool execution

        Returns:
            Observation summary string
        """
        pass

    @abstractmethod
    def evaluate(self, goal: str, observations: List[str]) -> bool:
        """
        Evaluate whether the goal has been achieved based on observations.

        Args:
            goal: The original objective
            observations: List of observation summaries from actions

        Returns:
            True if goal is achieved, False otherwise
        """
        pass

    @abstractmethod
    def reflect(
        self, goal: str, observations: List[str], failed_attempts: List[str]
    ) -> str:
        """
        Reflect on failures and generate improved strategies.

        Args:
            goal: The original objective
            observations: List of observation summaries
            failed_attempts: List of previous failed attempt descriptions

        Returns:
            Reflection insights for improving the approach
        """
        pass
    
    # ========================================================================
    # SHARED INFRASTRUCTURE METHODS
    # All common functionality used by multiple reasoners
    # ========================================================================
    
    def safe_llm_call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Call LLM in async-safe way. Prevents Discord bot freezing.
        """
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.llm.chat, messages, **kwargs)
                    return future.result()
        except RuntimeError:
            pass
        return self.llm.chat(messages, **kwargs)
    
    def execute_tool_safely(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Universal tool execution pattern with tracking and error handling.
        """
        logger.info(f"Executing tool: {tool_id}")
        
        try:
            result = self.jentic_client.execute(tool_id, params)
            
            # Track the tool call
            call_record = {
                "tool_id": tool_id,
                "params": params,
                "result": result,
                "iteration": self.iteration_count,
            }
            self.tool_calls.append(call_record)
            
            logger.info(f"Tool execution completed: {tool_id}")
            return result
            
        except Exception as e:
            error_msg = f"Tool execution failed for {tool_id}: {str(e)}"
            logger.error(error_msg)
            
            error_result = {
                "status": "error", 
                "error": str(e),
                "tool_id": tool_id,
                "params": params
            }
            
            # Track failed tool call
            call_record = {
                "tool_id": tool_id,
                "params": params,
                "result": error_result,
                "iteration": self.iteration_count,
            }
            self.tool_calls.append(call_record)
            
            return error_result
    
    def create_reasoning_result(
        self,
        final_answer: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> ReasoningResult:
        """
        Universal result creation pattern used by all reasoners.
        """
        return ReasoningResult(
            final_answer=final_answer,
            iterations=self.iteration_count,
            tool_calls=self.tool_calls,
            success=success,
            error_message=error_message
        )
    
    def process_llm_response_for_escalation(self, response: str, context: str = "") -> str:
        """
        Handle XML escalation requests in LLM responses.
        """
        response = response.strip()
        
        escalation_pattern = (
            r'<escalate_to_human\s+reason="([^"]+)"\s+question="([^"]+)"\s*/>'
        )
        match = re.search(escalation_pattern, response)
        
        if match:
            reason = match.group(1).strip()
            question = match.group(2).strip()
            logger.info(f"🤖➡️👤 LLM requested escalation: {reason}")
            
            self._last_escalation_question = question
            
            if self.intervention_hub.is_available():
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
    
    def add_human_guidance_to_prompt(self, base_prompt: str) -> str:
        """
        Add recent human guidance from memory to prompts.
        """
        try:
            latest_guidance = self.memory.retrieve("human_guidance_latest")
            if latest_guidance and latest_guidance.strip():
                guidance_section = f"\n\nRECENT HUMAN GUIDANCE: {latest_guidance}\n"
                return base_prompt + guidance_section
        except KeyError:
            pass
        return base_prompt
    
    def increment_iteration(self) -> None:
        """Universal iteration tracking."""
        self.iteration_count += 1
        logger.debug(f"Iteration count: {self.iteration_count}")
    
    def reset_state(self) -> None:
        """Reset reasoner state for reuse."""
        self.tool_calls.clear()
        self.iteration_count = 0
        logger.debug("Reasoner state reset")

    # ========================================================================
    # NEW SHARED HIGH-LEVEL HELPERS (deduplicated from concrete reasoners)
    # ========================================================================

    # ---- PARAMETER GENERATION ------------------------------------------------
    def _default_param_prompt(self, tool: Dict[str, Any], plan_context: str) -> str:
        """Return a generic prompt asking the LLM to generate JSON parameters for *tool*.
        Subclasses can override if they need a custom wording.
        """
        # Pretty-print the parameter schema for the prompt
        parameters_block = json.dumps(tool.get("parameters", {}), indent=2, ensure_ascii=False)
        tool_name = tool.get("name", tool.get("id", "unknown tool"))
        tool_desc = tool.get("description", "")
        return dedent(
            f"""
            You are an action-parameter generator. Given a tool specification and the current plan step,
            produce a single JSON object containing arguments for the tool call – **no extra keys**.

            Tool: {tool_name}
            Description: {tool_desc}
            Parameter schema (for reference):
            {parameters_block}

            Current plan step / context:
            {plan_context}

            Respond **only** with a JSON object.
            """
        ).strip()

    def generate_and_validate_parameters(
        self,
        tool: Dict[str, Any],
        plan_context: str,
        *,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """Generic helper that loops LLM → JSON parse → placeholder validation.
        Returns parameters dict or raises *RuntimeError* after *max_attempts*.
        """
        required_fields: List[str] = tool.get("required", [])
        prompt = self._default_param_prompt(tool, plan_context)
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Parameter-generation attempt {attempt}/{max_attempts} for tool '{tool.get('id')}'")
            response = self.safe_llm_call([
                {"role": "user", "content": prompt}
            ], max_tokens=400, temperature=0.3)

            try:
                args = json.loads(response.strip())
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e}"
                logger.warning(last_error)
                # Minimal correction hint
                prompt = (
                    f"ERROR: Your previous output was not valid JSON. Please output only a JSON object.\n\n"
                    f"{self._default_param_prompt(tool, plan_context)}"
                )
                continue

            # Check required fields
            missing = [f for f in required_fields if f not in args]
            if missing:
                last_error = f"Missing required fields: {missing}"
                logger.warning(last_error)
                prompt = (
                    f"ERROR: You omitted required fields {', '.join(missing)}. Please regenerate **all** parameters.\n\n"
                    f"{self._default_param_prompt(tool, plan_context)}"
                )
                continue

            # Validate memory placeholders if the memory backend supports it
            if hasattr(self.memory, "validate_placeholders"):
                error_msg, correction_prompt = self.memory.validate_placeholders(args, required_fields)  # type: ignore[arg-type]
                if error_msg:
                    last_error = error_msg
                    logger.warning(error_msg)
                    prompt = correction_prompt or prompt  # type: ignore[assignment]
                    continue

            # All good
            return args

        raise RuntimeError(
            f"Parameter generation failed after {max_attempts} attempts for tool '{tool.get('id')}'. Last error: {last_error}"
        )

    # ---- TOOL SELECTION WITH LLM --------------------------------------------
    def choose_tool_with_llm(
        self,
        plan_description: str,
        available_tools: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Generic numeric-list based tool choice. Returns selected tool dict or None."""
        if not available_tools:
            return None
        if len(available_tools) == 1:
            return available_tools[0]

        tool_descriptions = "\n".join(
            [f"- {tool['id']}: {tool.get('name', '')} - {tool.get('description', '')}" for tool in available_tools]
        )

        prompt = dedent(
            f"""
            You are a tool-selection assistant. Given a plan step and a list of available tools, choose the most appropriate tool **ID**.
            Reply with one of the IDs exactly, or the word NONE if no tool fits.

            Plan step:
            {plan_description}

            Available tools:
            {tool_descriptions}
            """
        ).strip()

        response = self.safe_llm_call([
            {"role": "user", "content": prompt}
        ], max_tokens=30, temperature=0.0)

        selected_id = (response or "").strip()
        logger.info(f"LLM tool-selection response: '{selected_id}'")

        if selected_id.upper() == "NONE" or selected_id == "":
            return None if selected_id.upper() == "NONE" else available_tools[0]

        for tool in available_tools:
            if tool["id"] == selected_id:
                return tool
        logger.warning(f"No tool matched id '{selected_id}', returning first candidate as fallback")
        return available_tools[0]

    # ---- FINAL ANSWER SYNTHESIS ---------------------------------------------
    def generate_final_answer(self, goal: str, observations: List[str]) -> str:
        """Utility to turn a list of observations into a concise final answer."""
        prompt = dedent(
            f"""
            You are a summarisation assistant. Given the original goal and a list of observations / results, craft a clear, concise final answer.

            Goal:
            {goal}

            Observations:
            {'\n'.join('- ' + o for o in observations)}

            Provide the final answer only, no extra commentary.
            """
        ).strip()
        response = self.safe_llm_call([
            {"role": "user", "content": prompt}
        ], max_tokens=300, temperature=0.5)
        return response.strip()

    # ---- GOAL EVALUATION & REFLECTION ---------------------------------------
    def llm_goal_evaluation(self, goal: str, observations: List[str]) -> bool:
        """Ask the LLM if the goal is achieved based on observations. Returns bool."""
        if not observations:
            return False

        prompt = dedent(
            f"""
            You are an evaluation assistant. Decide whether the following goal has been fully achieved based on the observations.

            Goal:
            {goal}

            Observations:
            {'\n'.join('- ' + o for o in observations)}

            Reply with YES if achieved, otherwise NO. No other words.
            """
        ).strip()

        response = self.safe_llm_call([
            {"role": "user", "content": prompt}
        ], max_tokens=5, temperature=0.1)

        return response.strip().upper() == "YES"

    def llm_reflect(self, goal: str, observations: List[str], failed_attempts: List[str]) -> str:
        """Shared reflection prompt to suggest improvements after failures."""
        prompt = dedent(
            f"""
            You are a reflection assistant. The goal has not been achieved yet. Analyse the situation and propose improved next steps.

            Goal:
            {goal}

            Observations so far:
            {'\n'.join('- ' + o for o in observations) if observations else '(none)'}

            Failed attempts:
            {'\n'.join('- ' + a for a in failed_attempts) if failed_attempts else '(none)'}

            Provide insights or revised plan succinctly.
            """
        ).strip()

        response = self.safe_llm_call([
            {"role": "user", "content": prompt}
        ], max_tokens=200, temperature=0.7)
        return response.strip()

    # ---- RESULT HELPERS ------------------------------------------------------
    @staticmethod
    def is_tool_result_successful(result: Any) -> bool:
        """Duck-type check used by several reasoners."""
        if isinstance(result, dict):
            inner = result.get("result", result)
            if isinstance(inner, dict):
                return inner.get("success", False)
            return bool(inner)
        return getattr(result, "success", False)

    # ---- MEMORY PLACEHOLDER RESOLUTION --------------------------------------
    def resolve_memory_placeholders(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience wrapper with logging for memory placeholder resolution."""
        try:
            resolved = self.memory.resolve_placeholders(args)
            return resolved  # type: ignore[return-value]
        except Exception as e:
            logger.warning(f"Failed to resolve memory placeholders: {e}")
            return args
