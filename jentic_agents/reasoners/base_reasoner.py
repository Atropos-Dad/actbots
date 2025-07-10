"""
Abstract base class for reasoning loops that implement plan → select_tool → act → observe → evaluate → reflect.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import enum
import re
import logging

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
