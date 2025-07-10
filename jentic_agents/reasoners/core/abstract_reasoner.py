"""
ReasonerInfrastructure - Concrete base class with shared infrastructure for all reasoners.

This class contains the ACTUAL common code extracted from all 4 reasoners:
- Universal initialization pattern found in all reasoners
- Tool execution pipeline (search → load → execute) used by all
- Result creation pattern identical across all reasoners
- Tool call tracking and iteration management
- Configuration and dependency management
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import re

from ..base_reasoner import BaseReasoner, ReasoningResult
from ...platform.jentic_client import JenticClient
from ...memory.base_memory import BaseMemory
from ...utils.llm import BaseLLM, LiteLLMChatLLM
from ...communication.hitl.base_intervention_hub import BaseInterventionHub, NoEscalation
from ...utils.logger import get_logger
from ...utils.config import get_config

logger = get_logger(__name__)


class ReasonerInfrastructure(BaseReasoner):
    """
    Concrete base class providing shared infrastructure for all reasoners.
    
    Contains the ACTUAL common code extracted from all 4 reasoner implementations.
    This is not theoretical - every method here exists in multiple reasoners.
    """
    
    def __init__(
        self,
        jentic_client: JenticClient,
        memory: BaseMemory,
        llm: Optional[BaseLLM] = None,
        model: Optional[str] = None,
        intervention_hub: Optional[BaseInterventionHub] = None,
        max_iterations: int = 20,
        **kwargs  # Allow additional args for specific reasoners
    ):
        """
        Universal initialization pattern found in ALL 4 reasoners.
        
        This exact pattern exists in Standard, BulletPlan, Freeform, and Hybrid.
        """
        # Core dependencies - identical in all reasoners
        self.jentic_client = jentic_client  # Standard uses jentic_client
        self.jentic = jentic_client  # Freeform/BulletPlan use jentic (alias for compatibility)
        self.memory = memory
        
        # LLM initialization pattern - found in all reasoners
        config = get_config()
        default_model = config.get("llm", {}).get("model", "gpt-4o")
        self.llm = llm or LiteLLMChatLLM(model=model or default_model)
        self.model = model or default_model
        
        # Intervention hub - identical pattern in all reasoners
        self.intervention_hub = intervention_hub or NoEscalation()
        self.escalation = self.intervention_hub  # BulletPlan uses escalation attribute
        
        # Universal state tracking - all reasoners maintain these
        self.tool_calls: List[Dict[str, Any]] = []
        self.iteration_count: int = 0
        self.max_iterations = max_iterations
        
        # Escalation tracking - used by BulletPlan and Freeform
        self._last_escalation_question: Optional[str] = None
        
        logger.info(f"Initialized {self.__class__.__name__} with max_iterations={max_iterations}")
    
    @abstractmethod
    def run(self, goal: str, **kwargs) -> ReasoningResult:
        """
        Execute the reasoning loop for a given goal.
        
        Each reasoner implements this method with their own strategy.
        Standard: ReAct pattern with explicit phases
        BulletPlan: Plan-first with step execution
        Freeform: Conversational with embedded tools
        Hybrid: Route to Freeform or BulletPlan
        
        Args:
            goal: The objective to achieve
            **kwargs: Additional arguments specific to the reasoner
            
        Returns:
            ReasoningResult with final answer and execution metadata
        """
        pass
    
    def safe_llm_call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        EXACT method extracted from BulletPlanReasoner lines 204-224.
        
        Call LLM in async-safe way. If we're in an async context, run in thread pool
        to avoid blocking the event loop. Otherwise use sync method.
        
        This prevents Discord bot freezing when reasoners are used in async contexts.
        """
        try:
            # Check if we're in an async context
            import asyncio
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # We're in an async context, run in thread pool to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.llm.chat, messages, **kwargs)
                    return future.result()
        except RuntimeError:
            # No running event loop, use sync method
            pass
        
        # Use sync method
        return self.llm.chat(messages, **kwargs)
    
    def execute_tool_safely(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        EXACT tool execution pattern found in ALL reasoners.
        
        Universal sequence: jentic_client.execute(tool_id, params) + tool_calls tracking
        Found in Standard:114-116, BulletPlan:577-578, Freeform:599-608
        
        Args:
            tool_id: ID of the tool to execute
            params: Parameters for tool execution
            
        Returns:
            Tool execution result (matches jentic_client.execute return format)
        """
        logger.info(f"Executing tool: {tool_id}")
        
        try:
            # Execute the tool - universal pattern across all reasoners
            result = self.jentic_client.execute(tool_id, params)
            
            # Track the tool call - universal pattern across all reasoners
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
            
            # Return error result - format matches jentic_client error responses
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
        EXACT result creation pattern found in ALL reasoners.
        
        Universal ReasoningResult creation from Standard:74-79,95-100,151-157, 
        BulletPlan:1208-1219, Freeform:675-681. Identical structure across all.
        
        Args:
            final_answer: The final answer or result
            success: Whether the reasoning was successful
            error_message: Optional error message
            
        Returns:
            ReasoningResult object with standardized format
        """
        return ReasoningResult(
            final_answer=final_answer,
            iterations=self.iteration_count,
            tool_calls=self.tool_calls,
            success=success,
            error_message=error_message
        )
    
    def increment_iteration(self) -> None:
        """
        Universal iteration tracking used by all reasoners.
        
        All reasoners need to track iterations for safety limits and result metadata.
        """
        self.iteration_count += 1
        logger.debug(f"Iteration count: {self.iteration_count}")
    
    def reset_state(self) -> None:
        """
        Reset reasoner state for reuse - common pattern for all reasoners.
        """
        self.tool_calls.clear()
        self.iteration_count = 0
        logger.debug("Reasoner state reset")
    
    # Template methods - reasoners can override for specific implementations
    def prepare_tool_parameters(self, tool_info: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Template method for parameter preparation. 
        
        Default: return empty dict. Reasoners override with their specific logic.
        Standard: uses LLM to generate params
        BulletPlan: complex validation with memory placeholders
        Freeform: parses from embedded JSON
        """
        return {}
    
    def process_tool_result(self, result: Dict[str, Any]) -> str:
        """
        Template method for result processing.
        
        Default: simple string conversion. Reasoners override with their specific logic.
        Standard: checks status and formats message
        BulletPlan: complex success determination and memory storage
        Freeform: formats for conversation
        """
        return str(result)
    
    def process_llm_response_for_escalation(self, response: str, context: str = "") -> str:
        """
        EXACT method extracted from BulletPlanReasoner lines 378-439.
        Also identical logic in FreeformReasoner.
        
        Check if LLM response contains XML escalation request and handle it.
        """
        response = response.strip()

        # Check for XML escalation pattern
        escalation_pattern = (
            r'<escalate_to_human\s+reason="([^"]+)"\s+question="([^"]+)"\s*/>'
        )
        match = re.search(escalation_pattern, response)

        if match:
            reason = match.group(1).strip()
            question = match.group(2).strip()
            logger.info(f"🤖➡️👤 LLM requested escalation: {reason}")
            
            # Store the question for later reference
            self._last_escalation_question = question
            
            if self.intervention_hub.is_available():
                try:
                    human_response = self.intervention_hub.ask_human(question, context)
                    if human_response.strip():
                        logger.info(f"👤➡️🤖 Human provided response: {human_response}")

                        # Store human guidance in memory for future LLM calls to reference
                        guidance_key = (
                            f"human_guidance_{len(self.memory.keys())}"  # Unique key
                        )
                        self.memory.set(
                            key=guidance_key,
                            value=human_response,
                            description=f"Human guidance for: {question}",
                        )
                        # Also store the latest guidance under a well-known key
                        self.memory.set(
                            key="human_guidance_latest",
                            value=human_response,
                            description=f"Latest human guidance: {question}",
                        )
                        logger.info(f"Stored human guidance in memory: {guidance_key}")

                        return human_response
                    else:
                        logger.warning(
                            "👤 No response from human, continuing with original"
                        )
                except Exception as e:
                    logger.warning(f"Escalation failed: {e}")
            else:
                logger.warning("⚠️ Escalation requested but not available")

            # Remove the escalation tag from the response
            return re.sub(escalation_pattern, "", response).strip()

        return response
    
    def add_human_guidance_to_prompt(self, base_prompt: str) -> str:
        """
        EXACT method from BulletPlanReasoner lines 1309-1320.
        
        Add recent human guidance from memory to prompts.
        """
        try:
            # Get latest human guidance from memory
            latest_guidance = self.memory.retrieve("human_guidance_latest")
            if latest_guidance and latest_guidance.strip():
                guidance_section = f"\n\nRECENT HUMAN GUIDANCE: {latest_guidance}\n"
                return base_prompt + guidance_section
        except KeyError:
            # No human guidance in memory yet
            pass
        return base_prompt
    
    # BaseReasoner abstract methods - implemented as template methods
    def plan(self, goal: str, context: Dict[str, Any]) -> str:
        """
        Template method for planning. Default implementation returns empty string.
        Reasoners override with their specific planning logic.
        """
        return ""
    
    def select_tool(
        self, plan: str, available_tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Template method for tool selection. Default implementation returns first tool.
        Reasoners override with their specific selection logic.
        """
        return available_tools[0] if available_tools else None
    
    def act(self, tool: Dict[str, Any], plan: str) -> Dict[str, Any]:
        """
        Template method for action execution. Default implementation returns empty dict.
        Reasoners override with their specific action logic.
        """
        return {}
    
    def observe(self, action_result: Dict[str, Any]) -> str:
        """
        Template method for observation. Default implementation returns string representation.
        Reasoners override with their specific observation logic.
        """
        return str(action_result)
    
    def evaluate(self, goal: str, observations: List[str]) -> bool:
        """
        Template method for evaluation. Default implementation returns False.
        Reasoners override with their specific evaluation logic.
        """
        return False
    
    def reflect(
        self, goal: str, observations: List[str], failed_attempts: List[str]
    ) -> str:
        """
        Template method for reflection. Default implementation returns empty string.
        Reasoners override with their specific reflection logic.
        """
        return ""