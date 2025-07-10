"""
Abstract base class providing shared infrastructure for all reasoners.

This class extracts common functionality that all reasoners need:
- Tool execution pipeline (search → load → execute)
- Memory integration and placeholder resolution
- LLM communication with async safety
- Error handling and result processing
- Iteration and tool call tracking
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from ..base_reasoner import ReasoningResult
from ...platform.jentic_client import JenticClient
from ...memory.base_memory import BaseMemory
from ...utils.llm import BaseLLM
from ...communication.hitl.base_intervention_hub import BaseInterventionHub, NoEscalation
from ...utils.logger import get_logger

logger = get_logger(__name__)


class AbstractReasoner(ABC):
    """
    Abstract base class providing shared infrastructure for all reasoners.
    
    All reasoners inherit from this class to get common functionality while
    implementing their own reasoning strategies in the run() method.
    """
    
    def __init__(
        self,
        jentic_client: JenticClient,
        memory: BaseMemory,
        llm: BaseLLM,
        intervention_hub: Optional[BaseInterventionHub] = None,
        max_iterations: int = 20,
    ):
        """
        Initialize common reasoner infrastructure.
        
        Args:
            jentic_client: Client for tool search and execution
            memory: Memory system for state persistence  
            llm: Language model interface
            intervention_hub: Human intervention hub for escalations
            max_iterations: Safety limit on reasoning iterations
        """
        self.jentic_client = jentic_client
        self.memory = memory
        self.llm = llm
        self.intervention_hub = intervention_hub or NoEscalation()
        self.max_iterations = max_iterations
        
        # Shared state tracking
        self.tool_calls: List[Dict[str, Any]] = []
        self.iteration_count: int = 0
        self.error_messages: List[str] = []
        
        logger.info(f"Initialized {self.__class__.__name__} with max_iterations={max_iterations}")
    
    @abstractmethod
    def run(self, goal: str, **kwargs) -> ReasoningResult:
        """
        Execute the reasoning loop for a given goal.
        
        Each reasoner implements this method with their own strategy.
        
        Args:
            goal: The objective to achieve
            **kwargs: Additional arguments specific to the reasoner
            
        Returns:
            ReasoningResult with final answer and execution metadata
        """
        pass
    
    def safe_llm_call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Call LLM in async-safe way to avoid blocking event loops.
        
        If we're in an async context, run in thread pool to avoid blocking.
        Otherwise use sync method.
        
        Args:
            messages: Chat messages for the LLM
            **kwargs: Additional LLM parameters
            
        Returns:
            LLM response string
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
        Execute a tool with comprehensive error handling and result processing.
        
        Args:
            tool_id: ID of the tool to execute
            params: Parameters for tool execution
            
        Returns:
            Standardized tool execution result
        """
        logger.info(f"Executing tool safely: {tool_id}")
        
        try:
            # Resolve memory placeholders in parameters
            resolved_params = self.resolve_memory_placeholders(params)
            logger.debug(f"Resolved parameters: {resolved_params}")
            
            # Execute the tool
            result = self.jentic_client.execute(tool_id, resolved_params)
            
            # Track the tool call
            call_record = {
                "tool_id": tool_id,
                "params": resolved_params,
                "result": result,
                "iteration": self.iteration_count,
                "success": self.determine_execution_success(result)
            }
            self.tool_calls.append(call_record)
            
            logger.info(f"Tool execution completed successfully: {tool_id}")
            return result
            
        except Exception as e:
            error_msg = f"Tool execution failed for {tool_id}: {str(e)}"
            logger.error(error_msg)
            self.error_messages.append(error_msg)
            
            # Return standardized error result
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
                "success": False
            }
            self.tool_calls.append(call_record)
            
            return error_result
    
    def resolve_memory_placeholders(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve memory placeholders in tool parameters.
        
        Args:
            params: Parameters that may contain memory placeholders
            
        Returns:
            Parameters with placeholders resolved to actual values
        """
        try:
            return self.memory.resolve_placeholders(params)
        except Exception as e:
            logger.warning(f"Memory placeholder resolution failed: {e}")
            return params  # Return original params if resolution fails
    
    def determine_execution_success(self, result: Any) -> bool:
        """
        Determine if a tool execution was successful based on the result.
        
        Args:
            result: Tool execution result
            
        Returns:
            True if execution was successful, False otherwise
        """
        if isinstance(result, dict):
            # Check for explicit success field
            if "success" in result:
                return bool(result["success"])
            
            # Check for error indicators
            if "error" in result or "status" in result and result["status"] == "error":
                return False
            
            # Check nested result object
            inner_result = result.get("result")
            if hasattr(inner_result, "success"):
                return bool(inner_result.success)
            elif isinstance(inner_result, dict) and "success" in inner_result:
                return bool(inner_result["success"])
        
        # If result has success attribute
        elif hasattr(result, "success"):
            return bool(result.success)
        
        # Default to True if no clear error indicators
        return True
    
    def store_tool_result(
        self, 
        tool_id: str, 
        result: Any, 
        memory_key: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Store tool execution result in memory.
        
        Args:
            tool_id: ID of the executed tool
            result: Tool execution result
            memory_key: Key to store the result under (optional)
            description: Description of the stored result (optional)
        """
        if not memory_key:
            memory_key = f"tool_result_{tool_id}_{self.iteration_count}"
        
        if not description:
            description = f"Result from tool {tool_id} at iteration {self.iteration_count}"
        
        try:
            # Use enhanced memory interface if available
            if hasattr(self.memory, 'set'):
                self.memory.set(memory_key, result, description)
            else:
                # Fallback to basic interface
                self.memory.store(memory_key, result)
                
            logger.debug(f"Stored tool result in memory: {memory_key}")
            
        except Exception as e:
            logger.warning(f"Failed to store tool result in memory: {e}")
    
    def get_memory_summary(self) -> str:
        """
        Get a formatted summary of current memory contents.
        
        Returns:
            Human-readable summary of memory state
        """
        try:
            if hasattr(self.memory, 'enumerate_for_prompt'):
                return self.memory.enumerate_for_prompt()
            elif hasattr(self.memory, 'keys'):
                keys = self.memory.keys()
                if not keys:
                    return "Memory is empty."
                return f"Memory contains {len(keys)} items: {', '.join(keys)}"
            else:
                return "Memory status unknown."
        except Exception as e:
            logger.debug(f"Could not get memory summary: {e}")
            return "Memory status unavailable."
    
    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.iteration_count += 1
        logger.debug(f"Iteration count: {self.iteration_count}")
    
    def reset_state(self) -> None:
        """Reset reasoner state for new goals."""
        self.tool_calls.clear()
        self.error_messages.clear()
        self.iteration_count = 0
        logger.debug("Reasoner state reset")
    
    def create_reasoning_result(
        self,
        final_answer: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> ReasoningResult:
        """
        Create a standardized ReasoningResult.
        
        Args:
            final_answer: The final answer or result
            success: Whether the reasoning was successful
            error_message: Optional error message
            
        Returns:
            ReasoningResult object
        """
        return ReasoningResult(
            final_answer=final_answer,
            iterations=self.iteration_count,
            tool_calls=self.tool_calls,
            success=success,
            error_message=error_message
        )