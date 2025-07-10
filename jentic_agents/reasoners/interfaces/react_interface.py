"""
ReactInterface - Optional interface for reasoners following the ReAct pattern.

This interface provides the classic ReAct pattern structure while allowing
reasoners to implement only the methods they need. All methods have sensible
defaults, so reasoners can override only what's relevant to their strategy.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from ..core.abstract_reasoner import AbstractReasoner
from ..base_reasoner import ReasoningResult
from ...utils.logger import get_logger

logger = get_logger(__name__)


class ReactInterface(AbstractReasoner):
    """
    Optional interface for reasoners that follow the ReAct pattern.
    
    Provides the classic ReAct structure:
    plan → select_tool → act → observe → evaluate → reflect
    
    All methods have default implementations, so reasoners can override
    only what's relevant to their specific strategy.
    """
    
    @abstractmethod
    def run(self, goal: str, max_iterations: int = 10, **kwargs) -> ReasoningResult:
        """
        Execute the reasoning loop for a given goal.
        
        This is the only method that MUST be implemented by subclasses.
        The implementation can use the ReAct methods below or implement
        a completely different strategy.
        
        Args:
            goal: The objective to achieve
            max_iterations: Maximum number of reasoning iterations
            **kwargs: Additional reasoner-specific arguments
            
        Returns:
            ReasoningResult with final answer and execution metadata
        """
        pass
    
    # Optional ReAct pattern methods with sensible defaults
    
    def plan(self, goal: str, context: Dict[str, Any]) -> str:
        """
        Generate a plan for achieving the goal.
        
        Default implementation returns empty string (no explicit planning).
        Override if your reasoner does explicit planning.
        
        Args:
            goal: The objective to plan for
            context: Current reasoning context and history
            
        Returns:
            A plan description string
        """
        logger.debug(f"Default plan implementation called for goal: {goal}")
        return ""
    
    def select_tool(
        self, plan: str, available_tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Select the most appropriate tool for executing the current plan.
        
        Default implementation returns the first available tool.
        Override if your reasoner has specific tool selection logic.
        
        Args:
            plan: The current plan description
            available_tools: List of available tools/workflows
            
        Returns:
            Selected tool metadata, or None if no tool is needed
        """
        if available_tools:
            selected = available_tools[0]
            logger.debug(f"Default tool selection: {selected.get('id', 'unknown')}")
            return selected
        
        logger.debug("No tools available for selection")
        return None
    
    def act(self, tool: Dict[str, Any], plan: str) -> Dict[str, Any]:
        """
        Execute an action using the selected tool.
        
        Default implementation returns empty parameters.
        Override if your reasoner needs specific parameter generation.
        
        Args:
            tool: Tool metadata and definition
            plan: Current plan description
            
        Returns:
            Action parameters to pass to the tool
        """
        logger.debug(f"Default act implementation for tool: {tool.get('id', 'unknown')}")
        return {}
    
    def observe(self, action_result: Dict[str, Any]) -> str:
        """
        Process and interpret the result of an action.
        
        Default implementation returns a simple result summary.
        Override if your reasoner needs specific observation processing.
        
        Args:
            action_result: Result returned from tool execution
            
        Returns:
            Observation summary string
        """
        if isinstance(action_result, dict):
            if action_result.get("status") == "success":
                result_data = action_result.get("result", "Success")
                observation = f"Action completed successfully. Result: {result_data}"
            else:
                error_data = action_result.get("error", "Unknown error")
                observation = f"Action failed. Error: {error_data}"
        else:
            observation = f"Action result: {str(action_result)}"
        
        logger.debug(f"Default observation: {observation[:100]}...")
        return observation
    
    def evaluate(self, goal: str, observations: List[str]) -> bool:
        """
        Evaluate whether the goal has been achieved based on observations.
        
        Default implementation returns False (goal not achieved).
        Override if your reasoner has specific evaluation logic.
        
        Args:
            goal: The original objective
            observations: List of observation summaries from actions
            
        Returns:
            True if goal is achieved, False otherwise
        """
        logger.debug(f"Default evaluation for goal: {goal} with {len(observations)} observations")
        return False
    
    def reflect(
        self, goal: str, observations: List[str], failed_attempts: List[str]
    ) -> str:
        """
        Reflect on failures and generate improved strategies.
        
        Default implementation returns a simple reflection.
        Override if your reasoner has specific reflection logic.
        
        Args:
            goal: The original objective
            observations: List of observation summaries
            failed_attempts: List of previous failed attempt descriptions
            
        Returns:
            Reflection insights for improving the approach
        """
        reflection = f"Reflection on goal '{goal}': {len(failed_attempts)} failed attempts, {len(observations)} observations."
        if failed_attempts:
            reflection += f" Last failure: {failed_attempts[-1]}"
        
        logger.debug(f"Default reflection: {reflection}")
        return reflection
    
    # Utility methods for common ReAct patterns
    
    def search_tools(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for tools using the Jentic client.
        
        Args:
            query: Search query for tools
            top_k: Maximum number of tools to return
            
        Returns:
            List of matching tools
        """
        logger.debug(f"Searching for tools: '{query}' (top_k={top_k})")
        return self.jentic_client.search(query, top_k=top_k)
    
    def load_tool_definition(self, tool_id: str) -> Dict[str, Any]:
        """
        Load detailed tool definition.
        
        Args:
            tool_id: ID of the tool to load
            
        Returns:
            Tool definition with parameters and schema
        """
        logger.debug(f"Loading tool definition: {tool_id}")
        return self.jentic_client.load(tool_id)
    
    def execute_standard_react_loop(
        self, goal: str, max_iterations: int = 10
    ) -> ReasoningResult:
        """
        Execute a standard ReAct loop using the interface methods.
        
        This is a convenience method that implements the classic ReAct pattern
        using the interface methods. Reasoners can call this directly or use
        it as a reference for their own implementation.
        
        Args:
            goal: The objective to achieve
            max_iterations: Maximum number of iterations
            
        Returns:
            ReasoningResult with final answer and execution metadata
        """
        logger.info(f"Starting standard ReAct loop for goal: {goal}")
        
        observations: List[str] = []
        failed_attempts: List[str] = []
        
        for iteration in range(max_iterations):
            self.increment_iteration()
            logger.info(f"ReAct iteration {iteration + 1}/{max_iterations}")
            
            try:
                # Plan
                context = {
                    "iteration": iteration + 1,
                    "observations": observations,
                    "failed_attempts": failed_attempts,
                    "tool_calls": self.tool_calls,
                }
                plan = self.plan(goal, context)
                logger.debug(f"Plan: {plan}")
                
                # Check if goal is achieved
                if observations and self.evaluate(goal, observations):
                    final_answer = self._generate_final_answer(goal, observations)
                    return self.create_reasoning_result(
                        final_answer=final_answer,
                        success=True
                    )
                
                # Search for tools
                available_tools = self.search_tools(plan, top_k=5)
                
                # Select tool
                selected_tool = self.select_tool(plan, available_tools)
                if not selected_tool:
                    if observations:
                        final_answer = self._generate_final_answer(goal, observations)
                        return self.create_reasoning_result(
                            final_answer=final_answer,
                            success=True
                        )
                    else:
                        failed_attempts.append(f"No suitable tool found for plan: {plan}")
                        continue
                
                # Load tool details
                tool_details = self.load_tool_definition(selected_tool["id"])
                
                # Act
                action_params = self.act(tool_details, plan)
                
                # Execute tool
                execution_result = self.execute_tool_safely(
                    selected_tool["id"], action_params
                )
                
                # Observe
                observation = self.observe(execution_result)
                observations.append(observation)
                
                logger.debug(f"Iteration {iteration + 1} completed successfully")
                
            except Exception as e:
                error_msg = f"Error in iteration {iteration + 1}: {str(e)}"
                logger.error(error_msg)
                failed_attempts.append(error_msg)
                
                # Reflect on failure
                reflection = self.reflect(goal, observations, failed_attempts)
                logger.debug(f"Reflection: {reflection}")
        
        # Max iterations reached
        if observations:
            final_answer = self._generate_final_answer(goal, observations)
            success = True
        else:
            final_answer = "Unable to achieve goal within iteration limit."
            success = False
        
        return self.create_reasoning_result(
            final_answer=final_answer,
            success=success,
            error_message="Max iterations reached" if not success else None
        )
    
    def _generate_final_answer(self, goal: str, observations: List[str]) -> str:
        """
        Generate final answer based on goal and observations.
        
        Args:
            goal: The original goal
            observations: List of observations from the reasoning process
            
        Returns:
            Final answer string
        """
        if not observations:
            return "No observations available to generate final answer."
        
        # Simple default implementation
        return f"Based on the observations, here is the result for '{goal}': {observations[-1]}"