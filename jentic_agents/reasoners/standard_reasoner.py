"""
Standard reasoning implementation using ReAct pattern with Jentic SDK integration.
"""

from typing import Any, Dict, List, Optional
import json

from .base_reasoner import BaseReasoner, ReasoningResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StandardReasoner(BaseReasoner):
    """
    Concrete implementation of ReAct reasoning loop with Jentic SDK integration.

    Uses OpenAI for reasoning and Jentic platform for tool discovery and execution.
    """

    def __init__(
        self,
        jentic_client,
        memory,
        llm=None,
        model="gpt-4",
        max_tool_calls_per_iteration=3,
        **kwargs
    ):
        """
        Initialize the standard reasoner.

        Args:
            jentic_client: Client for Jentic platform operations
            memory: Memory system for state persistence
            llm: LLM client for LLM calls (if None, creates default)
            model: OpenAI model to use for reasoning
            max_tool_calls_per_iteration: Max tool calls per reasoning iteration
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(jentic_client, memory, llm, model, **kwargs)
        self.max_tool_calls_per_iteration = max_tool_calls_per_iteration

    def run(self, goal: str, max_iterations: int = 10) -> ReasoningResult:
        """
        Execute the reasoning loop for a given goal.
        """
        logger.info(
            f"Reasoning started for goal: {goal} | Max iterations: {max_iterations}"
        )

        observations: List[str] = []
        failed_attempts: List[str] = []
        
        # Reset state for fresh run
        self.reset_state()

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            try:
                # Increment iteration counter
                self.increment_iteration()
                
                context = {
                    "iteration": self.iteration_count,
                    "observations": observations,
                    "failed_attempts": failed_attempts,
                    "tool_calls": self.tool_calls,
                }

                # Plan
                plan = self.plan(goal, context)
                logger.info(f"Plan: {plan}")

                # Check if we can already answer (only if we have observations)
                if observations and self.evaluate(goal, observations):
                    final_answer = self._generate_final_answer(goal, observations)
                    return self.create_reasoning_result(final_answer, True)

                # Search for tools
                available_tools = self.jentic_client.search(plan, top_k=5)

                # Select tool
                selected_tool = self.select_tool(plan, available_tools)
                if selected_tool:
                    logger.info(f"Tool selected: {selected_tool['id']}")
                else:
                    logger.info("No tool selected for this step.")

                if selected_tool is None:
                    # No tool needed, try to generate answer
                    if observations:
                        final_answer = self._generate_final_answer(goal, observations)
                        return self.create_reasoning_result(final_answer, True)
                    else:
                        failed_attempts.append(
                            f"No suitable tool found for plan: {plan}"
                        )
                        continue

                # Load tool details
                tool_details = self.jentic_client.load(selected_tool["id"])

                # Act
                action_params = self.act(tool_details, plan)

                # Execute tool using base class method
                execution_result = self.execute_tool_safely(
                    selected_tool["id"], action_params
                )

                # Observe
                observation = self.observe(execution_result)
                observations.append(observation)

                logger.info("Step executed and observed.")

            except Exception as e:
                error_msg = f"Error in iteration {iteration + 1}: {str(e)}"
                logger.error(error_msg)
                failed_attempts.append(error_msg)

                # Reflect on failure
                if failed_attempts:
                    reflection = self.reflect(goal, observations, failed_attempts)
                    logger.info("Reflection attempted on failure.")

        # Max iterations reached
        if observations:
            final_answer = self._generate_final_answer(goal, observations)
            return self.create_reasoning_result(final_answer, True)
        else:
            final_answer = "I was unable to find a solution within the iteration limit."
            return self.create_reasoning_result(final_answer, False, "Max iterations reached")

    def plan(self, goal: str, context: Dict[str, Any]) -> str:
        """Generate a plan for achieving the goal."""
        messages = [
            {
                "role": "system",
                "content": """You are a planning assistant. Given a goal and context, create a clear, actionable plan.
                Keep plans concise and focused on the next immediate step needed.""",
            },
            {
                "role": "user",
                "content": f"""Goal: {goal}
                
Context:
- Iteration: {context.get('iteration', 1)}
- Previous observations: {context.get('observations', [])}
- Failed attempts: {context.get('failed_attempts', [])}

What should be the next step in the plan to achieve this goal?""",
            },
        ]

        response = self.safe_llm_call(messages=messages, max_tokens=200, temperature=0.7)

        return response.strip()

    def select_tool(
        self, plan: str, available_tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Delegate to shared helper in BaseReasoner."""
        return self.choose_tool_with_llm(plan, available_tools)

    def act(self, tool: Dict[str, Any], plan: str) -> Dict[str, Any]:
        """Generate parameters for the *tool* using shared helper."""
        return self.generate_and_validate_parameters(tool, plan)

    def observe(self, action_result: Dict[str, Any]) -> str:
        """Process and interpret the result of an action."""
        if action_result.get("status") == "success":
            return f"Tool executed successfully. Result: {action_result.get('result', 'No result provided')}"
        else:
            return f"Tool execution failed. Error: {action_result.get('error', 'Unknown error')}"

    def evaluate(self, goal: str, observations: List[str]) -> bool:
        """Evaluate whether the goal has been achieved based on observations."""
        # Delegate to shared helper in BaseReasoner
        return self.llm_goal_evaluation(goal, observations)

    def reflect(
        self, goal: str, observations: List[str], failed_attempts: List[str]
    ) -> str:
        """Reflect on failures and generate improved strategies."""
        return self.llm_reflect(goal, observations, failed_attempts)

    def _generate_final_answer(self, goal: str, observations: List[str]) -> str:
        """Use shared synthesis helper."""
        return self.generate_final_answer(goal, observations)