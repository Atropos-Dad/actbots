"""
Standard reasoning implementation using ReAct pattern with Jentic SDK integration.
"""

import logging
from typing import Any, Dict, List, Optional
import json

from .base_reasoner import BaseReasoner, ReasoningResult
from ..utils.llm import BaseLLM, LiteLLMChatLLM
from ..memory.base_memory import BaseMemory

logger = logging.getLogger(__name__)


class StandardReasoner(BaseReasoner):
    """
    Concrete implementation of ReAct reasoning loop with Jentic SDK integration.

    Uses OpenAI for reasoning and Jentic platform for tool discovery and execution.
    """

    def __init__(
        self,
        jentic_client: "JenticClient",
        memory: BaseMemory,
        llm: Optional[BaseLLM] = None,
        model: str = "gpt-4",
        max_tool_calls_per_iteration: int = 3,
    ):
        """
        Initialize the standard reasoner.

        Args:
            jentic_client: Client for Jentic platform operations
            memory: Memory system for the agent
            llm: LLM client for LLM calls (if None, creates default)
            model: OpenAI model to use for reasoning
            max_tool_calls_per_iteration: Max tool calls per reasoning iteration
        """

        self.jentic_client = jentic_client
        self.memory = memory
        self.llm = llm or LiteLLMChatLLM(model=model)
        self.model = model
        self.max_tool_calls_per_iteration = max_tool_calls_per_iteration

    def run(self, goal: str, max_iterations: int = 10) -> ReasoningResult:
        """
        Execute the reasoning loop for a given goal.
        """
        logger.info(
            f"Reasoning started for goal: {goal} | Max iterations: {max_iterations}"
        )

        observations: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        failed_attempts: List[str] = []

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            try:
                # Increment iteration counter
                self.increment_iteration()
                
                context = {
                    "iteration": self.iteration_count,
                    "observations": observations,
                    "failed_attempts": failed_attempts,
                    "tool_calls": tool_calls,
                }

                # Plan
                plan = self.plan(goal, context)
                logger.info(f"Plan: {plan}")

                # Check if we can already answer (only if we have observations)
                if observations and self.evaluate(goal, observations):
                    final_answer = self._generate_final_answer(goal, observations)
                    return ReasoningResult(
                        final_answer=final_answer,
                        iterations=iteration + 1,
                        tool_calls=tool_calls,
                        success=True,
                    )

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
                        return ReasoningResult(
                            final_answer=final_answer,
                            iterations=iteration + 1,
                            tool_calls=tool_calls,
                            success=True,
                        )
                    else:
                        failed_attempts.append(
                            f"No suitable tool found for plan: {plan}"
                        )
                        continue

                # Load tool details
                tool_details = self.jentic_client.load(selected_tool["id"])

                # Act
                action_params = self.act(tool_details, plan)

                # Execute tool
                execution_result = self.jentic_client.execute(
                    selected_tool["id"], action_params
                )

                tool_calls.append(
                    {
                        "tool_id": selected_tool["id"],
                        "tool_name": selected_tool["name"],
                        "params": action_params,
                        "result": execution_result,
                    }
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
            success = True
        else:
            final_answer = "I was unable to find a solution within the iteration limit."
            success = False

        return ReasoningResult(
            final_answer=final_answer,
            iterations=max_iterations,
            tool_calls=tool_calls,
            success=success,
            error_message="Max iterations reached" if not success else None,
        )

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

        response = self.llm.chat(messages=messages, max_tokens=200, temperature=0.7)

        return response.strip()

    def select_tool(
        self, plan: str, available_tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select the most appropriate tool for executing the current plan."""
        if not available_tools:
            return None

        if len(available_tools) == 1:
            return available_tools[0]

        tool_descriptions = "\n".join(
            [
                f"- {tool['id']}: {tool['name']} - {tool.get('description', '')}"
                for tool in available_tools
            ]
        )

        messages = [
            {
                "role": "system",
                "content": f"""You are a tool selection expert. Your job is to select the single best tool to execute the given plan.
Respond with ONLY the ID of the selected tool, or NONE if no tool is suitable.

Available tools:
{tool_descriptions}
""",
            },
            {"role": "user", "content": f"Plan: {plan}"},
        ]

        response = self.llm.chat(messages=messages, max_tokens=50, temperature=0.0)

        # Find the tool that matches the response
        selected_id = response.strip()
        if selected_id == "NONE":
            return None
        return next((t for t in available_tools if t["id"] == selected_id), None)

    def act(self, tool: Dict[str, Any], plan: str) -> Dict[str, Any]:
        """Generate parameters for the selected tool."""
        if not tool.get("parameters"):
            return {}

        messages = [
            {
                "role": "system",
                "content": f"""You are a parameter generation expert. Given a tool and a plan, generate the correct parameters in JSON format.
Tool: {tool['name']}
Tool description: {tool.get('description', '')}
Tool parameters schema: {json.dumps(tool['parameters'])}
""",
            },
            {
                "role": "user",
                "content": f"Plan: {plan}\n\nGenerate the JSON parameters for this tool based on the plan.",
            },
        ]

        response = self.llm.chat(messages=messages, max_tokens=500, temperature=0.0)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(
                f"Invalid JSON response for tool {tool['name']}: {response}"
            )
            return {}

    def observe(self, action_result: Dict[str, Any]) -> str:
        """Process and interpret the result of an action."""
        if action_result.get("status") == "success":
            return f"Tool executed successfully. Result: {action_result.get('result', 'No result provided')}"
        else:
            return f"Tool execution failed. Error: {action_result.get('error', 'Unknown error')}"

    def evaluate(self, goal: str, observations: List[str]) -> bool:
        """Evaluate whether the goal has been achieved based on observations."""
        messages = [
            {
                "role": "system",
                "content": """You are an evaluation expert. Your job is to determine if the goal has been achieved based on the observations.
Respond with only YES or NO.""",
            },
            {
                "role": "user",
                "content": f"""Goal: {goal}

Observations:
- {"\n- ".join(observations)}

Based on these observations, has the goal been achieved? Respond with only YES or NO.""",
            },
        ]
        response = self.llm.chat(messages=messages, max_tokens=10, temperature=0.0)
        return response.strip().upper() == "YES"

    def reflect(
        self, goal: str, observations: List[str], failed_attempts: List[str]
    ) -> str:
        """Reflect on failures and generate improved strategies."""
        messages = [
            {
                "role": "system",
                "content": """You are a reflection expert. Your job is to analyze failed attempts and provide a reflection to improve the next plan.""",
            },
            {
                "role": "user",
                "content": f"""Goal: {goal}

Observations:
- {"\n- ".join(observations)}

Failed Attempts:
- {"\n- ".join(failed_attempts)}

Provide a reflection on why the attempts failed and how to improve the approach.""",
            },
        ]

        response = self.llm.chat(messages=messages, max_tokens=300, temperature=0.7)

        return response.strip()

    def _generate_final_answer(self, goal: str, observations: List[str]) -> str:
        """Generate the final answer based on observations."""
        messages = [
            {
                "role": "system",
                "content": """You are a final answer generation expert. Your job is to provide a comprehensive answer to the original goal based on the observations.""",
            },
            {
                "role": "user",
                "content": f"""Goal: {goal}

Observations:
- {"\n- ".join(observations)}

Provide the final answer to the goal based on these observations.""",
            },
        ]

        response = self.llm.chat(messages=messages, max_tokens=1000, temperature=0.7)

        return response.strip()
