"""Central location for JenticReasoner prompt templates.

These prompts are intentionally kept minimal at this stage. They will be
iterated as we refine the reasoner's capabilities.
"""

PLAN_GENERATION_PROMPT: str = (
    """You are a planning assistant. Given the user's goal, produce an indented
    markdown bullet list representing a step-by-step plan. Each bullet should
    be short and action-oriented."""
)

TOOL_SELECTION_PROMPT: str = (
    """Given the current plan step and the list of available tools, return ONLY
    the identifier of the best tool to execute the step, or the word `none` if
    no tool is required."""
)

PARAMETER_GENERATION_PROMPT: str = (
    """Generate a JSON object of parameters for the selected tool so that the
    step can be executed in the context of the given goal."""
)

REFLECTION_PROMPT: str = (
    """The last step failed. Analyze the error, the step text, and the overall
    goal, then suggest a corrected approach or modification."""
)

FINAL_ANSWER_SYNTHESIS_PROMPT: str = (
    """Based on the successful steps and accumulated results, provide a concise
    final answer that satisfies the user's goal."""
)
