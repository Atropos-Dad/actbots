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
    """You are an expert orchestrator. Given the *step* and the *tools* list below,\n"
    "return **only** the `id` of the single best tool to execute the step, or\n"
    "the word `none` if no tool is required.\n\n"
    "Step:\n{step}\n\n"
    "Tools (JSON):\n{tools_json}\n\n"
    "Respond with just the id (e.g. `tool_123`) or `none`. Do not include any other text."""
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
