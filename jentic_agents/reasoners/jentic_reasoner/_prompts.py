"""Central location for JenticReasoner prompt templates.

These prompts are intentionally kept minimal at this stage. They will be
iterated as we refine the reasoner's capabilities.
"""

PLAN_GENERATION_PROMPT: str = (
    """You are a planning assistant. Given the user's goal below, output **ONLY** an
    indented markdown bullet list representing a step-by-step plan to achieve it.
    Each bullet must be short and action-oriented.

    Goal:
    {goal}

    Respond with the bullet list only."""
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
    """You are parameter-builder AI. Given the *goal*, the *step*, and the *tool*\n"
    "schema below, produce ONLY a valid JSON object of parameters matching the\n"
    "tool's specification. Do not wrap the JSON in triple-backticks or prose.\n\n"
    "Goal:\n{goal}\n\n"
    "Step:\n{step}\n\n"
    "Tool schema (JSON):\n{tool_schema}\n\n"
    "Respond with the JSON object only."""
)

REFLECTION_PROMPT: str = (
    """You are a self-healing reasoning engine. A step failed. Analyse the error and propose ONE fix.

Return a JSON object with exactly these keys:
  action: one of "retry_params", "change_tool", "rephrase_step", "give_up"
  tool_id: (required if action==change_tool) valid tool id
  params: (required if action in [retry_params, change_tool]) JSON object of parameters
  step:   (required if action==rephrase_step) new step text

If action==give_up, omit the other keys.

Context for you:
Goal: {goal}
Failed Step: {step}
ErrorType: {error_type}
ErrorMessage: {error_message}
ToolSchema: {tool_schema}
"""
)

FINAL_ANSWER_SYNTHESIS_PROMPT: str = (
    """Based on the successful steps and accumulated results, provide a concise
    final answer that satisfies the user's goal."""
)
