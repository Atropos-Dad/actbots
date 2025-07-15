# Cursor Implementation Prompt

## Task: Upgrade ReWOOReasoner with Universal Agent Prompts

I need you to upgrade the ReWOOReasoner in the standard_agent repository to use the enhanced prompts from `prompts_tryout.py`. This will add human-in-the-loop capabilities, better planning, and improved error handling.

## Files to Modify:

### 1. `/jentic_agents/reasoners/rewoo_reasoner/core.py`

**Changes needed:**
1. Import the UniversalAgentPrompts class
2. Replace all prompt usage with the new prompts
3. Add support for HUMAN step type in planning
4. Enhance reflection to support more recovery strategies
5. Add context analysis before tool selection

### 2. `/jentic_agents/reasoners/models.py`

**Changes needed:**
1. Add HUMAN to StepType enum if not present
2. Add optional fields to Step model: `dependencies`, `store_key`

### 3. `/jentic_agents/reasoners/rewoo_reasoner/_parser.py`

**Changes needed:**
1. Update the parser to handle JSON-formatted plans
2. Support new step attributes (dependencies, store_key)

## Implementation Instructions:

### Step 1: Add the prompts file
Create a new file `/jentic_agents/reasoners/rewoo_reasoner/_universal_prompts.py` and paste the entire content from `prompts_tryout.py`.

### Step 2: Update core.py imports
```python
from jentic_agents.reasoners.rewoo_reasoner._universal_prompts import UniversalAgentPrompts
```

### Step 3: Initialize prompts in ReWOOReasoner
```python
class ReWOOReasoner(BaseSequentialReasoner):
    def __init__(self, *, tool: ToolInterface, memory: BaseMemory, llm: BaseLLM) -> None:
        super().__init__(tool=tool, memory=memory, llm=llm)
        self._tool_cache: Dict[str, Tool] = {}
        self.prompts = UniversalAgentPrompts()
```

### Step 4: Update _generate_plan method
Replace the current implementation with:
```python
def _generate_plan(self, state: ReasonerState) -> None:
    """Generate initial plan from goal using enhanced prompts."""
    prompt = self.prompts.format_prompt(
        self.prompts.PLAN_GENERATION_PROMPT,
        goal=state.goal
    )
    plan_response = self._call_llm(prompt)
    
    # Extract JSON from response
    plan_json = self.prompts.get_json_from_prompt_response(plan_response)
    if not plan_json:
        # Fallback to original parser if JSON extraction fails
        self._logger.warning("Failed to extract JSON plan, using legacy parser")
        state.plan = parse_bullet_plan(plan_response)
    else:
        # Convert JSON plan to Step objects
        state.plan = self._parse_json_plan(plan_json)
    
    self._logger.info(f"phase=PLAN_GENERATED plan={state.plan}")
```

### Step 5: Add JSON plan parser
```python
def _parse_json_plan(self, plan_json: List[Dict[str, Any]]) -> List[Step]:
    """Parse JSON plan into Step objects."""
    steps = []
    for step_data in plan_json:
        step = Step(
            text=step_data.get("text", ""),
            step_type=Step.StepType[step_data.get("step_type", "TOOL")]
        )
        # Add optional attributes
        if "store_key" in step_data:
            step.output_key = step_data["store_key"]
        if "dependencies" in step_data:
            step.dependencies = step_data["dependencies"]
        steps.append(step)
    return steps
```

### Step 6: Update _select_tool method
```python
def _select_tool(self, step: Step) -> str:
    """Select tool with context awareness."""
    # Perform context analysis first
    context_prompt = self.prompts.format_prompt(
        self.prompts.CONTEXT_ANALYSIS_PROMPT,
        goal=self.memory.get("_goal", ""),
        current_step=step.text,
        previous_steps=str([s.text for s in self.memory.get("_completed_steps", [])]),
        memory_summary=str(list(self.memory.keys()))
    )
    context_analysis = self._call_llm(context_prompt)
    
    # Now select tool with context
    tools = self._search_tools(step.text)
    if not tools:
        return "none"
    
    tools_json = json.dumps([t.model_dump() for t in tools], ensure_ascii=False)
    prompt = self.prompts.format_prompt(
        self.prompts.TOOL_SELECTION_PROMPT,
        step=step.text,
        tools_json=tools_json,
        memory_keys=str(list(self.memory.keys())),
        previous_actions=str(self.memory.get("_action_history", []))
    )
    
    tool_id = self._call_llm(prompt).strip()
    return tool_id
```

### Step 7: Enhanced reflection
Update the reflection logic to use the new reflection prompt with more strategies:
```python
def _reflect_on_failure(self, error: Exception, step: Step, state: ReasonerState, failed_tool_id: Optional[str] = None) -> None:
    """Enhanced reflection with multiple recovery strategies."""
    prompt = self.prompts.format_prompt(
        self.prompts.REFLECTION_PROMPT,
        goal=state.goal,
        step=step.text,
        error_type=type(error).__name__,
        error_message=str(error),
        attempt_number=step.reflection_attempts + 1,
        failed_tool_id=failed_tool_id or "none",
        memory_keys=str(list(self.memory.keys())),
        recent_memory=str(dict(list(self.memory.items())[-5:])),
        previous_attempts=str(step.reflection_history) if hasattr(step, 'reflection_history') else "[]"
    )
    
    reflection_response = self._call_llm(prompt)
    reflection_json = self.prompts.get_json_from_prompt_response(reflection_response)
    
    if not reflection_json:
        # Fallback to original reflection logic
        return super()._reflect_on_failure(error, step, state, failed_tool_id)
    
    # Handle different recovery strategies
    action = reflection_json.get("action")
    details = reflection_json.get("details", {})
    
    if action == "retry_params":
        # Update step with new parameters
        step.forced_params = details.get("params")
        step.status = "pending"
        
    elif action == "change_tool":
        # Force a different tool
        step.forced_tool_id = details.get("tool_id")
        step.forced_params = details.get("params")
        step.status = "pending"
        
    elif action == "rephrase_step":
        # Update step text
        step.text = details.get("step", step.text)
        step.status = "pending"
        
    elif action == "escalate_human":
        # Create human escalation
        escalation = self.prompts.create_human_escalation_prompt(
            context=details.get("context", ""),
            specific_need=details.get("question", "")
        )
        self._logger.warning(f"HUMAN ESCALATION: {escalation}")
        # You might want to handle this differently based on your system
        raise HumanInterventionRequired(escalation)
        
    elif action == "decompose_step":
        # Replace step with sub-steps
        sub_steps = details.get("sub_steps", [])
        if sub_steps:
            # Insert new steps into the plan
            idx = state.plan.index(step)
            state.plan.pop(idx)
            for i, sub_text in enumerate(sub_steps):
                new_step = Step(text=sub_text, step_type=step.step_type)
                state.plan.insert(idx + i, new_step)
```

### Step 8: Handle HUMAN steps
Add logic to handle HUMAN step type:
```python
def _execute_step(self, step: Step, state: ReasonerState) -> Optional[Dict[str, Any]]:
    """Execute a step with support for HUMAN type."""
    step.status = "running"
    
    # Handle HUMAN steps
    if step.step_type == Step.StepType.HUMAN:
        escalation = self.prompts.create_human_escalation_prompt(
            context=f"Goal: {state.goal}",
            specific_need=step.text
        )
        self._logger.info(f"HUMAN INTERVENTION REQUIRED: {escalation}")
        # Store the request in memory for later reference
        self.memory[f"human_response_{len(state.completed_steps)}"] = {
            "request": step.text,
            "response": "Awaiting human input..."
        }
        step.status = "done"
        return None
    
    # Rest of the original logic...
    return super()._execute_step(step, state)
```

### Step 9: Update parameter generation
Use the enhanced parameter generation prompt:
```python
def _generate_params(self, step: Step, tool_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate parameters with enhanced validation."""
    tool = self._get_tool(tool_id)
    
    # Extract required parameters
    required_params = tool.input_schema.get("required", [])
    
    prompt = self.prompts.format_prompt(
        self.prompts.PARAMETER_GENERATION_PROMPT,
        tool_name=tool.name,
        tool_schema=json.dumps(tool.input_schema, ensure_ascii=False, indent=2),
        required_params=json.dumps(required_params),
        step=step.text,
        memory_data=json.dumps(inputs, ensure_ascii=False, indent=2),
        previous_results=json.dumps(list(self.memory.keys()))
    )
    
    params_json = self._call_llm(prompt)
    params = self._parse_json_params(params_json, step, tool_id)
    
    # Check for NEED_HUMAN_INPUT
    for key, value in params.items():
        if isinstance(value, str) and "NEED_HUMAN_INPUT" in value:
            raise HumanInterventionRequired(f"Missing required parameter: {key}")
    
    return params
```

### Step 10: Add HumanInterventionRequired exception
Create a new exception class:
```python
class HumanInterventionRequired(Exception):
    """Raised when human intervention is needed."""
    pass
```

## Testing the Implementation:

1. Test with a simple goal: "Send 'Hello World' to Discord channel 123456"
2. Test with a complex goal: "Find the most active GitHub repository and create an issue about improving documentation"
3. Test with an ambiguous goal: "Deploy the feature" (should trigger human escalation)
4. Test error recovery: Provide invalid parameters and see if it self-corrects

## Key Improvements Added:

1. **Human-in-the-Loop**: Agent can now ask for help when needed
2. **Better Planning**: JSON-structured plans with dependencies
3. **Context Awareness**: Analyzes context before tool selection
4. **Enhanced Reflection**: Multiple recovery strategies
5. **Smarter Parameters**: Better parameter generation with validation
6. **Step Decomposition**: Can break complex steps into simpler ones

Make sure to preserve all existing functionality while adding these enhancements. The agent should be more capable but still work with existing code.