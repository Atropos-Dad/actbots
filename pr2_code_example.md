# PR #2: Code Example - Before and After

## Example: Updating ReWOOReasoner Methods

### Before (Current Implementation)

```python
# jentic_agents/reasoners/rewoo_reasoner/core.py
import jentic_agents.reasoners.rewoo_reasoner._prompts as prompts

class ReWOOReasoner(BaseSequentialReasoner):
    
    def _generate_plan(self, state: ReasonerState) -> None:
        """Generate initial plan from goal using the LLM."""
        prompt = prompts.PLAN_GENERATION_PROMPT.replace("{goal}", state.goal)
        plan_md = self._call_llm(prompt)
        self._logger.info(f"phase=PLAN_GENERATED plan={plan_md}")
        state.plan = parse_bullet_plan(plan_md)
    
    def _select_tool(self, step: Step) -> str:
        """Select the best tool for executing the step."""
        tools = self._search_tools(step.text)
        if not tools:
            return "none"
        
        tools_json = json.dumps([t.model_dump() for t in tools], ensure_ascii=False)
        prompt = prompts.TOOL_SELECTION_PROMPT.replace("{step}", step.text).replace("{tools_json}", tools_json)
        
        tool_id = self._call_llm(prompt).strip()
        return tool_id
    
    def _generate_params(self, step: Step, tool_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for the selected tool."""
        tool = self._get_tool(tool_id)
        allowed_keys = ", ".join(f'"{k}"' for k in tool.input_schema["properties"].keys())
        tool_schema = json.dumps(tool.input_schema, ensure_ascii=False, indent=2)
        step_inputs = json.dumps(inputs, ensure_ascii=False, indent=2)
        
        prompt = prompts.PARAMETER_GENERATION_PROMPT.format(
            allowed_keys=allowed_keys,
            step=step.text,
            step_inputs=step_inputs,
            tool_schema=tool_schema
        )
        
        params_json = self._call_llm(prompt)
        return self._parse_json_params(params_json, step, tool_id)
```

### After (With PromptLoader)

```python
# jentic_agents/reasoners/rewoo_reasoner/core.py
from jentic_agents.utils.prompt_loader import load_prompt

class ReWOOReasoner(BaseSequentialReasoner):
    
    def _generate_plan(self, state: ReasonerState) -> None:
        """Generate initial plan from goal using the LLM."""
        prompt = load_prompt("rewoo/plan_generation", goal=state.goal)
        plan_md = self._call_llm(prompt)
        self._logger.info(f"phase=PLAN_GENERATED plan={plan_md}")
        state.plan = parse_bullet_plan(plan_md)
    
    def _select_tool(self, step: Step) -> str:
        """Select the best tool for executing the step."""
        tools = self._search_tools(step.text)
        if not tools:
            return "none"
        
        tools_json = json.dumps([t.model_dump() for t in tools], ensure_ascii=False)
        prompt = load_prompt("rewoo/tool_selection", step=step.text, tools_json=tools_json)
        
        tool_id = self._call_llm(prompt).strip()
        return tool_id
    
    def _generate_params(self, step: Step, tool_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for the selected tool."""
        tool = self._get_tool(tool_id)
        allowed_keys = ", ".join(f'"{k}"' for k in tool.input_schema["properties"].keys())
        tool_schema = json.dumps(tool.input_schema, ensure_ascii=False, indent=2)
        step_inputs = json.dumps(inputs, ensure_ascii=False, indent=2)
        
        prompt = load_prompt(
            "rewoo/parameter_generation",
            allowed_keys=allowed_keys,
            step=step.text,
            step_inputs=step_inputs,
            tool_schema=tool_schema
        )
        
        params_json = self._call_llm(prompt)
        return self._parse_json_params(params_json, step, tool_id)
```

## Key Differences

1. **Import Change**: 
   - Before: `import jentic_agents.reasoners.rewoo_reasoner._prompts as prompts`
   - After: `from jentic_agents.utils.prompt_loader import load_prompt`

2. **Prompt Loading**:
   - Before: `prompts.CONSTANT_NAME` with manual `.replace()` or `.format()`
   - After: `load_prompt("path/to/prompt", **kwargs)` with automatic formatting

3. **Benefits**:
   - Cleaner code (no manual string replacement)
   - Type-safe keyword arguments
   - Prompts can be edited without redeploying code
   - Better error messages for missing template variables

## Backward Compatibility

For teams that have custom code depending on the old constants:

```python
# jentic_agents/reasoners/rewoo_reasoner/_prompts.py
import warnings
from jentic_agents.utils.prompt_loader import load_prompt

warnings.warn(
    "The _prompts module is deprecated. Use load_prompt() instead.",
    DeprecationWarning,
    stacklevel=2
)

# Lazy-load prompts for backward compatibility
@property
def PLAN_GENERATION_PROMPT():
    return load_prompt("rewoo/plan_generation")

# Or simpler but less efficient:
PLAN_GENERATION_PROMPT = load_prompt("rewoo/plan_generation")
TOOL_SELECTION_PROMPT = load_prompt("rewoo/tool_selection")
# ... etc
```

This ensures existing code continues to work while encouraging migration to the new system.