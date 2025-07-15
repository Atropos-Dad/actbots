# PR #2: Quick Implementation Guide

## 🎯 Goal
Extract hardcoded prompts from `_prompts.py` into a flexible file-based system.

## 📁 File Changes Overview

### New Files to Create:
1. `jentic_agents/utils/prompt_loader.py` - The prompt loading utility
2. `jentic_agents/prompts/rewoo/*.txt` - 9 prompt files
3. `jentic_agents/prompts/README.md` - Documentation
4. `tests/test_prompt_loader.py` - Unit tests

### Files to Modify:
1. `jentic_agents/reasoners/rewoo_reasoner/core.py` - Use PromptLoader
2. `jentic_agents/reasoners/rewoo_reasoner/_prompts.py` - Add deprecation

## 🚀 Step-by-Step Implementation

### Step 1: Create the Directory Structure
```bash
cd /path/to/standard_agent
mkdir -p jentic_agents/prompts/rewoo
mkdir -p jentic_agents/prompts/shared
touch jentic_agents/utils/prompt_loader.py
```

### Step 2: Implement PromptLoader
Copy the PromptLoader implementation from the proposal into `jentic_agents/utils/prompt_loader.py`.

### Step 3: Extract Prompts to Files

Create these files in `jentic_agents/prompts/rewoo/`:

1. **plan_generation.txt** - Extract `PLAN_GENERATION_PROMPT`
2. **tool_selection.txt** - Extract `TOOL_SELECTION_PROMPT`
3. **parameter_generation.txt** - Extract `PARAMETER_GENERATION_PROMPT`
4. **base_reflection.txt** - Extract `BASE_REFLECTION_PROMPT`
5. **alternative_tools.txt** - Extract `ALTERNATIVE_TOOLS_SECTION`
6. **final_answer_synthesis.txt** - Extract `FINAL_ANSWER_SYNTHESIS_PROMPT`
7. **reasoning_step.txt** - Extract `REASONING_STEP_PROMPT`
8. **step_classification.txt** - Extract `STEP_CLASSIFICATION_PROMPT`
9. **json_correction.txt** - Extract `JSON_CORRECTION_PROMPT`

**Important**: Remove the triple quotes and keep only the prompt content!

### Step 4: Update ReWOOReasoner

In `core.py`, add import:
```python
from jentic_agents.utils.prompt_loader import load_prompt
```

Then update each prompt usage:

```python
# Before:
prompt = prompts.PLAN_GENERATION_PROMPT.replace("{goal}", state.goal)

# After:
prompt = load_prompt("rewoo/plan_generation", goal=state.goal)
```

### Step 5: Update All Prompt References

Search and replace patterns:
- `prompts.PLAN_GENERATION_PROMPT` → `load_prompt("rewoo/plan_generation")`
- `prompts.TOOL_SELECTION_PROMPT` → `load_prompt("rewoo/tool_selection")`
- etc.

### Step 6: Add Tests

Create `tests/test_prompt_loader.py` with the tests from the proposal.

### Step 7: Add Deprecation Warning

Update `_prompts.py` to add deprecation notice at the top and maintain backward compatibility.

## 📋 PR Checklist

- [ ] Create prompt directory structure
- [ ] Implement PromptLoader class
- [ ] Extract all 9 prompts to text files
- [ ] Update ReWOOReasoner to use PromptLoader
- [ ] Add deprecation warning to _prompts.py
- [ ] Create unit tests for PromptLoader
- [ ] Add integration test for ReWOOReasoner
- [ ] Create prompts/README.md documentation
- [ ] Update main README with prompt management info
- [ ] Test that existing functionality still works
- [ ] Verify prompts are loaded correctly
- [ ] Check that caching works as expected

## 🧪 Testing Commands

```bash
# Run tests
pytest tests/test_prompt_loader.py -v

# Test ReWOOReasoner still works
python main.py

# Verify deprecation warning
python -c "import jentic_agents.reasoners.rewoo_reasoner._prompts"
```

## 📝 Commit Message

```
feat: implement modular prompt management system

- Add PromptLoader utility for file-based prompts
- Extract ReWOO prompts to separate text files
- Maintain backward compatibility with deprecation warnings
- Add comprehensive tests and documentation
- Enable easy prompt iteration without code changes

This change separates prompts from code, making it easier to
experiment with different prompts and collaborate with non-developers.
```

## ⚠️ Common Pitfalls

1. **Don't include triple quotes** in the prompt files
2. **Preserve exact formatting** including newlines and indentation
3. **Keep template variables** like `{goal}` exactly as they are
4. **Test each prompt** after extraction to ensure it works
5. **Clear cache** during testing if prompts seem stuck

## 🎉 Success Criteria

- All existing tests pass
- New prompt loader tests pass
- ReWOOReasoner works identically to before
- Prompts can be edited without touching Python code
- Deprecation warnings appear when using old module