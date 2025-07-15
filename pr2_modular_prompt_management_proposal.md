# PR #2: Modular Prompt Management System - Implementation Proposal

## Overview

Transform the hardcoded prompts in `standard_agent` into a flexible, file-based system that allows easy iteration, testing, and sharing of prompts without modifying code.

## Current State Analysis

### Standard Agent (Current)
- All prompts hardcoded in `/jentic_agents/reasoners/rewoo_reasoner/_prompts.py`
- Prompts defined as Python string constants
- Changes require code modifications and redeployment
- Difficult to test prompt variations
- No separation between prompt logic and application code

### ActBots (Reference)
- Prompts stored as separate `.txt` files in `/jentic_agents/prompts/`
- `PromptLoader` utility for loading and caching prompts
- Support for both plain text and JSON prompts
- Clear separation of concerns
- Easy to modify prompts without touching code

## Implementation Plan

### 1. Directory Structure
```
jentic_agents/
├── prompts/                          # New directory for all prompts
│   ├── rewoo/                        # ReWOO-specific prompts
│   │   ├── plan_generation.txt
│   │   ├── tool_selection.txt
│   │   ├── parameter_generation.txt
│   │   ├── base_reflection.txt
│   │   ├── alternative_tools.txt
│   │   ├── final_answer_synthesis.txt
│   │   ├── reasoning_step.txt
│   │   ├── step_classification.txt
│   │   └── json_correction.txt
│   ├── shared/                       # Shared prompts for future use
│   │   └── system_prompt.txt
│   └── README.md                     # Documentation for prompt format
└── utils/
    └── prompt_loader.py              # Utility for loading prompts
```

### 2. PromptLoader Implementation

```python
# jentic_agents/utils/prompt_loader.py
"""Utility for loading prompts from the filesystem with caching and templating support."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import lru_cache

logger = logging.getLogger(__name__)

# Global cache for prompt contents to avoid repeated file I/O
_prompt_cache: Dict[str, Union[str, Dict[str, Any]]] = {}


class PromptLoader:
    """Load and manage prompts from the filesystem."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the prompt loader.
        
        Args:
            base_path: Base path for prompts. Defaults to jentic_agents/prompts
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent / "prompts"
        self.base_path = base_path
        self._local_cache: Dict[str, Any] = {}
    
    def load(self, prompt_path: str, **kwargs) -> str:
        """Load a prompt and optionally format it with kwargs.
        
        Args:
            prompt_path: Path to prompt file relative to base_path (without extension)
                        Can use '/' for subdirectories, e.g., 'rewoo/plan_generation'
            **kwargs: Variables to format into the prompt template
            
        Returns:
            Loaded and formatted prompt string
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
            ValueError: If JSON parsing fails for .json files
        """
        # Check cache first
        cache_key = prompt_path
        if cache_key in self._local_cache and not kwargs:
            return self._local_cache[cache_key]
        
        # Construct full path
        prompt_file = self.base_path / f"{prompt_path}.txt"
        if not prompt_file.exists():
            # Try .json extension
            prompt_file = self.base_path / f"{prompt_path}.json"
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        # Load content
        try:
            content = prompt_file.read_text(encoding='utf-8')
            
            # Parse JSON if needed
            if prompt_file.suffix == '.json':
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, ensure_ascii=False)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON prompt: {prompt_file}")
                    raise ValueError(f"Invalid JSON in prompt file: {e}")
            
            # Cache the raw content
            if not kwargs:
                self._local_cache[cache_key] = content
            
            # Format with kwargs if provided
            if kwargs:
                try:
                    return content.format(**kwargs)
                except KeyError as e:
                    logger.error(f"Missing template variable in prompt {prompt_path}: {e}")
                    raise
            
            return content
            
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_path}: {e}")
            raise
    
    def clear_cache(self):
        """Clear the local cache."""
        self._local_cache.clear()
    
    @staticmethod
    def get_global_loader() -> 'PromptLoader':
        """Get a singleton global prompt loader instance."""
        if not hasattr(PromptLoader, '_global_instance'):
            PromptLoader._global_instance = PromptLoader()
        return PromptLoader._global_instance


# Convenience functions for backward compatibility
def load_prompt(prompt_path: str, **kwargs) -> str:
    """Load a prompt using the global loader.
    
    This is a convenience function that uses a singleton PromptLoader.
    For more control, instantiate PromptLoader directly.
    """
    return PromptLoader.get_global_loader().load(prompt_path, **kwargs)


def clear_prompt_cache():
    """Clear the global prompt cache."""
    PromptLoader.get_global_loader().clear_cache()
    global _prompt_cache
    _prompt_cache.clear()
```

### 3. Prompt File Examples

#### `prompts/rewoo/plan_generation.txt`
```
You are an expert planning assistant.

TASK
• Decompose the *user goal* below into a **markdown bullet-list** plan.

OUTPUT FORMAT
1. Return **only** the fenced list (triple back-ticks) — no prose before or after.
2. Each top-level bullet starts at indent 0 with "- "; sub-steps indent by exactly two spaces.
3. Each bullet = <verb> <object> … followed, in this order, by (input: key_a, key_b) (output: key_c)
   where the parentheses are literal.
4. `output:` key is mandatory when the step's result is needed later; exactly one **snake_case** identifier.
5. `input:` is optional; if present, list comma-separated **snake_case** keys produced by earlier steps.
6. For any step that can fail, add an immediately-indented sibling bullet starting with "→ if fails:" describing a graceful fallback.
7. Do **not** mention specific external tool names.

SELF-CHECK  
After drafting, silently verify — regenerate the list if any check fails:
• All output keys unique & snake_case.  
• All input keys reference existing outputs.  
• Indentation correct (2 spaces per level).  
• No tool names or extra prose outside the fenced block.

EXAMPLE 1 
Task: "Search NYT articles about artificial intelligence and send them to Discord channel 12345"
```
- fetch recent NYT articles mentioning "artificial intelligence" (output: nyt_articles)
  → if fails: report that article search failed.
- send articles as a Discord message to Discord channel 12345 (input: article_list) (output: post_confirmation)
  → if fails: notify the user that posting to Discord failed.
```

EXAMPLE 2 
Task: "Gather the latest 10 Hacker News posts about 'AI', summarise them, and email the summary to alice@example.com"
```
- fetch latest 10 Hacker News posts containing "AI" (output: hn_posts)
  → if fails: report that fetching Hacker News posts failed.
- summarise hn_posts into a concise bullet list (input: hn_posts) (output: summary_text)
  → if fails: report that summarisation failed.
- email summary_text to alice@example.com (input: summary_text) (output: email_confirmation)
  → if fails: notify the user that email delivery failed.
```

REAL GOAL
Goal: {goal}
```
```

#### `prompts/rewoo/tool_selection.txt`
```
You are an expert orchestrator. Given the *step* and the *tools* list below,
return **only** the `id` of the single best tool to execute the step, or
the word `none` if **none of the tools in the provided list are suitable** for the step.

Step:
{step}

Tools (JSON):
{tools_json}

Respond with just the id (e.g. `tool_123`) or `none`. Do not include any other text.
```

### 4. Migration Steps

#### Step 1: Create Directory Structure
```bash
mkdir -p jentic_agents/prompts/rewoo
mkdir -p jentic_agents/prompts/shared
```

#### Step 2: Extract Prompts to Files
- Move each prompt constant from `_prompts.py` to its own `.txt` file
- Preserve exact formatting and template variables

#### Step 3: Update ReWOOReasoner
```python
# jentic_agents/reasoners/rewoo_reasoner/core.py
from jentic_agents.utils.prompt_loader import load_prompt

class ReWOOReasoner(BaseSequentialReasoner):
    def _generate_plan(self, state: ReasonerState) -> None:
        """Generate initial plan from goal using the LLM."""
        # Old way:
        # prompt = prompts.PLAN_GENERATION_PROMPT.replace("{goal}", state.goal)
        
        # New way:
        prompt = load_prompt("rewoo/plan_generation", goal=state.goal)
        plan_md = self._call_llm(prompt)
        self._logger.info(f"phase=PLAN_GENERATED plan={plan_md}")
        state.plan = parse_bullet_plan(plan_md)
    
    def _select_tool(self, step: Step) -> str:
        """Select appropriate tool for the step."""
        tools = self._search_tools(step.text)
        if not tools:
            return "none"
        
        tools_json = json.dumps([t.model_dump() for t in tools], ensure_ascii=False)
        
        # Old way:
        # prompt = prompts.TOOL_SELECTION_PROMPT.replace("{step}", step.text).replace("{tools_json}", tools_json)
        
        # New way:
        prompt = load_prompt("rewoo/tool_selection", step=step.text, tools_json=tools_json)
        return self._call_llm(prompt).strip()
```

#### Step 4: Add Deprecation Notice
```python
# jentic_agents/reasoners/rewoo_reasoner/_prompts.py
"""
DEPRECATED: This module is deprecated as of v0.2.0.
Prompts have been moved to the filesystem under jentic_agents/prompts/rewoo/

For backward compatibility, these constants remain but will be removed in v0.3.0.
Please update your code to use the PromptLoader:

    from jentic_agents.utils.prompt_loader import load_prompt
    prompt = load_prompt("rewoo/plan_generation", goal=goal)
"""
import warnings
from jentic_agents.utils.prompt_loader import load_prompt

warnings.warn(
    "The _prompts module is deprecated. Use PromptLoader instead.",
    DeprecationWarning,
    stacklevel=2
)

# Maintain backward compatibility
PLAN_GENERATION_PROMPT = load_prompt("rewoo/plan_generation")
TOOL_SELECTION_PROMPT = load_prompt("rewoo/tool_selection")
# ... etc
```

### 5. Testing Strategy

#### Unit Tests
```python
# tests/test_prompt_loader.py
import pytest
from pathlib import Path
from jentic_agents.utils.prompt_loader import PromptLoader, load_prompt

def test_load_simple_prompt(tmp_path):
    """Test loading a simple prompt."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    
    test_prompt = "Hello {name}!"
    (prompt_dir / "greeting.txt").write_text(test_prompt)
    
    loader = PromptLoader(prompt_dir)
    assert loader.load("greeting", name="World") == "Hello World!"

def test_load_nested_prompt(tmp_path):
    """Test loading prompts from subdirectories."""
    prompt_dir = tmp_path / "prompts" / "rewoo"
    prompt_dir.mkdir(parents=True)
    
    test_prompt = "Plan for: {goal}"
    (prompt_dir / "plan.txt").write_text(test_prompt)
    
    loader = PromptLoader(tmp_path / "prompts")
    assert loader.load("rewoo/plan", goal="test") == "Plan for: test"

def test_prompt_not_found():
    """Test error handling for missing prompts."""
    loader = PromptLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent/prompt")

def test_cache_behavior(tmp_path):
    """Test that prompts are cached correctly."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    
    prompt_file = prompt_dir / "cached.txt"
    prompt_file.write_text("Original")
    
    loader = PromptLoader(prompt_dir)
    assert loader.load("cached") == "Original"
    
    # Modify file
    prompt_file.write_text("Modified")
    
    # Should still return cached version
    assert loader.load("cached") == "Original"
    
    # Clear cache and reload
    loader.clear_cache()
    assert loader.load("cached") == "Modified"
```

#### Integration Tests
```python
# tests/test_rewoo_with_prompts.py
def test_rewoo_uses_prompt_loader(mocker):
    """Ensure ReWOOReasoner uses PromptLoader correctly."""
    mock_load = mocker.patch('jentic_agents.utils.prompt_loader.load_prompt')
    mock_load.return_value = "mocked prompt"
    
    reasoner = ReWOOReasoner(tool=mock_tool, memory=mock_memory, llm=mock_llm)
    state = ReasonerState(goal="test goal")
    
    reasoner._generate_plan(state)
    
    mock_load.assert_called_with("rewoo/plan_generation", goal="test goal")
```

### 6. Documentation

#### prompts/README.md
```markdown
# Prompt Management Guide

This directory contains all prompts used by the Jentic Agents system.

## Directory Structure

- `rewoo/` - Prompts specific to the ReWOO reasoner
- `shared/` - Shared prompts used across multiple components
- Future reasoners will have their own subdirectories

## Prompt Format

### Basic Text Prompts
Most prompts are plain text files with Python string formatting placeholders:

```
Hello {name}, welcome to {place}!
```

### JSON Prompts
Some prompts may be JSON files for structured data:

```json
{
  "system": "You are a helpful assistant",
  "examples": [
    {"input": "Hi", "output": "Hello!"}
  ]
}
```

## Using Prompts in Code

```python
from jentic_agents.utils.prompt_loader import load_prompt

# Load and format a prompt
prompt = load_prompt("rewoo/plan_generation", goal="Build a website")

# Load a shared prompt
system_prompt = load_prompt("shared/system_prompt")
```

## Adding New Prompts

1. Create a `.txt` or `.json` file in the appropriate directory
2. Use `{variable_name}` for template variables
3. Document the prompt's purpose and required variables
4. Update tests to ensure the prompt loads correctly

## Best Practices

1. Keep prompts focused and single-purpose
2. Use clear variable names that match the domain
3. Include examples in complex prompts
4. Version prompts by copying to new files (e.g., `plan_generation_v2.txt`)
5. Test prompts thoroughly before deployment
```

### 7. Benefits

1. **Separation of Concerns**: Prompts separated from code logic
2. **Easy Iteration**: Modify prompts without touching Python code
3. **Version Control**: Track prompt changes independently
4. **A/B Testing**: Easy to test different prompt versions
5. **Collaboration**: Non-programmers can modify prompts
6. **Reusability**: Share prompts between different components
7. **Hot Reloading**: Could add file watching for development

### 8. Migration Path

1. **Phase 1**: Implement PromptLoader and create directory structure
2. **Phase 2**: Extract all prompts to files
3. **Phase 3**: Update ReWOOReasoner to use PromptLoader
4. **Phase 4**: Add deprecation warnings to _prompts.py
5. **Phase 5**: Update tests and documentation
6. **Phase 6**: Remove _prompts.py in next major version

### 9. Future Enhancements

1. **Prompt Versioning**: Support for loading specific versions
2. **Hot Reloading**: Watch files and reload on change in dev mode
3. **Prompt Validation**: Schema validation for required variables
4. **Prompt Composition**: Combine multiple prompts
5. **Environment-Specific Prompts**: Load different prompts per environment
6. **Prompt Analytics**: Track which prompts are used most

## Conclusion

This modular prompt management system provides a clean foundation for prompt engineering and experimentation. It maintains backward compatibility while enabling powerful new workflows for prompt development and testing.