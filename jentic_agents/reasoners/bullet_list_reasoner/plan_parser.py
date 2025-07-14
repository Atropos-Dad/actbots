"""Plan parsing logic for BulletPlanReasoner."""

import json
import re
from collections import deque
from typing import Deque, List

from ...utils.parsing_helpers import strip_backtick_fences
from ...utils.logger import get_logger
from .reasoner_state import Step

logger = get_logger(__name__)

# Regex pattern for bullet parsing
BULLET_RE = re.compile(r"^(?P<indent>\s*)([-*]|\d+\.)\s+(?P<content>.+)$")


class BulletPlanParser:
    """Parser for converting markdown bullet lists and JSON arrays into Step objects."""

    def parse(self, markdown: str) -> Deque[Step]:
        """Parse a bullet plan from markdown or JSON format."""
        markdown_stripped = strip_backtick_fences(markdown)
        
        # Check if content is JSON array
        if markdown_stripped.startswith("[") and markdown_stripped.endswith("]"):
            return self._parse_json_plan(markdown_stripped)
        else:
            return self._parse_markdown_plan(markdown_stripped)

    def _parse_json_plan(self, json_content: str) -> Deque[Step]:
        """Parse plan from JSON array format."""
        try:
            logger.info("Parsing plan as JSON array")
            json_steps = json.loads(json_content)
            steps = []
            
            for step_data in json_steps:
                if isinstance(step_data, dict):
                    text = step_data.get("text", "")
                    step_type = step_data.get("step_type", "")
                    store_key = step_data.get("store_key")
                    load_from_store = step_data.get("load_from_store")
                    
                    # Extract goal context from parentheses if present
                    goal_context = self._extract_goal_context(text)
                    text = self._remove_goal_context(text)
                    
                    step = Step(
                        text=text,
                        indent=0,  # JSON format doesn't use indentation
                        store_key=store_key,
                        goal_context=goal_context,
                        step_type=step_type,
                        load_from_store=load_from_store,
                    )
                    steps.append(step)
            
            logger.info(f"Parsed {len(steps)} steps from plan (JSON mode)")
            return deque(steps)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse as JSON: {e}, falling back to markdown parsing")
            return self._parse_markdown_plan(json_content)

    def _parse_markdown_plan(self, markdown: str) -> Deque[Step]:
        """Parse plan from markdown bullet format."""
        steps: List[Step] = []
        
        for line_num, line in enumerate(markdown.splitlines(), 1):
            if not line.strip():
                continue  # skip blank lines
                
            match = BULLET_RE.match(line)
            if not match:
                logger.debug(f"Line {line_num} doesn't match bullet pattern: {line}")
                continue
                
            indent_spaces = len(match.group("indent"))
            indent_level = indent_spaces // 2  # assume two-space indents
            content = match.group("content").strip()

            # Extract goal context and store key
            goal_context = self._extract_goal_context(content)
            content = self._remove_goal_context(content)
            store_key = self._extract_store_key(content)
            content = self._remove_store_directive(content)

            step = Step(
                text=content,
                indent=indent_level,
                store_key=store_key,
                goal_context=goal_context,
            )
            steps.append(step)
            
            logger.debug(
                f"Parsed step: text='{step.text}', goal_context='{step.goal_context}', store_key='{step.store_key}'"
            )

        # Filter to leaf steps only (exclude parent/container bullets)
        leaf_steps = self._extract_leaf_steps(steps)
        
        logger.info(f"Parsed {len(leaf_steps)} leaf steps from plan (original {len(steps)})")
        return deque(leaf_steps)

    def _extract_goal_context(self, content: str) -> str:
        """Extract goal context from parentheses: '... ( goal: actual goal text )'"""
        goal_match = re.search(r"\(\s*goal:\s*([^)]+)\s*\)", content)
        if goal_match:
            goal_context = goal_match.group(1).strip()
            logger.debug(f"Extracted goal context: {goal_context}")
            return goal_context
        return None

    def _remove_goal_context(self, content: str) -> str:
        """Remove goal context from content."""
        return re.sub(r"\s*\(\s*goal:[^)]+\s*\)", "", content).strip()

    def _extract_store_key(self, content: str) -> str:
        """Extract store key from directive: '... -> store: weather'"""
        if "->" in content:
            parts = content.split("->", 1)
            if len(parts) == 2:
                directive = parts[1].strip()
                if directive.startswith("store:"):
                    store_key = directive.split(":", 1)[1].strip()
                    logger.debug(f"Found store directive: {store_key}")
                    return store_key
        return None

    def _remove_store_directive(self, content: str) -> str:
        """Remove store directive from content."""
        if "->" in content:
            return content.split("->", 1)[0].strip()
        return content

    def _extract_leaf_steps(self, steps: List[Step]) -> List[Step]:
        """Extract only leaf steps (exclude parent/container bullets)."""
        leaf_steps: List[Step] = []
        
        for idx, step in enumerate(steps):
            # Check if next step has deeper indentation (making this a parent)
            next_indent = steps[idx + 1].indent if idx + 1 < len(steps) else step.indent
            if next_indent > step.indent:
                logger.debug(f"Skipping container step: '{step.text}'")
                continue  # don't include parent/meta bullets
            leaf_steps.append(step)
            
        return leaf_steps 