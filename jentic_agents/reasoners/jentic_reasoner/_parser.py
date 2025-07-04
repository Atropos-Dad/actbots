"""Utilities for parsing LLM-generated markdown plans.

Currently supports simple indented bullet lists, e.g.::

    - Do X
        - Substep
    - Do Y

Leading bullet symbols (*, -, +) or enumerated lists (1., 2.) are removed and
indent level is determined from leading whitespace (2 spaces == 1 indent).
"""
from __future__ import annotations

import re
from collections import deque
from typing import Deque

from ._models import Step

# Two spaces per indent level keeps markdown compatibility.
_INDENT_SIZE = 2
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)(.*)$")


def _line_indent(text: str) -> int:
    """Return indent level (0-based) from leading spaces."""
    spaces = len(text) - len(text.lstrip(" "))
    return spaces // _INDENT_SIZE


def _strip_bullet(text: str) -> str:
    """Remove leading bullet/number and extra whitespace."""
    match = _BULLET_PATTERN.match(text)
    return match.group(1).rstrip() if match else text.strip()


def parse_bullet_plan(markdown: str) -> Deque[Step]:
    """Parse an indented markdown bullet list into a queue of ``Step`` objects.

    The function intentionally ignores any non-list lines. It also **only**
    extracts structure — higher-level orchestration (nesting relationships,
    container steps, etc.) is left for the caller to interpret.
    """
    steps: Deque[Step] = deque()
    for raw_line in markdown.splitlines():
        # Skip empty lines
        if not raw_line.strip():
            continue
        indent = _line_indent(raw_line)
        stripped = _strip_bullet(raw_line)
        steps.append(Step(text=stripped, indent=indent))
    return steps
