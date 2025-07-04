"""Data models used internally by JenticReasoner."""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional

from pydantic import BaseModel, Field


class Step(BaseModel):
    """A single plan step produced by the LLM planner.

    Attributes
    ----------
    text:
        Raw step text without leading bullet/indent.
    indent:
        Indentation level (0-based). Used to determine hierarchy / sub-steps.
    status:
        Execution status. One of ``pending``, ``running``, ``done``, ``failed``.
    result:
        Result data returned from the action.
    error:
        Error message if the step failed.
    reflection_attempts:
        Number of reflection retries already attempted for this step.
    """

    text: str
    indent: int
    status: str = "pending"  # pending | running | done | failed
    result: Optional[Any] = None
    error: Optional[str] = None
    reflection_attempts: int = 0


class ReasonerState(BaseModel):
    """Mutable state shared across reasoning iterations."""

    goal: str
    plan: Deque[Step] = Field(default_factory=deque)
    history: List[str] = Field(default_factory=list)
    is_complete: bool = False
