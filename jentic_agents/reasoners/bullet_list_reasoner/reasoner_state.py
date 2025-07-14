"""State management for BulletPlanReasoner."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Deque
from collections import deque
from enum import Enum


class StepStatus(Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    DONE = "done"
    FAILED = "failed"


@dataclass
class Step:
    """Individual step in a bullet plan."""
    text: str
    indent: int = 0
    store_key: Optional[str] = None
    goal_context: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    tool_id: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    tool_name: Optional[str] = None
    reflection_attempts: int = 0
    load_from_store: Optional[Dict[str, str]] = None
    step_type: Optional[str] = None

    def mark_running(self) -> "Step":
        """Return a new Step marked as running."""
        return Step(
            text=self.text,
            indent=self.indent,
            store_key=self.store_key,
            goal_context=self.goal_context,
            status=StepStatus.RUNNING,
            result=self.result,
            tool_id=self.tool_id,
            params=self.params,
            tool_name=self.tool_name,
            reflection_attempts=self.reflection_attempts,
            load_from_store=self.load_from_store,
            step_type=self.step_type,
        )

    def mark_done(self, result: Any = None) -> "Step":
        """Return a new Step marked as done with result."""
        return Step(
            text=self.text,
            indent=self.indent,
            store_key=self.store_key,
            goal_context=self.goal_context,
            status=StepStatus.DONE,
            result=result or self.result,
            tool_id=self.tool_id,
            params=self.params,
            tool_name=self.tool_name,
            reflection_attempts=self.reflection_attempts,
            load_from_store=self.load_from_store,
            step_type=self.step_type,
        )

    def mark_failed(self, result: Any = None) -> "Step":
        """Return a new Step marked as failed with error result."""
        return Step(
            text=self.text,
            indent=self.indent,
            store_key=self.store_key,
            goal_context=self.goal_context,
            status=StepStatus.FAILED,
            result=result or self.result,
            tool_id=self.tool_id,
            params=self.params,
            tool_name=self.tool_name,
            reflection_attempts=self.reflection_attempts + 1,
            load_from_store=self.load_from_store,
            step_type=self.step_type,
        )

    def with_tool(self, tool_id: str, tool_name: str = None) -> "Step":
        """Return a new Step with tool information."""
        return Step(
            text=self.text,
            indent=self.indent,
            store_key=self.store_key,
            goal_context=self.goal_context,
            status=self.status,
            result=self.result,
            tool_id=tool_id,
            params=self.params,
            tool_name=tool_name or tool_id,
            reflection_attempts=self.reflection_attempts,
            load_from_store=self.load_from_store,
            step_type=self.step_type,
        )


@dataclass(frozen=True)
class ReasonerState:
    """Immutable state for BulletPlanReasoner."""
    goal: str
    plan: Deque[Step] = field(default_factory=deque)
    history: List[str] = field(default_factory=list)
    goal_completed: bool = False
    failed: bool = False
    iteration: int = 0

    def with_plan(self, plan: Deque[Step]) -> "ReasonerState":
        """Return new state with updated plan."""
        return ReasonerState(
            goal=self.goal,
            plan=plan,
            history=self.history,
            goal_completed=self.goal_completed,
            failed=self.failed,
            iteration=self.iteration,
        )

    def with_history(self, new_entry: str) -> "ReasonerState":
        """Return new state with added history entry."""
        return ReasonerState(
            goal=self.goal,
            plan=self.plan,
            history=self.history + [new_entry],
            goal_completed=self.goal_completed,
            failed=self.failed,
            iteration=self.iteration,
        )

    def mark_completed(self) -> "ReasonerState":
        """Return new state marked as completed."""
        return ReasonerState(
            goal=self.goal,
            plan=self.plan,
            history=self.history,
            goal_completed=True,
            failed=self.failed,
            iteration=self.iteration,
        )

    def mark_failed(self) -> "ReasonerState":
        """Return new state marked as failed."""
        return ReasonerState(
            goal=self.goal,
            plan=self.plan,
            history=self.history,
            goal_completed=self.goal_completed,
            failed=True,
            iteration=self.iteration,
        )

    def next_iteration(self) -> "ReasonerState":
        """Return new state with incremented iteration."""
        return ReasonerState(
            goal=self.goal,
            plan=self.plan,
            history=self.history,
            goal_completed=self.goal_completed,
            failed=self.failed,
            iteration=self.iteration + 1,
        )

    def advance_plan(self) -> "ReasonerState":
        """Return new state with first step removed from plan."""
        new_plan = deque(list(self.plan)[1:])
        return ReasonerState(
            goal=self.goal,
            plan=new_plan,
            history=self.history,
            goal_completed=self.goal_completed,
            failed=self.failed,
            iteration=self.iteration,
        )

    def update_current_step(self, updated_step: Step) -> "ReasonerState":
        """Return new state with current step updated."""
        if not self.plan:
            return self
        
        new_plan = deque([updated_step] + list(self.plan)[1:])
        return ReasonerState(
            goal=self.goal,
            plan=new_plan,
            history=self.history,
            goal_completed=self.goal_completed,
            failed=self.failed,
            iteration=self.iteration,
        )

    @property
    def current_step(self) -> Optional[Step]:
        """Get the current step being executed."""
        return self.plan[0] if self.plan else None

    @property
    def remaining_steps(self) -> int:
        """Get number of remaining steps."""
        return len(self.plan)

    def validate(self) -> bool:
        """Validate state consistency."""
        if self.goal_completed and self.failed:
            return False  # Can't be both completed and failed
        if self.goal_completed and self.plan:
            return False  # Can't be completed with remaining steps
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for debugging."""
        return {
            "goal": self.goal,
            "plan_steps": len(self.plan),
            "history_entries": len(self.history),
            "goal_completed": self.goal_completed,
            "failed": self.failed,
            "iteration": self.iteration,
            "current_step": self.current_step.text if self.current_step else None,
        } 