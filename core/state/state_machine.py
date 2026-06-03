"""
FRIDAY-MAF State Machine
Enforces valid task lifecycle transitions.
Invalid transitions raise immediately — no silent state corruption.

Valid flow:
  CREATED → PLANNING → EXECUTING → REVIEWING → COMPLETED
                                 ↘ RETRYING  → EXECUTING (loop guard: max 3 per task)
                              anywhere → FAILED
"""
from __future__ import annotations
from typing import Dict, Set
from core.models import TaskStatus
from core.observability.logger import get_logger, metrics

logger = get_logger(__name__)

# Adjacency map: current_state → allowed next states
TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.CREATED:   {TaskStatus.PLANNING,  TaskStatus.FAILED},
    TaskStatus.PLANNING:  {TaskStatus.EXECUTING, TaskStatus.FAILED},
    TaskStatus.EXECUTING: {TaskStatus.REVIEWING, TaskStatus.RETRYING, TaskStatus.FAILED},
    TaskStatus.REVIEWING: {TaskStatus.COMPLETED, TaskStatus.RETRYING, TaskStatus.FAILED},
    TaskStatus.RETRYING:  {TaskStatus.EXECUTING, TaskStatus.FAILED},   # ← only path out
    TaskStatus.COMPLETED: set(),   # terminal
    TaskStatus.FAILED:    set(),   # terminal
}

MAX_RETRIES = 3


class StateMachineError(Exception):
    pass


class TaskStateMachine:
    """
    Tracks state for a single task.
    Use one instance per task — do not share across tasks.
    """

    def __init__(self, task_id: str):
        self.task_id     = task_id
        self.state       = TaskStatus.CREATED
        self.retry_count = 0
        self._transitions_log: list = []

    def transition(self, new_state: TaskStatus) -> None:
        """
        Attempt a transition.
        Raises StateMachineError on:
          - invalid transition
          - exceeding retry limit (infinite-loop guard)
        """
        allowed = TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise StateMachineError(
                f"Task {self.task_id}: invalid transition {self.state} → {new_state}. "
                f"Allowed: {allowed}"
            )

        # Infinite-loop guard — >= means the MAX_RETRIES-th attempt is the last allowed
        if new_state == TaskStatus.RETRYING:
            self.retry_count += 1
            if self.retry_count >= MAX_RETRIES:
                logger.warning(
                    "retry_limit_exceeded",
                    extra={"task_id": self.task_id, "retry_count": self.retry_count},
                )
                # Force to FAILED instead
                new_state = TaskStatus.FAILED

        old_state  = self.state
        self.state = new_state
        self._transitions_log.append((old_state, new_state))

        logger.info(
            "state_transition",
            extra={
                "task_id":   self.task_id,
                "from":      old_state,
                "to":        new_state,
                "retries":   self.retry_count,
            },
        )
        metrics.increment("state_transitions", from_state=old_state, to_state=new_state)

    def can_transition(self, new_state: TaskStatus) -> bool:
        return new_state in TRANSITIONS.get(self.state, set())

    def is_terminal(self) -> bool:
        return self.state in {TaskStatus.COMPLETED, TaskStatus.FAILED}

    def history(self) -> list:
        return list(self._transitions_log)