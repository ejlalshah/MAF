"""
FRIDAY-MAF Autonomy Execution Loop
Continuously picks up CREATED/pending tasks and executes them in the background.
Persists all state changes to SQLite immediately.
Resumes unfinished tasks on startup.
"""
from __future__ import annotations
import asyncio
from typing import Dict, Optional, Set

from core.models import Task, TaskStatus
from core.persistence.sqlite_store import SQLiteStore
from core.observability.logger import get_logger, metrics

logger = get_logger(__name__)


class AutonomyLoop:
    """
    Background execution engine.
    - Runs every LOOP_INTERVAL seconds
    - Picks up CREATED tasks from the store
    - Dispatches them through the communicator → planner → orchestrator pipeline
    - Persists state after every transition
    - Emits events to the event bus for dashboard observability
    """

    def __init__(
        self,
        store:        SQLiteStore,
        communicator,
        planner,
        orchestrator,
        task_store:   Dict[str, Task],
        interval:     float = 3.0,
    ):
        self._store       = store
        self._comm        = communicator
        self._planner     = planner
        self._orch        = orchestrator
        self._task_store  = task_store   # shared in-memory dict (also used by API)
        self._interval    = interval
        self._running     = False
        self._active: Set[str] = set()  # task_ids currently being executed
        self._task:   Optional[asyncio.Task] = None

    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background loop."""
        self._running = True
        self._task    = asyncio.create_task(self._loop(), name="autonomy_loop")
        logger.info("autonomy_loop_started", extra={"interval": self._interval})

    async def stop(self) -> None:
        """Gracefully stop the loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("autonomy_loop_stopped")

    async def resume_unfinished(self) -> int:
        """
        On startup: load unfinished tasks from SQLite and add them to
        the in-memory task_store so the loop picks them up.
        """
        tasks = await self._store.load_unfinished_tasks()
        for task in tasks:
            # Re-queue as CREATED so the loop re-executes them
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.status = TaskStatus.CREATED
                self._task_store[task.task_id] = task
                await self._store.save_task(task)
        if tasks:
            logger.info("autonomy_resume", extra={"count": len(tasks)})
        return len(tasks)

    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("autonomy_loop_error", extra={"error": str(exc)})
            await asyncio.sleep(self._interval)

    async def _tick(self) -> None:
        """Process one batch of pending tasks."""
        pending = [
            t for tid, t in list(self._task_store.items())
            if t.status == TaskStatus.CREATED and tid not in self._active
        ]
        if not pending:
            return

        for task in pending:
            if task.task_id not in self._active:
                self._active.add(task.task_id)
                asyncio.create_task(
                    self._execute(task),
                    name=f"task_{task.task_id[:8]}"
                )

    async def _execute(self, task: Task) -> None:
        """Run the full pipeline for one task, persisting at every step."""
        try:
            # 1. Plan
            task.status = TaskStatus.PLANNING
            task.mark_updated()
            self._task_store[task.task_id] = task
            await self._store.save_task(task)

            task = await self._planner.run(task)
            self._task_store[task.task_id] = task
            await self._store.save_task(task)

            # 2. Execute
            task = await self._orch.run(task)
            self._task_store[task.task_id] = task
            await self._store.save_task(task)

            # 3. Format final result
            if task.status == TaskStatus.COMPLETED:
                task.final_result = await self._comm.format_response(task)
                self._task_store[task.task_id] = task
                await self._store.save_task(task)

            metrics.increment(
                "autonomy_tasks_processed",
                status=task.status.value,
            )
            logger.info("autonomy_task_done", extra={
                "task_id": task.task_id,
                "status":  task.status,
            })

        except Exception as exc:
            task.error  = str(exc)
            task.status = TaskStatus.FAILED
            task.mark_updated()
            self._task_store[task.task_id] = task
            await self._store.save_task(task)
            logger.error("autonomy_task_failed", extra={
                "task_id": task.task_id,
                "error":   str(exc),
            })
            metrics.increment("autonomy_tasks_failed")
        finally:
            self._active.discard(task.task_id)
