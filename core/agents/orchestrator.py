"""
FRIDAY-MAF Orchestrator Agent
The execution engine. Given a planned Task:
1. Resolves execution order (topological sort of DAG)
2. Dispatches each SubTask to the correct worker
3. Runs critics on every output
4. Handles retries via RecoveryAgent
5. Manages state transitions throughout
6. Collects and assembles final result
"""
from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, Optional

from core.agents.base_agent import BaseAgent
from core.agents.recovery import RecoveryAgent, RecoveryAction
from core.critics.critics import CriticRunner
from core.models import AgentType, SubTask, Task, TaskStatus, WorkerOutput, EventType
from core.state.state_machine import TaskStateMachine
from core.workers.workers import BaseWorker, ResearchAgent, CodingAgent, DataAnalystAgent
from core.observability.logger import tracer, metrics


class OrchestratorAgent(BaseAgent):
    agent_type = AgentType.ORCHESTRATOR

    system_prompt = """You are the Orchestrator — the coordinator of all agents.
You do not make decisions yourself; you route tasks and collect results."""

    def __init__(self, llm, memory, event_bus, tool_router, critic_runner: CriticRunner, recovery: RecoveryAgent):
        super().__init__(llm, memory, event_bus)
        self.tool_router    = tool_router
        self.critic_runner  = critic_runner
        self.recovery       = recovery
        # Worker registry — keyed by AgentType
        self._workers: Dict[AgentType, BaseWorker] = {
            AgentType.RESEARCH:     ResearchAgent(llm, memory, event_bus, tool_router),
            AgentType.CODING:       CodingAgent(llm, memory, event_bus, tool_router),
            AgentType.DATA_ANALYST: DataAnalystAgent(llm, memory, event_bus, tool_router),
        }

    async def run(self, task: Task) -> Task:  # type: ignore[override]
        """
        Main orchestration loop.
        Executes the DAG respecting dependencies, applies critics, handles recovery.
        """
        sm = TaskStateMachine(task.task_id, task.status)

        try:
            sm.transition(TaskStatus.EXECUTING)
            task.status = TaskStatus.EXECUTING
            task.mark_updated()

            # Topological sort → batches that can run in parallel
            batches = self._topological_batches(task.subtasks)
            completed_results: Dict[str, Any] = {}

            for batch in batches:
                batch_results = await asyncio.gather(
                    *[self._execute_subtask(st, completed_results, sm) for st in batch],
                    return_exceptions=True,
                )
                for st, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        task.error  = str(result)
                        task.status = TaskStatus.FAILED
                        sm.transition(TaskStatus.FAILED)
                        await self.emit(EventType.TASK_FAILED, task_id=task.task_id, error=str(result))
                        return task
                    completed_results[st.task_id] = result

            # All subtasks done — assemble final result
            task.final_result = self._assemble_result(task, completed_results)
            sm.transition(TaskStatus.REVIEWING)
            task.status = TaskStatus.REVIEWING
            task.mark_updated()
            sm.transition(TaskStatus.COMPLETED)
            task.status = TaskStatus.COMPLETED
            task.mark_updated()

            await self.memory.store_task_state(task.task_id, task.model_dump(mode="json"))
            await self.memory.record_episode(task.task_id, json.dumps({"goal": task.goal, "result": str(task.final_result)[:200]}))
            await self.emit(EventType.TASK_COMPLETED, task_id=task.task_id)
            metrics.increment("tasks_completed")
            self.logger.info("task_completed", extra={"task_id": task.task_id})

        except Exception as exc:
            task.error  = str(exc)
            task.status = TaskStatus.FAILED
            await self.emit(EventType.TASK_FAILED, task_id=task.task_id, error=str(exc))
            metrics.increment("tasks_failed")
            self.logger.error("orchestrator_error", extra={"task_id": task.task_id, "error": str(exc)})

        return task

    async def _execute_subtask(
        self,
        subtask:   SubTask,
        context:   Dict[str, Any],
        sm:        TaskStateMachine,
    ) -> Any:
        """
        Execute one SubTask with the retry/critic loop.
        Returns the raw result on success or raises on unrecoverable failure.
        """
        retry_count = 0
        context_str = json.dumps(context, indent=2) if context else None

        while True:
            subtask.status = TaskStatus.EXECUTING
            subtask.mark_updated()
            await self.emit(EventType.TASK_ASSIGNED, task_id=subtask.task_id, agent=subtask.required_agent)

            worker = self._workers.get(subtask.required_agent)
            if worker is None:
                raise RuntimeError(f"No worker registered for agent type: {subtask.required_agent}")

            async with tracer.trace("subtask_execute", task_id=subtask.task_id, agent=str(subtask.required_agent)):
                output: WorkerOutput = await worker.run(subtask, context=context_str)

            if not output.success:
                action, instructions = await self.recovery.run(subtask, output.error or "unknown error", retry_count)
                if action == RecoveryAction.RETRY:
                    retry_count += 1
                    subtask.retry_count = retry_count
                    subtask.status = TaskStatus.RETRYING
                    continue
                elif action == RecoveryAction.ALTERNATIVE:
                    subtask.required_agent = self.recovery.get_fallback_agent(subtask.required_agent)
                    retry_count += 1
                    continue
                else:
                    raise RuntimeError(f"Subtask failed: {output.error}")

            # Worker succeeded → run critics
            subtask.status = TaskStatus.REVIEWING
            sm_result: bool
            critic_results: list
            sm_result, critic_results = await self.critic_runner.evaluate(subtask, output)

            if sm_result:
                subtask.status = TaskStatus.COMPLETED
                subtask.result = output.result
                await self.emit(EventType.TASK_COMPLETED, task_id=subtask.task_id)
                return output.result

            # Critics failed
            await self.emit(EventType.CRITIC_FAILED, task_id=subtask.task_id, critics=[r.model_dump() for r in critic_results])
            feedback = self.critic_runner.combined_feedback(critic_results)

            if retry_count >= 3:
                raise RuntimeError(f"SubTask failed after 3 critic retries.\n{feedback}")

            # Inject feedback into the next attempt's context
            context_str = (context_str or "") + f"\n\n--- CRITIC FEEDBACK (retry {retry_count+1}) ---\n{feedback}"
            retry_count += 1
            subtask.retry_count = retry_count
            subtask.status      = TaskStatus.RETRYING
            await self.emit(EventType.RETRY_TRIGGERED, task_id=subtask.task_id, retry=retry_count)
            self.logger.info("critic_retry", extra={"task_id": subtask.task_id, "retry": retry_count})

    def _topological_batches(self, subtasks: List[SubTask]) -> List[List[SubTask]]:
        completed: set = set()
        remaining       = list(subtasks)
        batches:   List[List[SubTask]] = []

        while remaining:
            batch = [st for st in remaining if all(d in completed for d in st.dependencies)]
            if not batch:
                raise ValueError("Circular dependency in task DAG")
            for st in batch:
                completed.add(st.task_id)
                remaining.remove(st)
            batches.append(batch)

        return batches

    def _assemble_result(self, task: Task, results: Dict[str, Any]) -> Any:
        """Merge subtask results into a single coherent output."""
        if len(results) == 1:
            return list(results.values())[0]
        return {
            "goal":     task.goal,
            "subtasks": [
                {"description": st.task_description, "result": results.get(st.task_id)}
                for st in task.subtasks
            ],
        }