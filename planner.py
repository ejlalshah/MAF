"""
FRIDAY-MAF Planner Agent
Converts a high-level goal into a Directed Acyclic Graph (DAG) of SubTasks.
Each node specifies: task_id, description, required_agent, dependencies.
"""
from __future__ import annotations
import json
from typing import List
from core.agents.base_agent import BaseAgent
from core.models import AgentType, SubTask, Task, TaskStatus, EventType
from core.observability.logger import tracer


class PlannerAgent(BaseAgent):
    agent_type = AgentType.PLANNER

    system_prompt = """You are the Planner Agent for FRIDAY-MAF.
Your job is to decompose a high-level goal into a minimal, executable DAG.

Rules:
- Only use these agent types: research, coding, data_analyst
- Each subtask must be a single, atomic unit of work
- dependencies must list task_ids of subtasks that MUST complete first
- Avoid circular dependencies
- Keep the plan under 8 subtasks — prefer fewer, broader tasks

Always respond with a JSON array:
[
  {
    "task_description": "<what this subtask must do>",
    "required_agent":   "research|coding|data_analyst",
    "dependencies":     []
  },
  ...
]
"""

    async def run(self, task: Task) -> Task:
        """
        Take a Task with a goal, return the same Task with subtasks populated.
        """
        task.status = TaskStatus.PLANNING
        task.mark_updated()

        # Pull relevant past episodes to inform planning
        past_episodes = await self.memory.recall_similar(task.goal, limit=3)
        context = ""
        if past_episodes:
            context = "Related past executions:\n" + json.dumps(
                [e["value"] for e in past_episodes], indent=2
            )

        async with tracer.trace("planner_run", task_id=task.task_id, agent="planner"):
            raw_plan = await self.think_json(
                f"Decompose this goal into subtasks:\n\n{task.goal}",
                context=context or None,
            )

        subtasks: List[SubTask] = []
        # Map sequential "dependencies" indices to actual task_ids
        id_map: dict = {}   # index → task_id
        for i, item in enumerate(raw_plan):
            deps = [id_map[d] for d in item.get("dependencies", []) if d in id_map]
            st   = SubTask(
                task_description = item["task_description"],
                required_agent   = AgentType(item["required_agent"]),
                dependencies     = deps,
            )
            id_map[i]  = st.task_id
            subtasks.append(st)

        task.subtasks = subtasks
        task.mark_updated()
        await self.memory.store_task_state(task.task_id, task.model_dump(mode="json"))
        await self.emit(EventType.PLAN_COMPLETED, task_id=task.task_id, subtask_count=len(subtasks))
        self.logger.info("plan_created", extra={"task_id": task.task_id, "subtasks": len(subtasks)})
        return task

    def _execution_order(self, subtasks: List[SubTask]) -> List[List[SubTask]]:
        """
        Topological sort → list of parallel batches.
        Each batch's tasks have all dependencies in previous batches.
        """
        id_to_st  = {st.task_id: st for st in subtasks}
        completed: set = set()
        batches:   List[List[SubTask]] = []
        remaining  = list(subtasks)

        while remaining:
            batch = [
                st for st in remaining
                if all(dep in completed for dep in st.dependencies)
            ]
            if not batch:
                raise ValueError("Circular dependency detected in task DAG")
            for st in batch:
                completed.add(st.task_id)
                remaining.remove(st)
            batches.append(batch)

        return batches