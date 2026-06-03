"""
FRIDAY-MAF Communicator Agent
Handles ALL user-facing interactions.
  In:  raw natural language string
  Out: structured Task object  (on the way in)
       human-readable string    (on the way out)
"""
from __future__ import annotations
from core.agents.base_agent import BaseAgent
from core.models import AgentType, Task, EventType
from core.observability.logger import tracer


class CommunicatorAgent(BaseAgent):
    agent_type = AgentType.COMMUNICATOR

    system_prompt = """You are the Communicator Agent for FRIDAY-MAF.
Your responsibilities:
1. Parse user messages and extract a clear, atomic goal.
2. Format completed task results into friendly, concise responses.

When parsing input, respond ONLY with a JSON object:
{
  "goal": "<one-sentence description of what must be accomplished>",
  "metadata": {
    "complexity": "low|medium|high",
    "domain":     "research|coding|data|general",
    "urgency":    "normal|high"
  }
}

When formatting output, respond with clean, human-readable Markdown.
Never expose internal agent names, task IDs, or system errors to the user.
"""

    async def run(self, user_input: str) -> Task:
        """
        Convert a user message into a structured Task.
        Called by the API layer on every new request.
        """
        async with tracer.trace("communicator_parse", agent="communicator"):
            parsed = await self.think_json(
                f"Parse this user request:\n\n{user_input}"
            )

        task = Task(
            user_input = user_input,
            goal       = parsed.get("goal", user_input),
            metadata   = parsed.get("metadata", {}),
        )

        # Persist for recovery / replay — mode='json' ensures datetime is ISO-serializable
        await self.memory.store_task_state(task.task_id, task.model_dump(mode="json"))
        await self.emit(EventType.TASK_CREATED, task_id=task.task_id, goal=task.goal)
        self.logger.info("task_created", extra={"task_id": task.task_id, "goal": task.goal})
        return task

    async def format_response(self, task: Task) -> str:
        """
        Convert a completed Task into a user-facing string.
        Called by the API layer after orchestration completes.
        """
        if task.error:
            return f"I'm sorry, I wasn't able to complete your request.\n\n**Reason:** {task.error}"

        context = (
            f"User's original request: {task.user_input}\n\n"
            f"Result:\n{task.final_result}"
        )

        async with tracer.trace("communicator_format", agent="communicator"):
            formatted = await self.think(
                "Format this result for the user. Be helpful, clear, and concise.",
                context=context,
                temperature=0.5,
            )
        return formatted