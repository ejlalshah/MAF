"""
FRIDAY-MAF Recovery Agent
Handles everything that goes wrong:
  - Tool failures
  - Malformed / unparseable worker output
  - Timeouts
  - Context window overflow
  - Decides: retry | alternative worker | escalate
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from core.agents.base_agent import BaseAgent
from core.models import AgentType, SubTask, WorkerOutput, EventType
from core.observability.logger import tracer, metrics


class RecoveryAction(str, Enum):
    RETRY           = "retry"
    ALTERNATIVE     = "alternative_worker"
    SIMPLIFY        = "simplify_task"
    ESCALATE        = "escalate"


# Alternative agent map — if primary agent fails, try this one
FALLBACK_AGENT: dict = {
    AgentType.RESEARCH:     AgentType.DATA_ANALYST,
    AgentType.DATA_ANALYST: AgentType.RESEARCH,
    AgentType.CODING:       AgentType.CODING,  # No fallback for coding
}


class RecoveryAgent(BaseAgent):
    agent_type = AgentType.RECOVERY

    system_prompt = """You are RecoveryAgent — the safety net of the FRIDAY-MAF system.
When a worker or tool fails, you decide the best recovery path.

Given:
- Original subtask description
- Error message or failure reason
- Number of retries already attempted

Decide one of these actions:
1. "retry"              — the failure is transient (network, timeout); just try again
2. "alternative_worker" — the current agent type is wrong for this task
3. "simplify_task"      — the task is too complex; break it into smaller pieces
4. "escalate"           — the failure is unrecoverable; surface to the user

Respond ONLY with JSON:
{
  "action":       "retry|alternative_worker|simplify_task|escalate",
  "reason":       "<why this action>",
  "instructions": "<specific instructions for the retry or simplified task>"
}
"""

    async def run(  # type: ignore[override]
        self,
        subtask:      SubTask,
        error:        str,
        retry_count:  int = 0,
    ) -> tuple[RecoveryAction, Optional[str]]:
        """
        Analyse the failure and decide recovery action.

        Returns:
            (action, instructions)
            instructions may contain a modified task description or None
        """
        async with tracer.trace("recovery_run", task_id=subtask.task_id, agent="recovery"):
            raw = await self.think_json(
                f"""Task: {subtask.task_description}
Agent: {subtask.required_agent}
Error: {error}
Retries already attempted: {retry_count}

Decide the recovery action."""
            )

        action_str   = raw.get("action", RecoveryAction.ESCALATE)
        instructions = raw.get("instructions")
        reason       = raw.get("reason", "")

        # Safety: after 2 retries, force escalation regardless of LLM decision
        if retry_count >= 2 and action_str == RecoveryAction.RETRY:
            action_str   = RecoveryAction.ESCALATE
            instructions = f"Max retries reached. Original error: {error}"

        action = RecoveryAction(action_str)

        self.logger.info(
            "recovery_decision",
            extra={
                "task_id":     subtask.task_id,
                "action":      action,
                "reason":      reason,
                "retry_count": retry_count,
            },
        )
        metrics.increment("recovery_actions", action=action)
        await self.emit(EventType.RETRY_TRIGGERED, task_id=subtask.task_id, action=action, reason=reason)
        return action, instructions

    def get_fallback_agent(self, agent_type: AgentType) -> AgentType:
        return FALLBACK_AGENT.get(agent_type, AgentType.RESEARCH)