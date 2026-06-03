"""
FRIDAY-MAF Critic System
Three independent critics evaluate every WorkerOutput before it's accepted.
Fail → feedback sent back → worker retries (max 3 times).

Each critic is stateless — it sees only the task + output, nothing else.
This keeps them auditable and easy to test in isolation.
"""
from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List

from core.agents.base_agent import BaseAgent
from core.models import AgentType, CriticResult, CriticType, SubTask, WorkerOutput
from core.observability.logger import tracer, metrics


# ---------------------------------------------------------------------------
# Base Critic
# ---------------------------------------------------------------------------

class BaseCritic(BaseAgent):
    critic_type: CriticType
    # Critics only call the LLM — no tool access needed
    agent_type = AgentType.COMMUNICATOR   # reuse comms slot for LLM access

    PASS_THRESHOLD = 60   # score ≥ 60 → pass

    async def run(self, subtask: SubTask, worker_output: WorkerOutput) -> CriticResult:  # type: ignore[override]
        async with tracer.trace("critic_run", agent=str(self.critic_type), task_id=subtask.task_id):
            raw = await self.think_json(
                self._build_prompt(subtask, worker_output)
            )

        result = CriticResult(
            critic_type = self.critic_type,
            passed      = bool(raw.get("pass", False)),
            score       = int(raw.get("score", 0)),
            feedback    = str(raw.get("feedback", "")),
        )
        # Override pass/fail based on threshold (LLM may say "pass" but give score 40)
        result.passed = result.passed and result.score >= self.PASS_THRESHOLD

        metrics.increment(
            "critic_evaluations",
            critic=str(self.critic_type),
            passed=str(result.passed),
        )
        return result

    @abstractmethod
    def _build_prompt(self, subtask: SubTask, output: WorkerOutput) -> str:
        ...


# ---------------------------------------------------------------------------
# Logic Critic — checks correctness and coherence
# ---------------------------------------------------------------------------

class LogicCritic(BaseCritic):
    critic_type = CriticType.LOGIC

    system_prompt = """You are LogicCritic — a rigorous quality reviewer.
Evaluate whether the worker output is logically correct and coherent.

Check for:
- Factual accuracy
- Internal consistency  
- Completeness relative to the task
- No hallucinations or unsupported claims

Respond ONLY with JSON:
{
  "pass":     true/false,
  "score":    0-100,
  "feedback": "<specific, actionable explanation of what to fix>"
}
A score ≥ 70 means high quality. Below 60 means the task should be retried.
"""

    def _build_prompt(self, subtask: SubTask, output: WorkerOutput) -> str:
        return (
            f"Original task:\n{subtask.task_description}\n\n"
            f"Worker output:\n{output.result}\n\n"
            "Evaluate the logical correctness of this output."
        )


# ---------------------------------------------------------------------------
# Security Critic — checks for vulnerabilities, data exposure, unsafe code
# ---------------------------------------------------------------------------

class SecurityCritic(BaseCritic):
    critic_type = CriticType.SECURITY
    PASS_THRESHOLD = 50   # slightly lenient — most tasks aren't security-sensitive

    system_prompt = """You are SecurityCritic — a security expert.
Review the worker output for security risks.

Check for:
- SQL injection, XSS, SSRF, path traversal
- Hardcoded secrets or credentials
- Unsafe eval(), exec(), or shell injection
- Exposed PII or sensitive data
- Insecure defaults or missing auth checks

Respond ONLY with JSON:
{
  "pass":     true/false,
  "score":    0-100,
  "feedback": "<list specific vulnerabilities found or confirm clean>"
}
Score 100 = no issues. Below 50 = critical issue requiring a retry.
"""

    def _build_prompt(self, subtask: SubTask, output: WorkerOutput) -> str:
        return (
            f"Task context:\n{subtask.task_description}\n\n"
            f"Output to review:\n{output.result}\n\n"
            "Identify any security vulnerabilities."
        )


# ---------------------------------------------------------------------------
# Requirement Critic — checks spec compliance
# ---------------------------------------------------------------------------

class RequirementCritic(BaseCritic):
    critic_type = CriticType.REQUIREMENT

    system_prompt = """You are RequirementCritic — a product manager and QA engineer.
Check whether the output satisfies the original requirement.

Evaluate:
- Does the output directly address the task description?
- Are all specified requirements met?
- Is the output format correct?
- Are there missing edge cases or gaps?

Respond ONLY with JSON:
{
  "pass":     true/false,
  "score":    0-100,
  "feedback": "<what requirements are missing or what was done well>"
}
"""

    def _build_prompt(self, subtask: SubTask, output: WorkerOutput) -> str:
        return (
            f"Requirement:\n{subtask.task_description}\n\n"
            f"Delivered output:\n{output.result}\n\n"
            "Does this output satisfy the requirement?"
        )


# ---------------------------------------------------------------------------
# Critic Runner — orchestrates all critics for a single worker output
# ---------------------------------------------------------------------------

class CriticRunner:
    """
    Runs all critics in parallel and returns a combined verdict.
    If ANY critic fails → overall fail.
    """

    def __init__(self, critics: List[BaseCritic]):
        self.critics = critics

    async def evaluate(
        self,
        subtask: SubTask,
        output:  WorkerOutput,
    ) -> tuple[bool, List[CriticResult]]:
        """
        Returns (overall_pass, [CriticResult, ...])
        overall_pass = True only if ALL critics pass.
        """
        import asyncio
        results: List[CriticResult] = await asyncio.gather(
            *[critic.run(subtask, output) for critic in self.critics]
        )
        overall_pass = all(r.passed for r in results)
        return overall_pass, list(results)

    def combined_feedback(self, results: List[CriticResult]) -> str:
        failed = [r for r in results if not r.passed]
        if not failed:
            return "All critics passed."
        lines = ["The following critics found issues:\n"]
        for r in failed:
            lines.append(f"**{r.critic_type.value.title()} Critic** (score {r.score}/100):\n{r.feedback}")
        return "\n\n".join(lines)