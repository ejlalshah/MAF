"""
FRIDAY-MAF Worker Agents
BaseWorker + ResearchAgent + CodingAgent + DataAnalystAgent
Each worker:
  - Has its own system prompt tuned to its role
  - Has access ONLY to tools relevant to that role
  - Returns structured JSON via WorkerOutput
"""
from __future__ import annotations
import json
from abc import abstractmethod
from typing import Any, Optional

from core.agents.base_agent import BaseAgent
from core.models import AgentType, SubTask, WorkerOutput
from core.tools.tool_router import ToolRouter
from core.observability.logger import tracer, metrics


# ---------------------------------------------------------------------------
# Base Worker
# ---------------------------------------------------------------------------

class BaseWorker(BaseAgent):
    """
    Workers execute a single SubTask and return a WorkerOutput.
    They are the only agents that call tools.
    """

    def __init__(self, llm, memory, event_bus, tool_router: ToolRouter):
        super().__init__(llm, memory, event_bus)
        self.tool_router = tool_router

    async def run(self, subtask: SubTask, context: Optional[str] = None) -> WorkerOutput:
        async with tracer.trace("worker_run", task_id=subtask.task_id, agent=str(self.agent_type)):
            try:
                result = await self.execute(subtask, context)
                metrics.increment("worker_successes", agent=str(self.agent_type))
                return WorkerOutput(
                    agent_type = self.agent_type,
                    task_id    = subtask.task_id,
                    success    = True,
                    result     = result,
                )
            except Exception as exc:
                self.logger.error(
                    "worker_error",
                    extra={"task_id": subtask.task_id, "error": str(exc)},
                )
                metrics.increment("worker_failures", agent=str(self.agent_type))
                return WorkerOutput(
                    agent_type = self.agent_type,
                    task_id    = subtask.task_id,
                    success    = False,
                    error      = str(exc),
                )

    @abstractmethod
    async def execute(self, subtask: SubTask, context: Optional[str]) -> Any:
        """Subclasses implement the actual task logic here."""
        ...

    async def _call_tool(self, tool_name: str, **kwargs) -> Any:
        result = await self.tool_router.call(tool_name, self.agent_type, **kwargs)
        if not result["success"]:
            raise RuntimeError(f"Tool '{tool_name}' failed: {result['error']}")
        return result["data"]


# ---------------------------------------------------------------------------
# Research Agent
# ---------------------------------------------------------------------------

class ResearchAgent(BaseWorker):
    agent_type = AgentType.RESEARCH

    system_prompt = """You are ResearchAgent — a focused, rigorous researcher.
Given a task, your job is to:
1. Identify the key questions that need answering.
2. Search the web or read files for relevant information.
3. Synthesise findings into a structured JSON report.

Always output:
{
  "summary": "<2-3 sentence overview>",
  "key_findings": ["finding 1", ...],
  "sources": ["url or filename", ...],
  "confidence": 0-100
}
"""

    async def execute(self, subtask: SubTask, context: Optional[str]) -> dict:
        # Step 1: identify search queries
        queries_raw = await self.think_json(
            f"Given this research task, list 2-3 specific web search queries:\n\n{subtask.task_description}",
            context=context,
        )
        queries = queries_raw if isinstance(queries_raw, list) else queries_raw.get("queries", [subtask.task_description])

        # Step 2: execute searches
        all_results = []
        for q in queries[:3]:
            results = await self._call_tool("web_search", query=q, num_results=3)
            all_results.extend(results)

        # Step 3: synthesise
        findings = await self.think_json(
            "Synthesise these search results into a research report:\n\n"
            f"Task: {subtask.task_description}\n\n"
            f"Results:\n{json.dumps(all_results, indent=2)}",
        )
        return findings


# ---------------------------------------------------------------------------
# Coding Agent
# ---------------------------------------------------------------------------

class CodingAgent(BaseWorker):
    agent_type = AgentType.CODING

    system_prompt = """You are CodingAgent — a senior software engineer.
Given a coding task:
1. Write clean, production-quality Python code.
2. Execute it to verify it works.
3. Return the final implementation.

Always output:
{
  "language": "python",
  "code": "<complete runnable code>",
  "explanation": "<what the code does>",
  "execution_output": "<actual stdout from running the code>",
  "tests_passed": true/false
}
"""

    async def execute(self, subtask: SubTask, context: Optional[str]) -> dict:
        # Step 1: generate code
        code_response = await self.think_json(
            f"Write Python code for this task:\n\n{subtask.task_description}",
            context=context,
        )
        code = code_response.get("code", "")

        if not code:
            raise ValueError("LLM returned empty code")

        # Step 2: execute and capture output
        exec_output = await self._call_tool("python_exec", code=code)

        # Step 3: return structured result
        return {
            "language":         "python",
            "code":             code,
            "explanation":      code_response.get("explanation", ""),
            "execution_output": exec_output,
            "tests_passed":     True,
        }


# ---------------------------------------------------------------------------
# Data Analyst Agent
# ---------------------------------------------------------------------------

class DataAnalystAgent(BaseWorker):
    agent_type = AgentType.DATA_ANALYST

    system_prompt = """You are DataAnalystAgent — an expert data analyst and statistician.
Given an analysis task:
1. Understand what insights are needed.
2. Write Python (pandas / numpy / matplotlib) to perform the analysis.
3. Execute the code and collect results.
4. Interpret the results in plain language.

Always output:
{
  "analysis_type": "<descriptive|predictive|diagnostic>",
  "code":          "<analysis code>",
  "results":       "<raw output or computed values>",
  "insights":      ["insight 1", ...],
  "recommendations": ["action 1", ...]
}
"""

    async def execute(self, subtask: SubTask, context: Optional[str]) -> dict:
        # Step 1: plan the analysis
        plan = await self.think_json(
            f"Plan the data analysis for this task:\n\n{subtask.task_description}",
            context=context,
        )

        # Step 2: generate analysis code
        code_raw = await self.think_json(
            f"Write Python analysis code for:\n\n{subtask.task_description}\n\nPlan:\n{json.dumps(plan)}",
        )
        code = code_raw.get("code", "")

        if not code:
            raise ValueError("No analysis code generated")

        # Step 3: execute
        exec_output = await self._call_tool("python_exec", code=code)

        # Step 4: interpret
        interpretation = await self.think_json(
            f"Interpret this analysis output:\n{exec_output}\n\nOriginal task:\n{subtask.task_description}"
        )

        return {
            "analysis_type":   plan.get("analysis_type", "descriptive"),
            "code":            code,
            "results":         exec_output,
            "insights":        interpretation.get("insights", []),
            "recommendations": interpretation.get("recommendations", []),
        }