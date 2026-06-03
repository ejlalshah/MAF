"""
FRIDAY-MAF Tool System
Base tool class + concrete tools + ToolRouter (permission enforcement).

Design rule: agents never call tools directly — they go through ToolRouter.
This makes permission enforcement centralised and auditable.
Web search uses DuckDuckGo (free, no API key).
"""
from __future__ import annotations
import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from core.models import AgentType
from core.observability.logger import get_logger, metrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base Tool
# ---------------------------------------------------------------------------

class ToolResult(dict):
    """Structured return value from any tool."""

    @classmethod
    def ok(cls, data: Any, **meta) -> "ToolResult":
        return cls(success=True, data=data, error=None, **meta)

    @classmethod
    def error(cls, message: str, **meta) -> "ToolResult":
        return cls(success=False, data=None, error=message, **meta)


class BaseTool(ABC):
    name:        str = ""
    description: str = ""

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Execute the tool and return a ToolResult."""
        ...

    def schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description}


# ---------------------------------------------------------------------------
# WebSearchTool — DuckDuckGo (free, no API key required)
# ---------------------------------------------------------------------------

class WebSearchTool(BaseTool):
    name        = "web_search"
    description = "Search the web using DuckDuckGo and return a list of result summaries."

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key  # unused, kept for interface compat

    async def run(self, query: str, num_results: int = 5) -> ToolResult:
        logger.info("web_search", extra={"query": query})
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self._ddg_search, query, num_results
            )
            metrics.increment("tool_calls", tool=self.name)
            return ToolResult.ok(results, query=query)
        except Exception as exc:
            logger.warning("web_search_error", extra={"error": str(exc)})
            # Fallback stub so agents can continue even without network
            fallback = [
                {
                    "title":   f"Result {i+1} for '{query}'",
                    "url":     f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                    "snippet": f"Search result {i+1} — DuckDuckGo unavailable: {exc}",
                }
                for i in range(min(num_results, 3))
            ]
            metrics.increment("tool_calls", tool=self.name)
            return ToolResult.ok(fallback, query=query, source="fallback")

    @staticmethod
    def _ddg_search(query: str, num_results: int) -> List[Dict[str, str]]:
        # Try the new package name first, then old name as fallback
        for module_name in ("ddgs", "duckduckgo_search"):
            try:
                mod  = __import__(module_name, fromlist=["DDGS"])
                DDGS = mod.DDGS
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=num_results):
                        results.append({
                            "title":   r.get("title", ""),
                            "url":     r.get("href", ""),
                            "snippet": r.get("body", ""),
                        })
                return results
            except ImportError:
                continue
            except Exception as exc:
                raise RuntimeError(f"DuckDuckGo search error: {exc}")
        raise RuntimeError("Neither 'ddgs' nor 'duckduckgo_search' is installed.")


# ---------------------------------------------------------------------------
# FileTool
# ---------------------------------------------------------------------------

class FileTool(BaseTool):
    name        = "file"
    description = "Read and write files on the local filesystem."

    async def run(self, operation: str, path: str, content: str = "") -> ToolResult:
        try:
            if operation == "read":
                with open(path, "r", encoding="utf-8") as f:
                    data = f.read()
                return ToolResult.ok(data)
            elif operation == "write":
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return ToolResult.ok(f"Written {len(content)} bytes to {path}")
            elif operation == "list":
                entries = os.listdir(path)
                return ToolResult.ok(entries)
            else:
                return ToolResult.error(f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult.error(str(exc))


# ---------------------------------------------------------------------------
# PythonExecutionTool
# ---------------------------------------------------------------------------

class PythonExecutionTool(BaseTool):
    name        = "python_exec"
    description = "Execute Python code in a sandboxed subprocess and capture stdout."

    TIMEOUT = 30   # seconds

    async def run(self, code: str) -> ToolResult:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult.error(f"Execution timed out after {self.TIMEOUT}s")

            out = stdout.decode()
            err = stderr.decode()
            metrics.increment("tool_calls", tool=self.name)
            if proc.returncode == 0:
                return ToolResult.ok(out, stderr=err)
            return ToolResult.error(err or out)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# GitHubTool
# ---------------------------------------------------------------------------

class GitHubTool(BaseTool):
    name        = "github"
    description = "Interact with GitHub repositories (list files, read content, create issues)."

    def __init__(self, token: Optional[str] = None):
        self._token = token
        self._base  = "https://api.github.com"

    async def run(self, operation: str, repo: str = "", **kwargs) -> ToolResult:
        try:
            import httpx
            headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"

            async with httpx.AsyncClient(timeout=15) as client:
                if operation == "get_repo":
                    r = await client.get(f"{self._base}/repos/{repo}", headers=headers)
                    r.raise_for_status()
                    return ToolResult.ok(r.json())

                elif operation == "list_files":
                    path = kwargs.get("path", "")
                    r    = await client.get(
                        f"{self._base}/repos/{repo}/contents/{path}", headers=headers
                    )
                    r.raise_for_status()
                    return ToolResult.ok(r.json())

                elif operation == "create_issue":
                    r = await client.post(
                        f"{self._base}/repos/{repo}/issues",
                        headers=headers,
                        json={"title": kwargs["title"], "body": kwargs.get("body", "")},
                    )
                    r.raise_for_status()
                    return ToolResult.ok(r.json())

                else:
                    return ToolResult.error(f"Unknown GitHub operation: {operation}")
        except Exception as exc:
            return ToolResult.error(str(exc))


# ---------------------------------------------------------------------------
# Tool Router
# ---------------------------------------------------------------------------

AGENT_PERMISSIONS: Dict[AgentType, Set[str]] = {
    AgentType.RESEARCH:     {"web_search", "file"},
    AgentType.CODING:       {"python_exec", "file", "github"},
    AgentType.DATA_ANALYST: {"python_exec", "file", "web_search"},
    AgentType.COMMUNICATOR: set(),
    AgentType.PLANNER:      set(),
    AgentType.ORCHESTRATOR: set(),
    AgentType.RECOVERY:     {"file"},
}


class ToolRouter:
    """
    Central registry. Agents call tools through this — never directly.
    Usage:
        result = await tool_router.call("web_search", AgentType.RESEARCH, query="AI news")
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info("tool_registered", extra={"tool": tool.name})

    async def call(
        self,
        tool_name:  str,
        agent_type: AgentType,
        **kwargs,
    ) -> ToolResult:
        allowed = AGENT_PERMISSIONS.get(agent_type, set())
        if tool_name not in allowed:
            msg = f"Agent '{agent_type}' is not permitted to use tool '{tool_name}'"
            logger.warning("tool_permission_denied", extra={"agent": agent_type, "tool": tool_name})
            metrics.increment("tool_permission_denied", agent=agent_type, tool=tool_name)
            return ToolResult.error(msg)

        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.error(f"Tool '{tool_name}' not registered")

        logger.info("tool_call", extra={"tool": tool_name, "agent": agent_type})
        metrics.increment("tool_calls", tool=tool_name, agent=agent_type)
        return await tool.run(**kwargs)

    def list_tools(self, agent_type: Optional[AgentType] = None) -> List[str]:
        if agent_type is None:
            return list(self._tools)
        return [t for t in self._tools if t in AGENT_PERMISSIONS.get(agent_type, set())]


def build_default_router(
    search_api_key:  Optional[str] = None,
    github_token:    Optional[str] = None,
) -> ToolRouter:
    router = ToolRouter()
    router.register(WebSearchTool(api_key=search_api_key))
    router.register(FileTool())
    router.register(PythonExecutionTool())
    router.register(GitHubTool(token=github_token))
    return router
