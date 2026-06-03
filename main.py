"""
FRIDAY-MAF API Layer
FastAPI application exposing:
  POST /task        — submit a new task
  GET  /task/{id}   — poll task status + result
  GET  /health      — liveness check + metrics snapshot
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config.settings import settings
from core.llm.providers import create_llm_provider
from core.memory.memory_manager import MemoryManager
from core.events.event_bus import event_bus
from core.tools.tool_router import build_default_router
from core.agents.communicator import CommunicatorAgent
from core.agents.planner import PlannerAgent
from core.agents.orchestrator import OrchestratorAgent
from core.agents.recovery import RecoveryAgent
from core.critics.critics import LogicCritic, SecurityCritic, RequirementCritic, CriticRunner
from core.models import Task, TaskStatus
from core.observability.logger import get_logger, metrics, tracer

logger = get_logger("api")


# ---------------------------------------------------------------------------
# Dependency container (built once at startup)
# ---------------------------------------------------------------------------

class Container:
    communicator:  CommunicatorAgent
    planner:       PlannerAgent
    orchestrator:  OrchestratorAgent
    task_store:    Dict[str, Task] = {}


container = Container()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build all singletons at startup."""
    logger.info("friday_maf_starting", extra={"provider": settings.LLM_PROVIDER})

    llm     = create_llm_provider(settings.LLM_PROVIDER, settings.LLM_MODEL, settings.LLM_API_KEY)
    memory  = MemoryManager(redis_url=settings.REDIS_URL)
    router  = build_default_router(
        search_api_key = settings.SEARCH_API_KEY,
        github_token   = settings.GITHUB_TOKEN,
    )

    # Critics share the same LLM / memory / event_bus
    critics = CriticRunner([
        LogicCritic(llm, memory, event_bus),
        SecurityCritic(llm, memory, event_bus),
        RequirementCritic(llm, memory, event_bus),
    ])
    recovery = RecoveryAgent(llm, memory, event_bus)

    container.communicator  = CommunicatorAgent(llm, memory, event_bus)
    container.planner       = PlannerAgent(llm, memory, event_bus)
    container.orchestrator  = OrchestratorAgent(llm, memory, event_bus, router, critics, recovery)
    container.task_store    = {}

    logger.info("friday_maf_ready")
    yield
    logger.info("friday_maf_shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "FRIDAY-MAF",
    description = "Multi-Agent Framework API",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    message: str


class TaskResponse(BaseModel):
    task_id: str
    status:  str
    result:  Optional[Any] = None
    error:   Optional[str] = None


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------

async def _run_pipeline(task: Task) -> None:
    """
    Full pipeline: parse → plan → orchestrate → format.
    Runs in the background so the POST /task response is immediate.
    """
    try:
        # 1. Plan
        task = await container.planner.run(task)

        # 2. Execute (orchestrator handles workers + critics + recovery)
        task = await container.orchestrator.run(task)

        # 3. Format for user
        if task.status == TaskStatus.COMPLETED:
            task.final_result = await container.communicator.format_response(task)

    except Exception as exc:
        task.error  = str(exc)
        task.status = TaskStatus.FAILED
        logger.error("pipeline_error", extra={"task_id": task.task_id, "error": str(exc)})
    finally:
        container.task_store[task.task_id] = task


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/task", response_model=TaskResponse, status_code=202)
async def create_task(req: TaskRequest, background_tasks: BackgroundTasks):
    """
    Submit a natural language task.
    Returns a task_id immediately — poll GET /task/{id} for results.
    """
    task = await container.communicator.run(req.message)
    container.task_store[task.task_id] = task
    background_tasks.add_task(_run_pipeline, task)
    logger.info("api_task_submitted", extra={"task_id": task.task_id})
    return TaskResponse(task_id=task.task_id, status=task.status)


@app.get("/task/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Poll the status and result of a submitted task."""
    task = container.task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return TaskResponse(
        task_id = task.task_id,
        status  = task.status,
        result  = task.final_result,
        error   = task.error,
    )


@app.get("/health")
async def health():
    """Liveness + metrics snapshot."""
    return {
        "status":  "ok",
        "version": "1.0.0",
        "metrics": metrics.snapshot(),
        "recent_spans": tracer.recent(10),
    }