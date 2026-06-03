"""
FRIDAY-MAF API Layer
FastAPI application exposing:
  POST /task        — submit a new task
  GET  /task/{id}   — poll task status + result
  GET  /tasks       — list all tasks (with optional status filter)
  GET  /events      — recent event stream (for dashboard)
  GET  /health      — liveness check + metrics snapshot
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
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
from core.models import Task, TaskStatus, Event
from core.observability.logger import get_logger, metrics, tracer
from core.persistence.sqlite_store import SQLiteStore
from core.autonomy.execution_loop import AutonomyLoop

logger = get_logger("api")


# ---------------------------------------------------------------------------
# Dependency container (built once at startup)
# ---------------------------------------------------------------------------

class Container:
    communicator:  CommunicatorAgent
    planner:       PlannerAgent
    orchestrator:  OrchestratorAgent
    task_store:    Dict[str, Task]
    store:         SQLiteStore
    loop:          AutonomyLoop


container = Container()


# ---------------------------------------------------------------------------
# Persist events to SQLite as they fire
# ---------------------------------------------------------------------------

async def _persist_event(event: Event) -> None:
    try:
        await container.store.save_event(event)
    except Exception as exc:
        logger.warning("event_persist_error", extra={"error": str(exc)})


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build all singletons at startup, start autonomy loop, resume tasks."""
    logger.info("friday_maf_starting", extra={"provider": settings.LLM_PROVIDER})

    # 1. Infrastructure
    llm    = create_llm_provider(
        settings.LLM_PROVIDER, settings.LLM_MODEL, settings.LLM_API_KEY,
        base_url=settings.OLLAMA_BASE_URL,
    )
    memory = MemoryManager(redis_url=settings.REDIS_URL)
    router = build_default_router(
        search_api_key=settings.SEARCH_API_KEY,
        github_token=settings.GITHUB_TOKEN,
    )

    # 2. Agents
    critics  = CriticRunner([
        LogicCritic(llm, memory, event_bus),
        SecurityCritic(llm, memory, event_bus),
        RequirementCritic(llm, memory, event_bus),
    ])
    recovery = RecoveryAgent(llm, memory, event_bus)

    container.communicator  = CommunicatorAgent(llm, memory, event_bus)
    container.planner       = PlannerAgent(llm, memory, event_bus)
    container.orchestrator  = OrchestratorAgent(llm, memory, event_bus, router, critics, recovery)
    container.task_store    = {}

    # 3. Persistence
    container.store = SQLiteStore(settings.DB_PATH)

    # 4. Subscribe event bus → SQLite persistence
    for et in Event.__class__.__mro__:
        pass
    from core.models import EventType
    for et in EventType:
        event_bus.subscribe(et, _persist_event)

    # 5. Autonomy loop
    container.loop = AutonomyLoop(
        store        = container.store,
        communicator = container.communicator,
        planner      = container.planner,
        orchestrator = container.orchestrator,
        task_store   = container.task_store,
        interval     = settings.LOOP_INTERVAL,
    )

    # 6. Resume unfinished tasks from previous runs
    resumed = await container.loop.resume_unfinished()
    if resumed:
        logger.info("tasks_resumed", extra={"count": resumed})

    # 7. Start background loop
    await container.loop.start()

    logger.info("friday_maf_ready")
    yield

    # Shutdown
    await container.loop.stop()
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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


class TaskSummary(BaseModel):
    task_id:    str
    status:     str
    goal:       str
    user_input: str
    created_at: str
    updated_at: str
    subtask_count: int
    error:      Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/task", response_model=TaskResponse, status_code=202)
async def create_task(req: TaskRequest):
    """
    Submit a natural language task.
    The autonomy loop picks it up and executes it asynchronously.
    Poll GET /task/{id} for results.
    """
    task = await container.communicator.run(req.message)
    container.task_store[task.task_id] = task
    await container.store.save_task(task)
    logger.info("api_task_submitted", extra={"task_id": task.task_id})
    return TaskResponse(task_id=task.task_id, status=task.status)


@app.get("/task/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Poll the status and result of a submitted task."""
    task = container.task_store.get(task_id)
    if task is None:
        # Try loading from DB (e.g. after restart)
        task = await container.store.load_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return TaskResponse(
        task_id = task.task_id,
        status  = task.status,
        result  = task.final_result,
        error   = task.error,
    )


@app.get("/tasks", response_model=List[TaskSummary])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit:  int            = Query(50, ge=1, le=500),
):
    """List all tasks, optionally filtered by status."""
    all_tasks = await container.store.load_all_tasks()
    # Merge with in-memory (may be more up to date)
    for tid, t in container.task_store.items():
        # replace DB version with live version
        for i, dt in enumerate(all_tasks):
            if dt.task_id == tid:
                all_tasks[i] = t
                break
        else:
            all_tasks.append(t)

    if status:
        all_tasks = [t for t in all_tasks if t.status.value == status.upper()]

    all_tasks.sort(key=lambda t: t.created_at, reverse=True)
    all_tasks = all_tasks[:limit]

    return [
        TaskSummary(
            task_id       = t.task_id,
            status        = t.status.value,
            goal          = t.goal,
            user_input    = t.user_input,
            created_at    = t.created_at.isoformat(),
            updated_at    = t.updated_at.isoformat(),
            subtask_count = len(t.subtasks),
            error         = t.error,
        )
        for t in all_tasks
    ]


@app.get("/task/{task_id}/detail")
async def get_task_detail(task_id: str):
    """Full task detail including subtasks, critic results, events."""
    task = container.task_store.get(task_id)
    if task is None:
        task = await container.store.load_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    events = await container.store.load_events(task_id=task_id, limit=100)

    return {
        "task":    task.model_dump(mode="json"),
        "events":  events,
    }


@app.get("/events")
async def get_events(
    task_id: Optional[str] = Query(None),
    limit:   int           = Query(100, ge=1, le=1000),
):
    """Recent event stream — used by dashboard."""
    events = await container.store.load_events(task_id=task_id, limit=limit)
    # Also include in-memory events not yet persisted
    bus_history = event_bus.history(limit)
    bus_dicts   = [
        {
            "event_id":   e.event_id,
            "task_id":    e.task_id,
            "event_type": e.event_type.value,
            "payload":    e.payload,
            "timestamp":  e.timestamp.isoformat(),
        }
        for e in bus_history
        if task_id is None or e.task_id == task_id
    ]
    # Merge and deduplicate by event_id
    seen      = {e["event_id"] for e in events}
    for e in bus_dicts:
        if e["event_id"] not in seen:
            events.append(e)
            seen.add(e["event_id"])
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events[:limit]


@app.get("/health")
async def health():
    """Liveness + metrics snapshot."""
    stats = await container.store.get_task_stats()
    event_count = await container.store.get_event_count()
    return {
        "status":       "ok",
        "version":      "1.0.0",
        "provider":     settings.LLM_PROVIDER,
        "model":        settings.LLM_MODEL,
        "active_tasks": len([t for t in container.task_store.values()
                              if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)]),
        "loop_running": container.loop._running,
        "task_stats":   stats,
        "event_count":  event_count,
        "metrics":      metrics.snapshot(),
        "recent_spans": tracer.recent(10),
    }
