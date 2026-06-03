# FRIDAY-MAF — Repository Handoff

> This document is the single source of truth for onboarding a new developer, resuming work after a break, or handing the project to another team. Read it top-to-bottom before touching any code.

---

## What This Project Is

FRIDAY-MAF (Multi-Agent Framework) is a production-grade, fully async Python framework for orchestrating multiple LLM-powered agents to complete complex tasks. Users submit a natural-language task via HTTP; the system plans, executes, critiques, and returns a result.

It supports OpenAI, Anthropic, and Gemini interchangeably through a unified interface. Switching providers requires only a settings change.

**Current state:** MVP complete. 40/40 tests passing. Not yet production-deployable (no auth, no persistent storage, stub web search). See `PROJECT_STATE.md` for the full component inventory and `ROADMAP.md` for what to build next.

---

## Repository Layout

```
friday_maf/
│
├── api/
│   └── main.py                 ← FastAPI app, routes, startup wiring
│
├── config/
│   └── settings.py             ← All config via env vars (pydantic-settings)
│
├── core/
│   ├── models.py               ← ALL shared Pydantic models and enums — read this first
│   │
│   ├── agents/
│   │   ├── base_agent.py       ← BaseAgent: LLM + memory + event bus + tracing
│   │   ├── communicator.py     ← NL input → Task; Task result → Markdown
│   │   ├── planner.py          ← Task.goal → DAG of SubTasks
│   │   ├── orchestrator.py     ← Execution engine: DAG → workers → critics → recovery
│   │   └── recovery.py         ← Failure handler: retry / fallback / escalate
│   │
│   ├── workers/
│   │   └── workers.py          ← BaseWorker + Research + Coding + DataAnalyst
│   │
│   ├── critics/
│   │   └── critics.py          ← Logic + Security + Requirement critics + CriticRunner
│   │
│   ├── memory/
│   │   └── memory_manager.py   ← Short-term (Redis) + Episodic (vector) + Semantic (KB)
│   │
│   ├── tools/
│   │   └── tool_router.py      ← Tools + permission enforcement
│   │
│   ├── llm/
│   │   └── providers.py        ← OpenAI / Anthropic / Gemini providers
│   │
│   ├── state/
│   │   └── state_machine.py    ← Task lifecycle state machine
│   │
│   ├── events/
│   │   └── event_bus.py        ← Async pub/sub
│   │
│   └── observability/
│       └── logger.py           ← Structured logging + metrics + tracing
│
└── tests/
    └── test_all.py             ← 40 unit tests (all passing)
```

---

## How to Run

### Prerequisites

- Python 3.11+
- Redis (optional — system falls back to in-process dict without it)

### Install

```bash
pip install -r requirements.txt
```

### Configure

Create `.env` in the project root (copy and edit):

```bash
# Required
LLM_API_KEY=sk-...           # Your API key
LLM_PROVIDER=anthropic       # openai | anthropic | gemini
LLM_MODEL=claude-3-5-haiku-20241022

# Optional
REDIS_URL=redis://localhost:6379/0
SEARCH_API_KEY=              # Not yet wired to a real provider — leave blank
GITHUB_TOKEN=                # Only needed if using GitHubTool

LOG_LEVEL=DEBUG
```

### Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run Tests

```bash
pytest tests/ -v
```

All 40 tests run without a real LLM key or Redis — the LLM is mocked and Redis gracefully degrades to in-process fallback.

---

## Request Lifecycle (Step-by-Step)

Understanding this flow is the most important thing for working on this codebase.

```
1.  Client sends:  POST /task  {"message": "Research quantum computing"}
2.  API creates a BackgroundTask and returns 202 immediately with {task_id}

3.  CommunicatorAgent.run(message)
    └─ Calls LLM: "Parse this user request" → JSON
    └─ Creates Task(goal=..., metadata=...)
    └─ Stores task state in MemoryManager (short-term)
    └─ Publishes TASK_CREATED event

4.  PlannerAgent.run(task)
    └─ Recalls similar past episodes from EpisodicMemory
    └─ Calls LLM: "Decompose this goal" → JSON array of subtasks
    └─ Creates SubTask objects with dependency links
    └─ Publishes PLAN_COMPLETED event

5.  OrchestratorAgent.run(task)
    └─ Topological sort → parallel batches
    └─ For each SubTask (concurrently within a batch):
        a. Dispatches to correct Worker (Research / Coding / DataAnalyst)
        b. Worker calls tools (web_search, python_exec, file, github)
        c. Worker returns WorkerOutput
        d. If worker failed: RecoveryAgent decides retry|alternative|escalate
        e. If worker succeeded: CriticRunner.evaluate() in parallel
           - LogicCritic checks correctness
           - SecurityCritic checks for vulnerabilities
           - RequirementCritic checks spec compliance
        f. If any critic failed: inject feedback → retry subtask (max 3x)
        g. If all critics passed: SubTask.status = COMPLETED
    └─ Assembles final result from all subtask results
    └─ Stores episode summary in EpisodicMemory for future planning
    └─ Publishes TASK_COMPLETED event

6.  CommunicatorAgent.format_response(task)
    └─ Calls LLM: "Format this result for the user" → Markdown string
    └─ task.final_result = formatted string

7.  Client polls:  GET /task/{task_id}
    └─ Returns {task_id, status: "COMPLETED", result: "..."}
```

---

## Key Design Patterns

### 1. Agents never reference each other directly
All inter-agent communication flows through `event_bus.publish()`. The `OrchestratorAgent` is the only exception — it holds direct references to workers, critics, and recovery because it is the execution coordinator, not a peer.

### 2. Every LLM call goes through `BaseAgent.think()`
This is intentional. `think()` handles: tracing, token counting, JSON-mode injection, context appending. Never call `self.llm.generate()` directly from a concrete agent.

### 3. Tools are called through `ToolRouter`, never directly
`ToolRouter.call()` enforces the permission matrix. If you add a new tool or a new agent type, update `AGENT_PERMISSIONS` in `tool_router.py` — that's the only place permissions live.

### 4. Workers always return `WorkerOutput`, never raise
`BaseWorker.run()` catches all exceptions and converts them to `WorkerOutput(success=False, error=...)`. The Orchestrator handles failure through the recovery loop, not through exception handling.

### 5. State transitions are enforced by `TaskStateMachine`
Never mutate `task.status` directly without also calling `sm.transition(new_state)`. Invalid transitions raise `StateMachineError` immediately.

---

## Adding a New Worker

This is the most common extension point.

```python
# 1. In core/workers/workers.py:
class SummaryAgent(BaseWorker):
    agent_type    = AgentType.RESEARCH   # reuse or add a new AgentType enum value
    system_prompt = "You summarise documents clearly and concisely..."

    async def execute(self, subtask: SubTask, context: Optional[str]) -> Any:
        content = await self._call_tool("file", operation="read", path=subtask.task_description)
        return await self.think_json(f"Summarise this document:\n\n{content}")

# 2. In core/tools/tool_router.py — AGENT_PERMISSIONS:
AgentType.RESEARCH: {"web_search", "file", "summarise"},   # if adding a new tool

# 3. In core/agents/orchestrator.py — OrchestratorAgent.__init__:
self._workers[AgentType.RESEARCH] = SummaryAgent(llm, memory, event_bus, tool_router)
# (or add a new AgentType key if it's a distinct type)
```

---

## Adding a New LLM Provider

```python
# 1. In core/llm/providers.py:
class MistralProvider(BaseLLMProvider):
    BASE_URL = "https://api.mistral.ai/v1"

    async def generate(self, messages, max_tokens=2048, temperature=0.7) -> LLMResponse:
        ...   # implement HTTP call

    async def stream(self, messages, max_tokens=2048, temperature=0.7):
        ...

    async def embed(self, text: str) -> List[float]:
        ...

# 2. Register in _PROVIDERS:
_PROVIDERS = {
    "openai":    OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini":    GeminiProvider,
    "mistral":   MistralProvider,   # ← add here
}

# 3. In .env:
LLM_PROVIDER=mistral
LLM_MODEL=mistral-large-latest
```

No other changes needed.

---

## Adding a New Event Type

```python
# 1. In core/models.py — EventType enum:
class EventType(str, Enum):
    ...
    TOOL_TIMEOUT = "TOOL_TIMEOUT"   # new

# 2. Subscribe anywhere:
event_bus.subscribe(EventType.TOOL_TIMEOUT, my_async_handler)

# 3. Publish from any agent:
await self.emit(EventType.TOOL_TIMEOUT, task_id=subtask.task_id, tool="web_search")
```

---

## Configuration Reference

| Variable | Default | Required | Description |
|---|---|---|---|
| `LLM_PROVIDER` | `anthropic` | No | `openai`, `anthropic`, or `gemini` |
| `LLM_MODEL` | `claude-3-5-haiku-20241022` | No | Model name for the chosen provider |
| `LLM_API_KEY` | `""` | **Yes** | API key for the LLM provider |
| `EMBED_MODEL` | `text-embedding-3-small` | No | Embedding model (currently only used by OpenAI provider) |
| `REDIS_URL` | `redis://localhost:6379/0` | No | Redis connection string; falls back to in-process dict |
| `SEARCH_API_KEY` | `None` | No | API key for web search (stub until real provider is wired) |
| `GITHUB_TOKEN` | `None` | No | GitHub personal access token for `GitHubTool` |
| `API_HOST` | `0.0.0.0` | No | uvicorn bind host |
| `API_PORT` | `8000` | No | uvicorn bind port |
| `LOG_LEVEL` | `DEBUG` | No | Python logging level; use `INFO` in production |
| `MAX_RETRIES` | `3` | No | Max retry attempts per subtask |
| `TASK_TIMEOUT` | `300` | No | Pipeline timeout in seconds (currently unenforced — see Roadmap M5.2) |

---

## What NOT to Do

- **Do not call `self.llm.generate()` directly from a concrete agent.** Always use `self.think()` or `self.think_json()`. The trace spans and token metrics will be missing otherwise.
- **Do not add tool calls to Communicator, Planner, or Orchestrator.** Their `AGENT_PERMISSIONS` entries are deliberately empty. They reason; workers act.
- **Do not mutate `task.status` without calling `sm.transition()`.** The state machine is the authoritative source of state — bypassing it silently corrupts the task lifecycle.
- **Do not store secrets in `.env` and commit it.** The `.env` file is for local development only. Use a secrets manager (AWS Secrets Manager, HashiCorp Vault, Doppler) in any deployed environment.
- **Do not add synchronous blocking I/O.** The entire framework is async. A single `time.sleep()` or synchronous `requests.get()` inside an agent will block the event loop and degrade all concurrent tasks.

---

## Open Questions for the Next Developer

1. **Task store persistence:** Should the task store be Redis (low latency, simple) or Postgres (queryable history, joins)? The decision affects Milestone 1.1 significantly.

2. **Embedding provider:** `AnthropicProvider.embed()` returns a zero vector. Should the system always use OpenAI for embeddings regardless of the chat provider, or should a separate `EMBED_PROVIDER` setting be introduced?

3. **Streaming:** `BaseLLMProvider.stream()` is fully implemented in all three providers but is never called from any agent path. Is streaming to the client (SSE on `GET /task/{id}`) a desired feature? If not, the stream methods can be removed to reduce surface area.

4. **`PlannerAgent._execution_order()`:** Dead code. Remove it or consolidate with `OrchestratorAgent._topological_batches()`?

5. **Multi-tenancy:** Is this a single-user tool or a multi-tenant SaaS? The memory namespacing and auth strategy are very different for each. Decide before building Milestone 3 to avoid rework.

---

## Emergency Contacts / Resources

- Architecture diagram: `docs/friday_maf_architecture.svg`
- API reference: `docs/api_reference.md`
- Architecture deep-dive: `docs/architecture.md`
- Test suite: `tests/test_all.py` (run first when something breaks — 40 tests, ~5s total)