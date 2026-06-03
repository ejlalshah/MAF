# FRIDAY-MAF — Project State

**Snapshot date:** 2026-06-03  
**Version:** 1.0.0 (MVP complete)  
**Test status:** 40/40 passing  
**Overall readiness:** Development / MVP — not yet production-deployable

---

## 1. Architecture Summary

FRIDAY-MAF is a fully asynchronous multi-agent AI framework exposed via a FastAPI HTTP layer. Agents communicate exclusively through a central event bus (pub/sub); no agent holds a direct reference to another. The pipeline runs in five ordered stages:

```
HTTP POST /task
    └─► CommunicatorAgent   → raw NL → structured Task
    └─► PlannerAgent        → Task.goal → DAG of SubTasks
    └─► OrchestratorAgent   → topological execution of DAG
          ├─► Worker (Research | Coding | DataAnalyst)
          ├─► CriticRunner (Logic + Security + Requirement — parallel)
          └─► RecoveryAgent (on failure: retry | alternate worker | escalate)
    └─► CommunicatorAgent   → Task result → user-facing Markdown
HTTP GET /task/{id}         → poll status + result
HTTP GET /health            → liveness + in-process metrics snapshot
```

The framework is provider-agnostic: OpenAI, Anthropic, and Gemini are all supported through a single `BaseLLMProvider` interface with zero code changes required beyond a settings change.

---

## 2. Implemented Components

### Core Infrastructure

| Component | File | Status |
|---|---|---|
| Pydantic data models + enums | `core/models.py` | ✅ Complete |
| Task lifecycle state machine | `core/state/state_machine.py` | ✅ Complete |
| Async pub/sub event bus | `core/events/event_bus.py` | ✅ Complete |
| Structured logging (JSON) | `core/observability/logger.py` | ✅ Complete |
| In-process metrics counters | `core/observability/logger.py` | ✅ Complete |
| Execution tracer (spans) | `core/observability/logger.py` | ✅ Complete |
| Env-var configuration | `config/settings.py` | ✅ Complete |

### LLM Abstraction

| Component | File | Status |
|---|---|---|
| `BaseLLMProvider` interface | `core/llm/providers.py` | ✅ Complete |
| OpenAI provider (chat + embed + stream) | `core/llm/providers.py` | ✅ Complete |
| Anthropic provider (chat + stream) | `core/llm/providers.py` | ✅ Complete |
| Gemini provider (chat + stream + embed) | `core/llm/providers.py` | ✅ Complete |
| Provider factory | `core/llm/providers.py` | ✅ Complete |

### Memory System

| Component | File | Status |
|---|---|---|
| `BaseMemory` interface | `core/memory/memory_manager.py` | ✅ Complete |
| Short-term memory (Redis / in-process fallback) | `core/memory/memory_manager.py` | ✅ Complete |
| Episodic memory (in-process vector store) | `core/memory/memory_manager.py` | ✅ Complete |
| Semantic memory (knowledge base) | `core/memory/memory_manager.py` | ✅ Complete |
| `MemoryManager` facade | `core/memory/memory_manager.py` | ✅ Complete |

### Tool System

| Component | File | Status |
|---|---|---|
| `BaseTool` + `ToolResult` | `core/tools/tool_router.py` | ✅ Complete |
| `ToolRouter` (permission enforcement) | `core/tools/tool_router.py` | ✅ Complete |
| `WebSearchTool` | `core/tools/tool_router.py` | ⚠️ Stub (returns mock results) |
| `FileTool` | `core/tools/tool_router.py` | ✅ Real (local filesystem) |
| `PythonExecutionTool` | `core/tools/tool_router.py` | ✅ Real (subprocess) |
| `GitHubTool` | `core/tools/tool_router.py` | ✅ Real (GitHub REST API) |
| Permission matrix (`AGENT_PERMISSIONS`) | `core/tools/tool_router.py` | ✅ Complete |

### Agents

| Component | File | Status |
|---|---|---|
| `BaseAgent` (LLM + memory + events) | `core/agents/base_agent.py` | ✅ Complete |
| `CommunicatorAgent` (parse + format) | `core/agents/communicator.py` | ✅ Complete |
| `PlannerAgent` (goal → DAG) | `core/agents/planner.py` | ✅ Complete |
| `OrchestratorAgent` (DAG execution engine) | `core/agents/orchestrator.py` | ✅ Complete |
| `RecoveryAgent` (failure handler) | `core/agents/recovery.py` | ✅ Complete |

### Workers

| Component | File | Status |
|---|---|---|
| `BaseWorker` (tool-calling agent base) | `core/workers/workers.py` | ✅ Complete |
| `ResearchAgent` | `core/workers/workers.py` | ✅ Complete |
| `CodingAgent` | `core/workers/workers.py` | ✅ Complete |
| `DataAnalystAgent` | `core/workers/workers.py` | ✅ Complete |

### Critics

| Component | File | Status |
|---|---|---|
| `BaseCritic` | `core/critics/critics.py` | ✅ Complete |
| `LogicCritic` (factual correctness) | `core/critics/critics.py` | ✅ Complete |
| `SecurityCritic` (vulnerability review) | `core/critics/critics.py` | ✅ Complete |
| `RequirementCritic` (spec compliance) | `core/critics/critics.py` | ✅ Complete |
| `CriticRunner` (parallel orchestration) | `core/critics/critics.py` | ✅ Complete |

### API Layer

| Component | File | Status |
|---|---|---|
| FastAPI app + lifespan wiring | `api/main.py` | ✅ Complete |
| `POST /task` (async submission) | `api/main.py` | ✅ Complete |
| `GET /task/{id}` (polling) | `api/main.py` | ✅ Complete |
| `GET /health` (liveness + metrics) | `api/main.py` | ✅ Complete |

### Test Suite

| Suite | File | Tests | Status |
|---|---|---|---|
| Models | `tests/test_all.py` | 5 | ✅ Pass |
| State machine | `tests/test_all.py` | 5 | ✅ Pass |
| Event bus | `tests/test_all.py` | 4 | ✅ Pass |
| Memory (3 tiers) | `tests/test_all.py` | 7 | ✅ Pass |
| Tool router | `tests/test_all.py` | 5 | ✅ Pass |
| Critics | `tests/test_all.py` | 4 | ✅ Pass |
| Communicator agent | `tests/test_all.py` | 3 | ✅ Pass |
| Memory manager (integration) | `tests/test_all.py` | 3 | ✅ Pass |
| Full pipeline (mocked LLM) | `tests/test_all.py` | 4 | ✅ Pass |
| **Total** | | **40** | **✅ 40/40** |

---

## 3. Mock vs. Production-Ready Classification

### ✅ Production-ready (real, tested, deployable as-is)

- State machine with infinite-loop guard
- Event bus (in-process; swap surface is clearly defined)
- Tool permission enforcement via `ToolRouter`
- `FileTool` and `PythonExecutionTool`
- `GitHubTool` (real GitHub REST API calls)
- All three LLM providers (OpenAI, Anthropic, Gemini) — real HTTP, not mocked
- JSON-structured logging (Prometheus-ingestible format)
- Execution tracer with span metadata
- `ShortTermMemory` Redis path (graceful fallback included)
- Pydantic v2 models + serialization
- FastAPI app, lifespan wiring, background task runner

### ⚠️ Stub / dev-only (requires replacement before production)

| Component | Issue | Replacement Path |
|---|---|---|
| `WebSearchTool` | Returns hardcoded placeholder results | Integrate Tavily / Brave / SerpAPI |
| `AnthropicProvider.embed()` | Returns a zero vector (Anthropic has no public embedding endpoint) | Route to OpenAI `text-embedding-3-small` or Cohere |
| `EpisodicMemory` cosine store | In-process only; lost on restart | Drop in ChromaDB, Pinecone, or Weaviate |
| `SemanticMemory` | In-process key-value, no persistence | Same vector DB upgrade path as episodic |
| `container.task_store` | Plain Python `dict`; lost on restart | Redis hash or Postgres `tasks` table |
| Metrics (`MetricsStore`) | In-process counters only | Expose via `/metrics` with `prometheus-client` |
| `Tracer` | In-process span buffer (last N spans only) | Export to Jaeger / OpenTelemetry |
| `PythonExecutionTool` sandbox | No resource limits, runs as host process | Docker exec or gVisor sandbox |
| Auth on `/task` | No authentication | API key middleware or OAuth2 |
| Rate limiting | Not implemented | `slowapi` or upstream nginx |

---

## 4. Known Issues / Technical Debt

| # | Severity | Location | Description |
|---|---|---|---|
| 1 | Medium | `models.py` | `SubTask.dependancies` is misspelled (should be `dependencies`). Field name is correct elsewhere as `dependencies`. This is a latent source of confusion and serialization bugs. |
| 2 | Low | `orchestrator.py` | `container.task_store` is a module-level `dict` — concurrent requests share state without locking. Safe under uvicorn's single-worker async loop but will corrupt under multi-process deployment. |
| 3 | Low | `providers.py` | `OpenAIProvider` uses deprecated `m.dict()` (Pydantic v2 should be `m.model_dump()`). Will emit deprecation warnings; non-breaking for now. |
| 4 | Low | `recovery.py` | `RecoveryAgent.run()` forces escalation after 2 retries, but `OrchestratorAgent._execute_subtask()` independently tracks `retry_count` with its own cap of 3. The two counters are not shared — edge-case retry counts may be off by one. |
| 5 | Low | `planner.py` | `_execution_order()` is defined but never called by `PlannerAgent.run()`. The Orchestrator performs its own topological sort. Dead code. |
| 6 | Info | All providers | Streaming (`stream()`) is implemented but never called from any agent path. Currently unused infrastructure. |