# FRIDAY-MAF — Roadmap: MVP → Production

> Current state: MVP complete, 40/40 tests passing, no production infrastructure.  
> This roadmap is sequenced by risk and dependency order. Each milestone is independently deployable.

---

## Milestone 0 — Pre-work: Fix Known Technical Debt
**Effort:** 1–2 days | **Risk:** Low | **Blocks:** Everything else

These are correctness fixes that should be made before any new features land on top of them.

### 0.1 Fix the `dependancies` typo in `SubTask`
`models.py` line: `dependancies: List[str]` — misspelled. Every other reference in the codebase uses `dependencies`.

```python
# models.py — SubTask
dependencies: List[str] = Field(default_factory=list)   # was: dependancies
```

Add a migration note for any serialised data in Redis that was stored with the old key name.

### 0.2 Fix Pydantic v2 deprecation in `OpenAIProvider`
Replace `m.dict()` with `m.model_dump()` in `providers.py`. One-line fix; prevents deprecation warnings from polluting logs.

### 0.3 Remove dead code in `PlannerAgent`
`_execution_order()` in `planner.py` is never called. Remove it or consolidate with the Orchestrator's `_topological_batches()`.

### 0.4 Unify retry counters between `RecoveryAgent` and `OrchestratorAgent`
Pass `subtask.retry_count` as the single source of truth. The Orchestrator's local `retry_count` variable and the subtask's `retry_count` field should be the same object.

---

## Milestone 1 — Persistence Layer
**Effort:** 3–5 days | **Risk:** Medium | **Blocks:** Milestones 3, 4, 5

In production, task state and episodic memory must survive process restarts.

### 1.1 Persistent task store
Replace `container.task_store: Dict[str, Task]` with a Redis hash or Postgres `tasks` table.

```
Recommended: Redis sorted set for task IDs + Redis hash per task.
Schema: HSET task:{task_id} status EXECUTING goal "..." ...
```

Expose `TaskRepository` with `save(task)`, `get(task_id)`, `list_recent(n)`.

### 1.2 Real vector store for episodic memory
Replace `EpisodicMemory`'s in-process cosine store with ChromaDB (self-hosted) or Pinecone (managed).

```python
# memory_manager.py — MemoryManager.__init__
# Before:
self.episodic = EpisodicMemory(embed_fn=embed_fn)
# After:
self.episodic = ChromaDBEpisodicMemory(collection="friday_episodes", embed_fn=embed_fn)
```

`ChromaDBEpisodicMemory` implements `BaseMemory` — no agent code changes needed.

### 1.3 Embedding provider for episodic memory
`AnthropicProvider.embed()` returns a zero vector. Wire `MemoryManager` to use `OpenAIProvider.embed()` (or Cohere) regardless of which chat provider is configured.

```python
# settings.py
EMBED_PROVIDER: str = "openai"   # always openai, even when chat is anthropic
EMBED_MODEL:    str = "text-embedding-3-small"
```

---

## Milestone 2 — Real Web Search
**Effort:** 1–2 days | **Risk:** Low | **Blocks:** ResearchAgent, DataAnalystAgent

`WebSearchTool` currently returns hardcoded stub results. Replace with a real provider.

### 2.1 Integrate Tavily (recommended) or Brave Search

```python
class TavilySearchTool(BaseTool):
    name = "web_search"
    
    async def run(self, query: str, num_results: int = 5) -> ToolResult:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self._api_key, "query": query, "max_results": num_results},
            )
            r.raise_for_status()
            return ToolResult.ok(r.json()["results"])
```

Drop `WebSearchTool` and register `TavilySearchTool` in `build_default_router()`. No permission map changes needed — the tool name stays `"web_search"`.

---

## Milestone 3 — Security & Access Control
**Effort:** 3–5 days | **Risk:** High (do not skip) | **Blocks:** Production deployment

### 3.1 API key authentication
Add a FastAPI dependency that validates `Authorization: Bearer <key>` on all routes except `/health`.

```python
async def require_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in settings.API_KEYS:
        raise HTTPException(status_code=401)
```

Store `API_KEYS` as a comma-separated env var. Rotate without downtime.

### 3.2 Rate limiting
Add `slowapi` (Starlette-compatible `limits` wrapper) to `/task`. Start at 60 requests/minute/key.

### 3.3 Python execution sandbox
`PythonExecutionTool` currently runs code as the host process with no resource constraints. Replace with Docker exec (`docker run --rm --network none --memory 256m --cpus 0.5`) or Firecracker microVMs.

At minimum, add `ulimit` and `seccomp` restrictions if Docker is not available.

### 3.4 Input validation and prompt injection guard
Validate `TaskRequest.message` length (max 4000 chars). Add a lightweight prompt injection filter in `CommunicatorAgent.run()` that rejects messages containing obvious injection patterns (`ignore previous instructions`, `system:`, etc.).

---

## Milestone 4 — Observability Stack
**Effort:** 2–3 days | **Risk:** Low | **Blocks:** Milestone 5 (scaling)

### 4.1 Prometheus metrics endpoint
Replace `MetricsStore` with `prometheus-client` and expose `/metrics`.

```python
from prometheus_client import Counter, Histogram, make_asgi_app
# Mount at app startup:
app.mount("/metrics", make_asgi_app())
```

Map existing `metrics.increment(...)` and `metrics.observe(...)` calls to Prometheus `Counter` and `Histogram` objects. Call-site API is identical.

### 4.2 OpenTelemetry traces
Replace `Tracer` with `opentelemetry-sdk`. Export to Jaeger or Honeycomb.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
```

The `async with tracer.trace(name, ...)` interface is preserved — only the implementation changes.

### 4.3 Structured log shipping
Current `JSONFormatter` output is already structured. Add a log shipper sidecar (Fluent Bit or Vector) to forward to Loki, Datadog, or CloudWatch. No code changes needed.

---

## Milestone 5 — Scaling & Reliability
**Effort:** 5–8 days | **Risk:** Medium | **Blocks:** None (can run in parallel with M3/M4)

### 5.1 Move pipeline execution off the HTTP worker
`BackgroundTasks` in FastAPI runs in the same process as the HTTP server. Under load, long-running pipelines degrade API responsiveness. Replace with a Celery/ARQ/TaskIQ worker queue.

```
POST /task → enqueue job → return task_id
Worker process → dequeue → run pipeline → store result in Redis
GET /task/{id} → read from Redis
```

This also enables horizontal scaling of the worker pool independently of the API tier.

### 5.2 Task timeout enforcement
`settings.TASK_TIMEOUT = 300` exists but is never used. Wrap `_run_pipeline` in `asyncio.wait_for(..., timeout=settings.TASK_TIMEOUT)`.

### 5.3 Concurrency-safe task store
With multiple API processes, the current `dict`-based task store is unsafe. After Milestone 1.1, the Redis task store handles this automatically via atomic Redis operations.

### 5.4 Health check depth
`GET /health` currently returns `"status": "ok"` unconditionally. Add real checks:
- Redis ping (`ShortTermMemory._init_redis()`)
- LLM provider reachability (lightweight probe)
- Return `503` if critical dependencies are down

---

## Milestone 6 — New Worker Types
**Effort:** 2–3 days per worker | **Risk:** Low

The framework is designed for extension. Adding a worker requires:
1. Subclass `BaseWorker`
2. Define `agent_type` and `system_prompt`
3. Implement `execute(subtask, context) → Any`
4. Register in `OrchestratorAgent._workers`
5. Add tool permissions in `AGENT_PERMISSIONS`

### Suggested workers

| Worker | Agent Type | Tools | Use case |
|---|---|---|---|
| `SummaryAgent` | `RESEARCH` (reuse) | `file` | Summarise uploaded documents |
| `ImageAgent` | New: `IMAGE` | Custom image tool | Generate or analyse images |
| `DatabaseAgent` | New: `DATABASE` | Custom SQL tool | Query structured data |
| `EmailAgent` | New: `COMMS` | Custom email tool | Draft and send emails |
| `BrowserAgent` | New: `BROWSER` | Playwright tool | Automate web interactions |

---

## Milestone 7 — Developer Experience
**Effort:** 3–4 days | **Risk:** Low | **Blocks:** None

### 7.1 Docker Compose for local development
```yaml
services:
  api:       # uvicorn api.main:app
  worker:    # celery/arq worker (post-M5)
  redis:     # redis:7-alpine
  chromadb:  # chromadb/chroma (post-M1.2)
  jaeger:    # jaegertracing/all-in-one (post-M4.2)
```

### 7.2 OpenAPI documentation
FastAPI generates `/docs` automatically. Add response examples and error schemas to all routes.

### 7.3 SDK client
A thin Python client wrapping the HTTP API for embedding in other projects:
```python
client = FridayClient(base_url="http://...", api_key="...")
result = await client.run("Summarise this document", poll_interval=2)
```

### 7.4 Expand test coverage
Current gaps:
- No test for `OrchestratorAgent` end-to-end (worker + critics + recovery path)
- No test for `RecoveryAgent` retry/escalation logic
- No test for `WebSearchTool`, `GitHubTool`
- No integration test against a real LLM (opt-in, skipped in CI)
- No load/concurrency test

Target: 80% line coverage.

---

## Milestone 8 — Multi-Tenancy (Optional / Long-term)
**Effort:** 2–3 weeks | **Risk:** High

For serving multiple users or organisations from a single deployment:
- Scope all `MemoryManager` keys by `tenant_id`
- Per-tenant LLM provider configuration
- Per-tenant tool permission overrides
- Billing / usage tracking per tenant

---

## Priority Matrix

| Milestone | Priority | Effort | Required for Production |
|---|---|---|---|
| M0 — Debt cleanup | P0 | Small | Yes |
| M1 — Persistence | P0 | Medium | Yes |
| M2 — Real web search | P1 | Small | Yes |
| M3 — Security | P0 | Medium | Yes — do not skip |
| M4 — Observability | P1 | Small | Strongly recommended |
| M5 — Scaling | P1 | Large | Depends on load |
| M6 — New workers | P2 | Small per worker | No |
| M7 — DX | P2 | Medium | No |
| M8 — Multi-tenancy | P3 | Large | No |

### Recommended first sprint (2 weeks to beta-deployable)

```
Week 1:  M0 (all items) + M3.1 + M3.2 + M2.1
Week 2:  M1.1 (Redis task store) + M1.3 (embed provider) + M4.1 (Prometheus) + M5.2 (timeout)
```

This produces a system that is: authenticated, rate-limited, using real search, persisting task state, and emitting metrics — the minimum bar for a beta deployment behind a private endpoint.