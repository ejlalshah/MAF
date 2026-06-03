# FRIDAY-MAF — Development Log

> This log is reconstructed from the repository's final state, code comments, test names, and commit-level notes recorded during the MVP phase. It serves as an auditable record of decisions made and problems solved.

---

## Phase 0 — Project Scaffolding

**Goal:** Establish module layout, shared models, and configuration before writing any agent logic.

### Decisions made

- Chose FastAPI for the HTTP layer: async-native, Pydantic v2 out of the box, excellent OpenAPI docs generation.
- All shared data structures centralised in `core/models.py` so no circular imports exist between agent modules.
- `pydantic-settings` used for config (`config/settings.py`); all tunable values are env-vars with safe defaults. No hardcoded secrets anywhere in the codebase.
- `from __future__ import annotations` applied to every module to keep forward-reference typing clean across Python 3.9+.

### Outcome

Core enums (`TaskStatus`, `AgentType`, `EventType`, `CriticType`) and models (`Task`, `SubTask`, `WorkerOutput`, `CriticResult`, `Event`, `LLMMessage`, `LLMResponse`) locked in. Downstream modules depend only on these.

---

## Phase 1 — Infrastructure Layer

**Goal:** Build the foundations everything else depends on: observability, state machine, event bus, memory, tools.

### Observability (`core/observability/logger.py`)

- `JSONFormatter` emits structured log lines (one JSON object per line) to make logs ingestible by any log aggregator without a parsing pipeline.
- `MetricsStore` is an in-process counter/histogram store. Intentionally simple — the public API (`increment`, `observe`, `snapshot`) mirrors Prometheus conventions so swapping to `prometheus-client` later requires no call-site changes.
- `Tracer` uses an async context manager (`async with tracer.trace(...)`) so spans automatically record duration and status even on exceptions. The `span_name` key was introduced to avoid a `KeyError` collision with Python's reserved `LogRecord.name` field — this was the first significant bug fixed.

### State Machine (`core/state/state_machine.py`)

- Explicit adjacency map (`TRANSITIONS`) rather than imperative if-chains: every valid transition is visible in one place, and invalid ones raise `StateMachineError` immediately.
- `MAX_RETRIES = 3` guard in `transition()`: on the third call to `RETRYING`, the machine forces `FAILED` instead. This prevents infinite retry loops without requiring any caller to track retry counts externally.

  **Bug fixed during this phase:** The initial retry guard used `> MAX_RETRIES` (strictly greater), meaning four retries were possible. Changed to `>= MAX_RETRIES` so the third attempt is the last. Test `test_retry_limit_forces_failed` codifies this behaviour.

### Event Bus (`core/events/event_bus.py`)

- Agents never reference each other directly — all inter-agent communication flows through `event_bus.publish()`. This makes the system testable (mock the bus) and extensible (new agents subscribe without touching existing code).
- `asyncio.gather(..., return_exceptions=True)` in `publish()` ensures a failing handler never blocks sibling handlers. Errors are logged and counted but swallowed.
- **Bug fixed during this phase:** Initial implementation used `asyncio.get_event_loop().run_until_complete()` in tests, which was removed in Python 3.12. Tests migrated to `@pytest.mark.asyncio` + true async test functions.

### Memory (`core/memory/memory_manager.py`)

- Three-tier design (short-term / episodic / semantic) with a common `BaseMemory` interface. `MemoryManager` is the single façade agents use — they are unaware of which tier handles a given operation.
- `ShortTermMemory` tries Redis on first use and falls back silently to an in-process `dict`. The fallback makes local development possible without Docker.
- **Bug fixed during this phase:** `json.dumps(value, default=str)` was added to `ShortTermMemory.store()` after `datetime` objects inside serialised `Task` models raised `TypeError`. The `default=str` serialiser handles `datetime`, `UUID`, and `Enum` values safely.
- `EpisodicMemory` uses character-trigram vectors as a dev-friendly embedding fallback. The design accepts an external `embed_fn` so upgrading to real embeddings (OpenAI, Cohere) is a one-line change to `MemoryManager.__init__`.

### Tool System (`core/tools/tool_router.py`)

- `ToolResult` extends `dict` (rather than being a separate Pydantic model) so callers can use both dict-access (`result["success"]`) and attribute-access patterns interchangeably.
- Permission enforcement lives entirely in `ToolRouter.call()` using `AGENT_PERMISSIONS` — a plain dict mapping `AgentType` → `Set[str]`. Centralised, auditable, and tested.
- `PythonExecutionTool` uses a temp file + `asyncio.create_subprocess_exec` rather than `eval()`/`exec()` to prevent interpreter-level leakage. A 30-second timeout kills runaway processes.

---

## Phase 2 — Agent Layer

**Goal:** Implement all agents from `BaseAgent` up through the full pipeline.

### BaseAgent (`core/agents/base_agent.py`)

- `think()` is the only LLM call site. Every agent goes through it, so observability (tracing, token counting) and JSON-mode forcing are handled once, not in every agent.
- `think_json()` strips markdown code fences (` ```json ` … ` ``` `) before parsing, since some LLMs add them despite JSON-mode instructions.

### CommunicatorAgent (`core/agents/communicator.py`)

- Strictly input/output boundary: parses NL → `Task` on ingress, formats `Task` result → Markdown on egress. Never exposes internal agent names, task IDs, or stack traces to users.
- On error, `format_response()` short-circuits to a canned error message rather than passing raw exception text to the LLM.

### PlannerAgent (`core/agents/planner.py`)

- Prompts the LLM for a JSON array of subtasks with `dependencies` expressed as array indices (0-based). These are mapped to actual UUIDs during deserialization so the LLM never needs to generate UUIDs.
- Semantic recall from `EpisodicMemory` is injected into the planner prompt as context, allowing the planner to learn from past executions over time.
- **Dead code identified:** `_execution_order()` in the Planner duplicates the same topological sort already in the Orchestrator. It was written speculatively but is never called by `run()`. Candidate for removal.

### OrchestratorAgent (`core/agents/orchestrator.py`)

- `_topological_batches()` produces parallel execution batches: subtasks with no unmet dependencies run concurrently via `asyncio.gather`. The system is naturally parallelised without any thread management.
- The retry/critic loop in `_execute_subtask()` injects critic feedback into `context_str` for the next attempt, giving workers specific, actionable improvement instructions rather than blind retries.
- Critic feedback injection: when `retry_count >= 3`, a `RuntimeError` is raised to exit the loop. This surfaces to the batch-level handler which marks the whole task `FAILED`.

### RecoveryAgent (`core/agents/recovery.py`)

- **Bug fixed during this phase:** Recovery and the Orchestrator independently cap retries (`retry_count >= 2` in Recovery, `>= 3` in Orchestrator). The Recovery guard was intentionally set lower (2) so the LLM cannot vote for a fourth retry; however, the two counters are not shared, creating an edge case where retry counts diverge. Documented as technical debt #4.
- `FALLBACK_AGENT` map defines alternative agents for each primary agent. `CODING → CODING` (no fallback) is explicit rather than a missing key, so the intent is clear.

### Workers (`core/workers/workers.py`)

- `BaseWorker.run()` wraps `execute()` in a try/except and always returns a `WorkerOutput`, never raises. This means the Orchestrator never needs to handle raw exceptions from workers — only structured `WorkerOutput(success=False, ...)`.
- `_call_tool()` raises `RuntimeError` on tool failure, which is caught by `BaseWorker.run()` and converted to a failed `WorkerOutput`. Tool permission denials are handled the same way as tool execution errors.

### Critics (`core/critics/critics.py`)

- Each critic is stateless and independently testable. `agent_type = AgentType.COMMUNICATOR` is reused for LLM access (critics don't need any tools).
- The `PASS_THRESHOLD` override in `BaseCritic.run()`: even if the LLM returns `"pass": true`, a score below the threshold forces a failure. This prevents a lazy LLM from rubber-stamping low-quality outputs.
- `SecurityCritic.PASS_THRESHOLD = 50` (lower than the default 60): most tasks are not security-sensitive, so this avoids false positives that would cause unnecessary retries.

---

## Phase 3 — API Layer & Integration

**Goal:** Wire all components together under FastAPI and prove end-to-end correctness with tests.

### API Wiring (`api/main.py`)

- `lifespan` context manager builds all singletons exactly once at startup and holds them in a `Container` dataclass. FastAPI's dependency injection is not used — the container is simpler and sufficient for this architecture.
- `POST /task` returns `202 Accepted` immediately. The pipeline runs as a `BackgroundTask`. Clients poll `GET /task/{id}`.
- **Bug fixed during this phase:** `task.model_dump(mode="json")` (Pydantic v2) replaced the earlier `.dict()` call after datetime serialization to Redis raised `TypeError`. The `mode="json"` flag forces all field types to JSON-safe primitives before handing off to `json.dumps`.

### Test Suite (`tests/test_all.py`)

- 40 tests across 9 test classes. All use `AsyncMock` for the LLM provider to avoid real API calls.
- `make_llm(response)` helper creates a mock LLM that returns any desired string, enabling precise control over agent behavior in every test.
- Pipeline smoke tests use a realistic planner response (valid JSON array) to exercise the full `PlannerAgent → OrchestratorAgent` path.
- **Python 3.12 compatibility fix:** `asyncio.get_event_loop()` usage in test helpers replaced with `@pytest.mark.asyncio` test methods, which is the correct pattern under `pytest-asyncio` 0.23+.

---

## Bugs Fixed Summary

| # | Bug | Fix |
|---|---|---|
| 1 | `KeyError: 'name'` in `JSONFormatter` | Renamed span field from `name` to `span_name` |
| 2 | Retry state machine allowed 4 retries | Changed `> MAX_RETRIES` to `>= MAX_RETRIES` |
| 3 | `asyncio.get_event_loop()` removed in Python 3.12 | Migrated tests to `@pytest.mark.asyncio` |
| 4 | `datetime` in task state not JSON-serializable | Added `default=str` to `json.dumps`; used `model_dump(mode="json")` |
| 5 | Pydantic v2 `.dict()` deprecation warnings | Migrated to `.model_dump()` throughout |