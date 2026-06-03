---
name: FRIDAY-MAF architecture
description: Package layout, port assignments, workflow names, and critical state-machine rules
---
## Package layout
- config/settings.py — Settings with Ollama defaults, API_PORT=8000
- core/models/ — SubTask.dependencies (fixed from original typo "dependancies")
- core/llm/providers.py — OllamaProvider (default), stub fallback via _stub_response()
- core/tools/tool_router.py — WebSearchTool uses DuckDuckGo (free, no key, ddgs package)
- core/persistence/sqlite_store.py — SQLite task + event persistence (friday_maf.db)
- core/autonomy/execution_loop.py — Background autonomy loop (3s interval)
- core/state/state_machine.py — TaskStateMachine(task_id, initial_state=CREATED)
- core/agents/orchestrator.py — orchestrator
- dashboard/app.py — Streamlit dashboard

## Ports
- Backend FastAPI: port 8000 (workflow: "Backend API", outputType: console, NO --reload flag)
- Streamlit dashboard: port 5000 (workflow: "Start application", outputType: webview)

## Critical state machine rules
**NEVER skip REVIEWING.** Valid path: CREATED→PLANNING→EXECUTING→REVIEWING→COMPLETED.
Orchestrator must call sm.transition(REVIEWING) then sm.transition(COMPLETED).
TaskStateMachine must be initialized with task.status (not always CREATED) when
the orchestrator receives an already-planned task.

**Why:** The SM enforces the full lifecycle. Skipping REVIEWING raises StateMachineError
which gets caught as a task failure.

## LLM stub fix (providers.py)
_stub_response() matches on ALL message content (system+user), not just the last message.
Critic prompts must be matched FIRST (before "search") since subtask descriptions like
"Research the topic" contain "search" as a substring of "research".
