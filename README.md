# FRIDAY-MAF — Multi-Agent Framework

A production-grade, fully async multi-agent AI framework built in Python.
Supports OpenAI, Anthropic, and Gemini through a single unified interface.

---

## Architecture at a Glance

```
User Request
     │
     ▼
 API Layer (FastAPI)
     │
     ▼
 Communicator Agent   ← NL → structured Task
     │
     ▼
 Event Bus            ← all inter-agent messages flow through here
     │
     ▼
 Planner Agent        ← Task → DAG of SubTasks
     │
     ▼
 Orchestrator Agent   ← executes DAG, manages state machine
     │
   ┌─┴──────────────────────────┐
   ▼                            ▼
Worker Agents              Critic System
 - ResearchAgent            - LogicCritic
 - CodingAgent              - SecurityCritic
 - DataAnalystAgent         - RequirementCritic
   │                            │
   └──────── fail ──────────────┘
                │
                ▼
         Recovery Agent   ← retry / fallback / escalate
```

---

## Folder Structure

```
friday_maf/
├── core/
│   ├── models.py              # All Pydantic models and enums
│   ├── agents/
│   │   ├── base_agent.py      # BaseAgent: LLM + memory + events
│   │   ├── communicator.py    # NL parsing + response formatting
│   │   ├── planner.py         # Goal → DAG
│   │   ├── orchestrator.py    # DAG execution engine
│   │   └── recovery.py        # Failure handler
│   ├── workers/
│   │   └── workers.py         # BaseWorker + Research/Coding/DataAnalyst
│   ├── critics/
│   │   └── critics.py         # Logic/Security/Requirement critics
│   ├── memory/
│   │   └── memory_manager.py  # Short-term/Episodic/Semantic memory
│   ├── tools/
│   │   └── tool_router.py     # Tools + permission-enforcing router
│   ├── llm/
│   │   └── providers.py       # OpenAI / Anthropic / Gemini abstraction
│   ├── state/
│   │   └── state_machine.py   # Task lifecycle state machine
│   ├── events/
│   │   └── event_bus.py       # Async pub/sub event bus
│   └── observability/
│       └── logger.py          # Structured logging + metrics + tracing
├── api/
│   └── main.py                # FastAPI app (POST /task, GET /task/{id}, GET /health)
├── config/
│   └── settings.py            # Env-var config via pydantic-settings
├── tests/
│   └── test_all.py            # 40 unit tests (100% pass)
└── docs/
    ├── README.md              # This file
    ├── api_reference.md       # Endpoint documentation
    └── architecture.md        # Deep-dive architecture notes
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set at minimum:
# LLM_API_KEY=your_key_here
# LLM_PROVIDER=anthropic   # or openai, gemini
# LLM_MODEL=claude-3-5-haiku-20241022
```

### 3. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Submit a task

```bash
# Submit
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"message": "Research the latest developments in quantum computing and summarise them"}'

# Response:
# {"task_id": "abc-123", "status": "CREATED", "result": null}

# Poll for result
curl http://localhost:8000/task/abc-123
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Switching LLM Providers

Change two lines in `.env` — zero code changes needed:

```bash
# Use OpenAI GPT-4o
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# Use Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-20241022

# Use Google Gemini
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
```

---

## Core Concepts

### Task Lifecycle (State Machine)

```
CREATED → PLANNING → EXECUTING → REVIEWING → COMPLETED
                              ↘ RETRYING  → EXECUTING
                           anywhere → FAILED
```

- `RETRYING` is capped at 3 attempts per subtask. On the 3rd retry, the state machine
  forces `FAILED` to prevent infinite loops — this is by design.

### DAG Execution

The Planner produces a list of `SubTask` objects with `dependencies` (task IDs).
The Orchestrator performs a topological sort and runs independent subtasks in parallel
using `asyncio.gather`. Dependent subtasks wait for their prerequisites.

### Critic System

Every worker output passes through three independent critics in parallel:
- **LogicCritic** — checks factual correctness and coherence
- **SecurityCritic** — checks for vulnerabilities, hardcoded secrets, unsafe code
- **RequirementCritic** — checks spec compliance

If any critic scores below its threshold, the output is sent back to the worker
with specific feedback. This loop runs at most 3 times before escalating to FAILED.

### Memory Tiers

| Tier | Backend | Use |
|------|---------|-----|
| Short-term | Redis (falls back to in-process dict) | Active task state, fast I/O |
| Episodic | In-process vector store (drop-in for ChromaDB) | Past conversations, semantic recall |
| Semantic | In-process key-value | Knowledge base, documentation |

### Tool Permissions

Tools are routed through `ToolRouter`, which enforces a permission matrix:

| Agent | Tools Allowed |
|-------|--------------|
| ResearchAgent | web_search, file |
| CodingAgent | python_exec, file, github |
| DataAnalystAgent | python_exec, file, web_search |
| Communicator / Planner / Orchestrator | none |

Calling a tool without permission returns a `ToolResult.error` — never raises.

---

## Adding a New Worker

1. Subclass `BaseWorker` in `core/workers/`
2. Define `agent_type` and `system_prompt`
3. Implement `execute(subtask, context) → Any`
4. Register it in `OrchestratorAgent._workers`
5. Add tool permissions in `AGENT_PERMISSIONS`

```python
class SummaryAgent(BaseWorker):
    agent_type    = AgentType.RESEARCH   # reuse or extend AgentType enum
    system_prompt = "You summarise documents..."

    async def execute(self, subtask, context):
        content = await self._call_tool("file", operation="read", path=subtask.task_description)
        return await self.think_json(f"Summarise:\n{content}")
```

---

## Production Checklist

- [ ] Set `LLM_API_KEY` via secrets manager (not `.env` in production)
- [ ] Point `REDIS_URL` to a real Redis instance for shared short-term memory
- [ ] Replace `EpisodicMemory` with ChromaDB / Pinecone for persistent vector search
- [ ] Add a real web search API key (`SEARCH_API_KEY`) — Tavily / SerpAPI / Brave
- [ ] Deploy behind a reverse proxy (nginx/Caddy) with TLS
- [ ] Set `LOG_LEVEL=INFO` in production (DEBUG is very verbose)
- [ ] Add rate limiting to the `/task` endpoint
- [ ] Export metrics to Prometheus via the `/health` snapshot endpoint