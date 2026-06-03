# FRIDAY-MAF API Reference

Base URL: `http://localhost:8000`

All request/response bodies are JSON. All timestamps are ISO 8601 UTC.

---

## POST /task

Submit a natural language task for multi-agent processing.

**Returns immediately** with a `task_id`. The pipeline runs in the background.
Poll `GET /task/{id}` to check for completion.

### Request

```json
{
  "message": "string — natural language instruction for the agents"
}
```

### Response `202 Accepted`

```json
{
  "task_id": "3f2a1b4c-...",
  "status":  "CREATED",
  "result":  null,
  "error":   null
}
```

### Example

```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a Python function that calculates Fibonacci numbers and test it"}'
```

---

## GET /task/{task_id}

Poll the status and result of a previously submitted task.

### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| task_id | string (UUID) | The ID returned by POST /task |

### Response `200 OK`

```json
{
  "task_id": "3f2a1b4c-...",
  "status":  "COMPLETED",
  "result":  "Here is the Fibonacci implementation...",
  "error":   null
}
```

### Status values

| Status | Meaning |
|--------|---------|
| `CREATED` | Task received, not yet started |
| `PLANNING` | Planner is decomposing the goal |
| `EXECUTING` | Workers are running subtasks |
| `REVIEWING` | Critics are evaluating outputs |
| `RETRYING` | A critic failed; worker is retrying |
| `COMPLETED` | All subtasks done, result available |
| `FAILED` | Unrecoverable error; see `error` field |

### Response `404 Not Found`

```json
{
  "detail": "Task '3f2a1b4c-...' not found"
}
```

### Polling pattern

```python
import time, httpx

task_id = httpx.post("http://localhost:8000/task", json={"message": "..."}).json()["task_id"]

while True:
    resp   = httpx.get(f"http://localhost:8000/task/{task_id}").json()
    status = resp["status"]
    if status in ("COMPLETED", "FAILED"):
        print(resp["result"] or resp["error"])
        break
    time.sleep(2)
```

---

## GET /health

Liveness check, runtime metrics, and recent execution traces.

### Response `200 OK`

```json
{
  "status":  "ok",
  "version": "1.0.0",
  "metrics": {
    "counters": {
      "tasks_completed":     12,
      "tasks_failed":         1,
      "tool_calls{tool=web_search,agent=research}": 45,
      "critic_evaluations{critic=logic,passed=True}": 34
    },
    "histograms": {
      "span_duration_ms{name=llm_call,agent=research}": {
        "count": 45,
        "min":   210.3,
        "max":  4801.2,
        "mean":  890.1,
        "p95":  2340.0
      }
    }
  },
  "recent_spans": [
    {
      "span_id":     "a1b2c3d4",
      "name":        "llm_call",
      "task_id":     "3f2a1b4c-...",
      "agent":       "research",
      "duration_ms": 832.4,
      "status":      "ok",
      "error":       null
    }
  ]
}
```

---

## Error Handling

The API never exposes internal agent names or stack traces to clients.
All errors are surfaced through the `error` field on the task object.

| HTTP Code | Condition |
|-----------|-----------|
| 202 | Task accepted |
| 200 | Task polled successfully |
| 404 | Task ID not found |
| 422 | Invalid request body (Pydantic validation) |
| 500 | Unexpected server error (check logs) |