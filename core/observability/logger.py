"""
FRIDAY-MAF Observability Stack
Structured logging · Metrics counters · Execution tracing · Error tracking
"""
from __future__ import annotations
import time
import logging
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from uuid import uuid4


# ---------------------------------------------------------------------------
# Structured Logger
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emits one JSON object per log line — easy to ingest into any log platform."""

    def format(self, record: logging.LogRecord) -> str:
        log: dict = {
            "timestamp":  datetime.utcnow().isoformat(),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    record.getMessage(),
        }
        # Attach any extra kwargs passed at call time
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            }:
                log[key] = value
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        return json.dumps(log)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class MetricsStore:
    """In-process counters and histograms. Replace with Prometheus in production."""
    _counters:   Dict[str, float]      = field(default_factory=lambda: defaultdict(float))
    _histograms: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def increment(self, key: str, amount: float = 1.0, **labels) -> None:
        label_str = ",".join(f"{k}={v}" for k, v in labels.items())
        full_key  = f"{key}{{{label_str}}}" if label_str else key
        self._counters[full_key] += amount

    def observe(self, key: str, value: float, **labels) -> None:
        label_str = ",".join(f"{k}={v}" for k, v in labels.items())
        full_key  = f"{key}{{{label_str}}}" if label_str else key
        self._histograms[full_key].append(value)

    def snapshot(self) -> Dict[str, Any]:
        histo_summary = {}
        for k, vals in self._histograms.items():
            if vals:
                sorted_v = sorted(vals)
                n        = len(sorted_v)
                histo_summary[k] = {
                    "count": n,
                    "min":   sorted_v[0],
                    "max":   sorted_v[-1],
                    "mean":  sum(sorted_v) / n,
                    "p95":   sorted_v[int(n * 0.95)],
                }
        return {"counters": dict(self._counters), "histograms": histo_summary}


# Singleton
metrics = MetricsStore()


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

@dataclass
class Span:
    span_id:    str = field(default_factory=lambda: str(uuid4())[:8])
    name:       str = ""
    task_id:    str = ""
    agent:      str = ""
    start_time: float = field(default_factory=time.monotonic)
    end_time:   Optional[float] = None
    status:     str = "running"   # running | ok | error
    error:      Optional[str] = None
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str = "ok", error: Optional[str] = None) -> float:
        self.end_time = time.monotonic()
        self.status   = status
        self.error    = error
        return self.duration_ms

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.monotonic()
        return (end - self.start_time) * 1000


class Tracer:
    def __init__(self):
        self._spans: List[Span] = []
        self._logger = get_logger("tracer")

    @asynccontextmanager
    async def trace(self, name: str, task_id: str = "", agent: str = "", **metadata):
        span = Span(name=name, task_id=task_id, agent=agent, metadata=metadata)
        try:
            yield span
            span.finish("ok")
            metrics.observe("span_duration_ms", span.duration_ms, name=name, agent=agent)
            metrics.increment("spans_completed", agent=agent)
        except Exception as exc:
            span.finish("error", error=str(exc))
            metrics.increment("spans_errored", agent=agent)
            raise
        finally:
            self._spans.append(span)
            self._logger.debug(
                "span",
                extra={
                    "span_id":     span.span_id,
                    "span_name":   span.name,    # 'name' is reserved by LogRecord
                    "task_id":     span.task_id,
                    "agent":       span.agent,
                    "duration_ms": round(span.duration_ms, 2),
                    "status":      span.status,
                },
            )

    def recent(self, n: int = 50) -> List[Dict[str, Any]]:
        return [
            {
                "span_id":     s.span_id,
                "name":        s.name,
                "task_id":     s.task_id,
                "agent":       s.agent,
                "duration_ms": round(s.duration_ms, 2),
                "status":      s.status,
                "error":       s.error,
            }
            for s in self._spans[-n:]
        ]


# Singleton
tracer = Tracer()