"""
FRIDAY-MAF Event Bus
Async publish/subscribe. Agents communicate through events, never directly.
This decouples the system — adding a new agent means subscribing, not rewiring.
"""
from __future__ import annotations
import asyncio
from typing import Awaitable, Callable, Dict, List
from core.models import Event, EventType
from core.observability.logger import get_logger, metrics

logger = get_logger(__name__)

Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    In-process async event bus.

    Production upgrade: replace the asyncio.Queue with a Redis Stream
    or Kafka topic — the public API stays identical.
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[Handler]] = {et: [] for et in EventType}
        self._history:     List[Event] = []

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register a coroutine to be called whenever event_type fires."""
        self._subscribers[event_type].append(handler)
        logger.debug("event_bus_subscribe", extra={"event_type": event_type, "handler": handler.__qualname__})

    async def publish(self, event: Event) -> None:
        """
        Fire an event.
        All handlers for that event type run concurrently.
        Failures in one handler do NOT block others.
        """
        self._history.append(event)
        metrics.increment("events_published", event_type=event.event_type)

        logger.debug(
            "event_published",
            extra={
                "event_id":   event.event_id,
                "event_type": event.event_type,
                "task_id":    event.task_id,
            },
        )

        handlers = self._subscribers.get(event.event_type, [])
        if not handlers:
            return

        results = await asyncio.gather(
            *[h(event) for h in handlers],
            return_exceptions=True,
        )
        for h, result in zip(handlers, results):
            if isinstance(result, Exception):
                logger.error(
                    "event_handler_error",
                    extra={
                        "handler":     h.__qualname__,
                        "event_type":  event.event_type,
                        "error":       str(result),
                    },
                )
                metrics.increment("event_handler_errors", event_type=event.event_type)

    def history(self, limit: int = 100) -> List[Event]:
        return self._history[-limit:]

    def clear_history(self) -> None:
        self._history.clear()


# Singleton used by all agents
event_bus = EventBus()