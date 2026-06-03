"""
FRIDAY-MAF Base Agent
All agents inherit from BaseAgent.
Provides: LLM access, memory access, event publishing, structured logging.
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.models import AgentType, Event, EventType, LLMMessage
from core.llm.providers import BaseLLMProvider
from core.memory.memory_manager import MemoryManager
from core.events.event_bus import EventBus
from core.observability.logger import get_logger, tracer, metrics


class BaseAgent(ABC):
    agent_type:   AgentType
    system_prompt: str = "You are a helpful AI agent."

    def __init__(
        self,
        llm:        BaseLLMProvider,
        memory:     MemoryManager,
        event_bus:  EventBus,
    ):
        self.llm       = llm
        self.memory    = memory
        self.event_bus = event_bus
        self.logger    = get_logger(f"agent.{self.agent_type}")

    async def think(
        self,
        user_message: str,
        context:      Optional[str] = None,
        temperature:  float = 0.7,
        max_tokens:   int   = 2048,
        json_mode:    bool  = False,
    ) -> str:
        """
        Send a prompt to the LLM and return the text response.
        Wraps the call in a trace span for observability.
        """
        system = self.system_prompt
        if json_mode:
            system += "\n\nRespond ONLY with valid JSON. No markdown, no preamble."
        if context:
            system += f"\n\n--- CONTEXT ---\n{context}"

        messages: List[LLMMessage] = [
            LLMMessage(role="system",    content=system),
            LLMMessage(role="user",      content=user_message),
        ]

        async with tracer.trace("llm_call", agent=str(self.agent_type)) as span:
            response = await self.llm.generate(
                messages    = messages,
                temperature = temperature,
                max_tokens  = max_tokens,
            )
            span.metadata.update({
                "input_tokens":  response.input_tokens,
                "output_tokens": response.output_tokens,
            })
            metrics.observe(
                "llm_tokens_used",
                response.input_tokens + response.output_tokens,
                agent=str(self.agent_type),
            )

        return response.content

    async def think_json(self, user_message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """think() but parses and returns the JSON response."""
        raw = await self.think(user_message, context=context, json_mode=True)
        # Strip markdown fences if model adds them
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(cleaned)

    async def emit(self, event_type: EventType, task_id: str, **payload) -> None:
        event = Event(event_type=event_type, task_id=task_id, payload=payload)
        await self.event_bus.publish(event)

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Main entry point — each agent implements its own logic here."""
        ...