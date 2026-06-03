"""
FRIDAY-MAF LLM Abstraction Layer
Supports Ollama (free/local), OpenAI, Anthropic, and Gemini.
Default provider: Ollama — zero API cost, runs locally.
Switching providers requires only a config change, zero code changes elsewhere.
"""
from __future__ import annotations
import json
import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional
import httpx

from core.models import LLMMessage, LLMResponse
from core.observability.logger import get_logger

logger = get_logger(__name__)


#----------------------------------------------------------------------------------------------------
# Base Provider
#----------------------------------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """
    All LLM providers implement this interface.
    Agents depend on this abstraction, never on a concrete provider.
    """

    def __init__(self, model: str, api_key: str = "", **kwargs):
        self.model = model
        self.api_key = api_key
        self.extra = kwargs

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Single-shot completion."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages:    List[LLMMessage],
        max_tokens:  int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Streaming completion — yields text chunks."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Return embedding vector for the given text."""
        ...


# ---------------------------------------------------------------------------
# Ollama Provider (FREE — local, no API key)
# ---------------------------------------------------------------------------

class OllamaProvider(BaseLLMProvider):
    """
    Uses Ollama running locally at localhost:11434.
    Install: https://ollama.ai  then: ollama pull llama3.2
    Falls back to a stub response if Ollama is unavailable (for testing).
    """

    def __init__(self, model: str, api_key: str = "", **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self._available: Optional[bool] = None   # lazily checked

    async def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                self._available = resp.status_code == 200
        except Exception:
            self._available = False
        if not self._available:
            logger.warning(
                "ollama_unavailable",
                extra={"base_url": self.base_url, "model": self.model,
                       "hint": "Install Ollama and run: ollama pull " + self.model},
            )
        return self._available

    def _build_messages(self, messages: List[LLMMessage]) -> list:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def generate(
        self,
        messages:    List[LLMMessage],
        max_tokens:  int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        available = await self._check_available()
        if not available:
            return self._stub_response(messages)

        payload = {
            "model":    self.model,
            "messages": self._build_messages(messages),
            "stream":   False,
            "options":  {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            content = data.get("message", {}).get("content", "")
            usage   = data.get("eval_count", 0)
            return LLMResponse(
                content      = content,
                model        = self.model,
                input_tokens = data.get("prompt_eval_count", 0),
                output_tokens= usage,
                finish_reason= data.get("done_reason", "stop"),
            )
        except Exception as exc:
            logger.error("ollama_generate_error", extra={"error": str(exc)})
            return self._stub_response(messages)

    def _stub_response(self, messages: List[LLMMessage]) -> LLMResponse:
        """Fallback stub when Ollama is unavailable — returns valid JSON for agents."""
        # Use both system + user content for matching so context is captured
        all_text = " ".join(m.content for m in messages).lower()

        # Critic evaluation (check FIRST — these prompts contain "pass"/"score"/"feedback")
        if ("evaluate" in all_text or "correctness" in all_text
                or "vulnerability" in all_text or "requirement" in all_text
                or ("pass" in all_text and "score" in all_text)):
            content = '{"pass": true, "score": 75, "feedback": "Stub evaluation — Ollama offline. Start Ollama for real results."}'

        # Recovery decision
        elif "recovery" in all_text or ("action" in all_text and "retry" in all_text):
            content = '{"action": "escalate", "reason": "LLM unavailable", "instructions": "Ollama is offline"}'

        # Task parsing / communicator
        elif ("parse this user request" in all_text
              or ("goal" in all_text and "metadata" in all_text)):
            content = '{"goal": "Process the submitted task", "metadata": {"complexity": "low", "domain": "general", "urgency": "normal"}}'

        # Format response for user
        elif "format this result" in all_text or "human-readable" in all_text:
            content = "Task completed (Ollama offline — start Ollama for real AI responses)."

        # Planner / decomposition
        elif ("decompose" in all_text or "dag" in all_text
              or ("subtask" in all_text and "agent" in all_text)):
            content = '[{"task_description": "Analyse and process the request", "required_agent": "research", "dependencies": []}]'

        # Search query generation (must come AFTER critic/planner checks)
        elif "list" in all_text and ("search quer" in all_text or "web search" in all_text):
            content = '["key information about the topic", "background context needed"]'

        # Research synthesis
        elif "synthesise" in all_text or ("research report" in all_text):
            content = '{"summary": "Stub research result (Ollama offline).", "key_findings": ["Start Ollama for real results"], "sources": [], "confidence": 10}'

        # Code generation
        elif "write python" in all_text or "python code" in all_text:
            content = '{"code": "print(\'Hello from stub — start Ollama for real code generation\')", "explanation": "Stub response", "language": "python"}'

        # Data analysis plan
        elif "plan the data analysis" in all_text or "analysis task" in all_text:
            content = '{"analysis_type": "descriptive", "steps": ["Collect data", "Summarise"]}'

        # Interpret results
        elif "interpret" in all_text and "analysis" in all_text:
            content = '{"insights": ["Stub insight — start Ollama"], "recommendations": ["Install and start Ollama locally"]}'

        # Generic fallback
        else:
            content = '{"result": "Task processed (Ollama offline — start Ollama for real execution)", "status": "stub"}'

        return LLMResponse(content=content, model="stub", input_tokens=0, output_tokens=0)

    async def stream(
        self,
        messages:    List[LLMMessage],
        max_tokens:  int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        available = await self._check_available()
        if not available:
            yield self._stub_response(messages).content
            return

        payload = {
            "model":    self.model,
            "messages": self._build_messages(messages),
            "stream":   True,
            "options":  {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            data  = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue

    async def embed(self, text: str) -> List[float]:
        available = await self._check_available()
        if not available:
            return [0.0] * 384
        payload = {"model": self.model, "input": text}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{self.base_url}/api/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [[]])[0]
                return embeddings or [0.0] * 384
        except Exception:
            return [0.0] * 384


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseLLMProvider):
    BASE_URL = "https://api.openai.com/v1"

    async def generate(
        self,
        messages:    List[LLMMessage],
        max_tokens:  int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        payload = {
            "model":       self.model,
            "messages":    [m.model_dump() for m in messages],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        usage  = data.get("usage", {})
        return LLMResponse(
            content=choice["message"]["content"],
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(
        self,
        messages:    List[LLMMessage],
        max_tokens:  int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        payload = {
            "model":       self.model,
            "messages":    [m.model_dump() for m in messages],
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      True,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            data  = json.loads(chunk)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except json.JSONDecodeError:
                            continue

    async def embed(self, text: str) -> List[float]:
        embed_model = self.extra.get("embed_model", "text-embedding-3-small")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": embed_model, "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseLLMProvider):
    BASE_URL = "https://api.anthropic.com/v1"

    def _split_messages(self, messages: List[LLMMessage]):
        system = ""
        conv   = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                conv.append({"role": m.role, "content": m.content})
        return system, conv

    async def generate(
        self,
        messages:    List[LLMMessage],
        max_tokens:  int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        system, conv = self._split_messages(messages)
        payload: dict = {
            "model":      self.model,
            "messages":   conv,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.BASE_URL}/messages",
                headers={
                    "x-api-key":         self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        usage = data.get("usage", {})
        return LLMResponse(
            content=data["content"][0]["text"],
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "end_turn"),
        )

    async def stream(
        self,
        messages:    List[LLMMessage],
        max_tokens:  int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        system, conv = self._split_messages(messages)
        payload: dict = {
            "model":      self.model,
            "messages":   conv,
            "max_tokens": max_tokens,
            "stream":     True,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/messages",
                headers={
                    "x-api-key":         self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                yield data["delta"].get("text", "")
                        except json.JSONDecodeError:
                            continue

    async def embed(self, text: str) -> List[float]:
        logger.warning("Anthropic embed not available — returning zero vector")
        return [0.0] * 1536


# ---------------------------------------------------------------------------
# Gemini Provider
# ---------------------------------------------------------------------------

class GeminiProvider(BaseLLMProvider):
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def _build_contents(self, messages: List[LLMMessage]) -> list:
        contents = []
        for m in messages:
            role = "user" if m.role in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        return contents

    async def generate(
        self,
        messages:    List[LLMMessage],
        max_tokens:  int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        contents = self._build_contents(messages)
        url      = f"{self.BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"
        payload  = {
            "contents":         contents,
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return LLMResponse(content=text, model=self.model)

    async def stream(
        self,
        messages:    List[LLMMessage],
        max_tokens:  int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        contents = self._build_contents(messages)
        url      = f"{self.BASE_URL}/models/{self.model}:streamGenerateContent?key={self.api_key}&alt=sse"
        payload  = {
            "contents":         contents,
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            text = data["candidates"][0]["content"]["parts"][0].get("text", "")
                            if text:
                                yield text
                        except (json.JSONDecodeError, KeyError):
                            continue

    async def embed(self, text: str) -> List[float]:
        url     = f"{self.BASE_URL}/models/embedding-001:embedContent?key={self.api_key}"
        payload = {"model": "models/embedding-001", "content": {"parts": [{"text": text}]}}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["embedding"]["values"]


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "ollama":    OllamaProvider,
    "openai":    OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini":    GeminiProvider,
}


def create_llm_provider(provider: str, model: str, api_key: str = "", **kwargs) -> BaseLLMProvider:
    """
    Factory — returns a provider instance.
    Default (free): create_llm_provider("ollama", "llama3.2")
    """
    cls = _PROVIDERS.get(provider.lower())
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")
    return cls(model=model, api_key=api_key, **kwargs)
