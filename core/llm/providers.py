"""
FRIDAY-MAF LLM Abstraction Layer
Supports OpenAI, Anthropic, and Gemini through a unified interface.
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

    def __init__(self, model: str, api_key: str, **kwargs):
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
            "messages":    [m.dict() for m in messages],
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
            "messages":    [m.dict() for m in messages],
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
        """Anthropic uses a separate system field."""
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
        # Anthropic does not offer a public embeddings endpoint yet.
        # Fall back to a simple hash-based mock for offline development.
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
        # Gemini streaming via SSE — simplified
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
    "openai":    OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini":    GeminiProvider,
}
 
 
def create_llm_provider(provider: str, model: str, api_key: str, **kwargs) -> BaseLLMProvider:
    """
    Factory — returns a provider instance.
    Example:
        llm = create_llm_provider("anthropic", "claude-3-5-sonnet-20241022", api_key)
    """
    cls = _PROVIDERS.get(provider.lower())
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")
    return cls(model=model, api_key=api_key, **kwargs)
