"""
FRIDAY-MAF Memory Layer
Three-tier memory system:
  A. Short-term  — Redis (active execution state, fast)
  B. Episodic    — Vector DB (conversation history, semantic search)
  C. Semantic    — Knowledge base (long-term, rarely mutated)

All tiers implement the same MemoryAPI so agents don't know which tier
they're talking to.
"""
from __future__ import annotations
import json
import time
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import uuid4

from core.observability.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base Memory API
# ---------------------------------------------------------------------------

class BaseMemory(ABC):

    @abstractmethod
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> str:
        """Persist a value. Returns the key."""

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Fetch by exact key. Returns None if missing."""

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Semantic or prefix search. Returns ranked results."""

    @abstractmethod
    async def update(self, key: str, value: Any) -> bool:
        """Overwrite an existing entry. Returns True on success."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove an entry. Returns True if it existed."""


# ---------------------------------------------------------------------------
# A. Short-Term Memory (Redis-backed)
#    Falls back to in-process dict when Redis is unavailable —
#    useful for local dev without Docker.
# ---------------------------------------------------------------------------

class ShortTermMemory(BaseMemory):
    """
    Active execution state.
    TTL defaults to 3600s so task state doesn't leak between runs.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis_url = redis_url
        self._redis     = None
        self._fallback: Dict[str, tuple] = {}   # key → (value, expire_at or None)
        self._use_redis = False

    async def _init_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis   # type: ignore
                self._redis     = aioredis.from_url(self._redis_url, decode_responses=True)
                await self._redis.ping()
                self._use_redis = True
                logger.info("short_term_memory_redis_connected")
            except Exception as exc:
                logger.warning("short_term_memory_fallback", extra={"reason": str(exc)})
                self._use_redis = False

    async def store(self, key: str, value: Any, ttl: Optional[int] = 3600) -> str:
        await self._init_redis()
        # default=str safely converts datetime/UUID/enum values to strings
        serialised = json.dumps(value, default=str)
        if self._use_redis:
            if ttl:
                await self._redis.setex(key, ttl, serialised)
            else:
                await self._redis.set(key, serialised)
        else:
            expire_at = time.time() + ttl if ttl else None
            self._fallback[key] = (serialised, expire_at)
        return key

    async def retrieve(self, key: str) -> Optional[Any]:
        await self._init_redis()
        if self._use_redis:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None
        entry = self._fallback.get(key)
        if entry is None:
            return None
        raw, expire_at = entry
        if expire_at and time.time() > expire_at:
            del self._fallback[key]
            return None
        return json.loads(raw)

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Prefix search over in-process fallback (Redis version would use SCAN)."""
        results = []
        for key, (raw, expire_at) in list(self._fallback.items()):
            if expire_at and time.time() > expire_at:
                continue
            if query.lower() in key.lower():
                results.append({"key": key, "value": json.loads(raw)})
        return results[:limit]

    async def update(self, key: str, value: Any) -> bool:
        existing = await self.retrieve(key)
        if existing is None:
            return False
        await self.store(key, value)
        return True

    async def delete(self, key: str) -> bool:
        await self._init_redis()
        if self._use_redis:
            return bool(await self._redis.delete(key))
        existed = key in self._fallback
        self._fallback.pop(key, None)
        return existed


# ---------------------------------------------------------------------------
# B. Episodic Memory (Vector DB)
#    Uses an in-process cosine-similarity store.
#    Drop in ChromaDB / Pinecone / Weaviate for production.
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


class EpisodicMemory(BaseMemory):
    """
    Stores conversation turns and past execution summaries.
    Retrieval is semantic — find similar past episodes.
    """

    def __init__(self, embed_fn=None):
        """
        embed_fn: async callable(text) → List[float]
        If None, falls back to a keyword-overlap similarity for dev.
        """
        self._embed_fn = embed_fn
        self._store: Dict[str, Dict[str, Any]] = {}   # key → {text, embedding, meta}

    async def _embed(self, text: str) -> List[float]:
        if self._embed_fn:
            return await self._embed_fn(text)
        # Dev fallback: represent text as a bag-of-char-trigrams in a fixed vector
        vocab = "abcdefghijklmnopqrstuvwxyz0123456789 "
        vec   = [text.lower().count(c) / (len(text) + 1) for c in vocab]
        return vec

    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> str:
        text      = value if isinstance(value, str) else json.dumps(value)
        embedding = await self._embed(text)
        self._store[key] = {"text": text, "embedding": embedding, "value": value, "ts": time.time()}
        logger.debug("episodic_store", extra={"key": key})
        return key

    async def retrieve(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        return entry["value"] if entry else None

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self._store:
            return []
        q_vec   = await self._embed(query)
        scored  = [
            (key, _cosine(q_vec, entry["embedding"]), entry)
            for key, entry in self._store.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"key": k, "score": round(s, 4), "value": e["value"]}
            for k, s, e in scored[:limit]
        ]

    async def update(self, key: str, value: Any) -> bool:
        if key not in self._store:
            return False
        await self.store(key, value)
        return True

    async def delete(self, key: str) -> bool:
        existed = key in self._store
        self._store.pop(key, None)
        return existed


# ---------------------------------------------------------------------------
# C. Semantic Memory (Knowledge Base)
#    Long-term, structured knowledge. Rarely written, often read.
# ---------------------------------------------------------------------------

class SemanticMemory(BaseMemory):
    """
    Static knowledge base — documentation, domain facts, tool manuals.
    In production: backed by a vector DB with nightly refresh jobs.
    """

    def __init__(self):
        self._kb: Dict[str, Dict[str, Any]] = {}

    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> str:
        self._kb[key] = {
            "content": value,
            "tags":    [],
            "ts":      time.time(),
        }
        return key

    async def retrieve(self, key: str) -> Optional[Any]:
        entry = self._kb.get(key)
        return entry["content"] if entry else None

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        q_lower  = query.lower()
        results  = []
        for key, entry in self._kb.items():
            content = json.dumps(entry["content"]).lower()
            if q_lower in content or q_lower in key.lower():
                score = content.count(q_lower) / (len(content) + 1)
                results.append({"key": key, "score": score, "value": entry["content"]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def update(self, key: str, value: Any) -> bool:
        if key not in self._kb:
            return False
        self._kb[key]["content"] = value
        return True

    async def delete(self, key: str) -> bool:
        existed = key in self._kb
        self._kb.pop(key, None)
        return existed


# ---------------------------------------------------------------------------
# Unified Memory Manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Single entry point for all memory operations.
    Agents use this — they don't instantiate tiers directly.
    """

    def __init__(self, embed_fn=None, redis_url: str = "redis://localhost:6379/0"):
        self.short_term = ShortTermMemory(redis_url=redis_url)
        self.episodic   = EpisodicMemory(embed_fn=embed_fn)
        self.semantic   = SemanticMemory()

    async def store_task_state(self, task_id: str, state: Any) -> None:
        await self.short_term.store(f"task:{task_id}:state", state)

    async def retrieve_task_state(self, task_id: str) -> Optional[Any]:
        return await self.short_term.retrieve(f"task:{task_id}:state")

    async def record_episode(self, task_id: str, summary: str) -> None:
        key = f"episode:{task_id}:{uuid4().hex[:8]}"
        await self.episodic.store(key, summary)

    async def recall_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        return await self.episodic.search(query, limit)

    async def learn(self, key: str, knowledge: Any) -> None:
        await self.semantic.store(key, knowledge)

    async def lookup(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        return await self.semantic.search(query, limit)