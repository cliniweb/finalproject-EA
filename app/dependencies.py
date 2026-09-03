"""Dependency factories for shared singletons (mirrors estimator's dependencies.py)."""

from __future__ import annotations

from functools import lru_cache

import structlog
from openai import OpenAI

from app.cag import ExactCache, SemanticCache
from app.config import get_settings

log = structlog.get_logger()


class OpenAITextVectorizer:
    """Thin embedding client for the semantic cache."""

    def __init__(self, model: str, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding


@lru_cache
def get_exact_cache() -> ExactCache:
    settings = get_settings()
    return ExactCache(ttl=settings.CACHE_TTL)


@lru_cache
def get_semantic_cache() -> SemanticCache | None:
    """Semantic cache singleton. ``None`` when no OpenAI key is configured."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        log.warning("semantic_cache_disabled", reason="no_openai_key")
        return None
    vectorizer = OpenAITextVectorizer(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    return SemanticCache(
        vectorizer=vectorizer,
        threshold=settings.SEMANTIC_CACHE_THRESHOLD,
        ttl=settings.SEMANTIC_CACHE_TTL,
        log_only=settings.SEMANTIC_CACHE_LOG_ONLY,
    )


# --- RAG layer ---------------------------------------------------------------


@lru_cache
def get_vectorizer() -> OpenAITextVectorizer | None:
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        log.warning("vectorizer_disabled", reason="no_openai_key")
        return None
    return OpenAITextVectorizer(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )


@lru_cache
def get_vector_store():
    from app.rag.store import InMemoryVectorStore

    return InMemoryVectorStore()


@lru_cache
def get_rag_ingest():
    """Ingestion service. ``None`` without an OpenAI key."""
    from app.rag.ingest import RagIngestService

    vectorizer = get_vectorizer()
    if vectorizer is None:
        return None
    return RagIngestService(store=get_vector_store(), vectorizer=vectorizer)


@lru_cache
def get_retriever():
    """Retriever singleton. ``None`` without an OpenAI key."""
    from app.rag.retriever import Retriever

    settings = get_settings()
    vectorizer = get_vectorizer()
    if vectorizer is None:
        return None
    return Retriever(
        store=get_vector_store(),
        vectorizer=vectorizer,
        top_k=settings.RAG_TOP_K,
        min_score=settings.RAG_MIN_SCORE,
    )


# --- Agents layer --------------------------------------------------------------


@lru_cache
def get_tool_registry():
    """Least-privilege tool registry. Booking-URL generation is only callable
    from the confirm node; slot fetching only from fetch_slots."""
    from app.agents.privileges import ToolRegistry

    settings = get_settings()
    registry = ToolRegistry(strict=settings.PRIVILEGE_STRICT)
    registry.register("build_booking_url", {"confirm"})
    registry.register("fetch_available_slots", {"fetch_slots"})
    registry.register("fetch_doctor", {"api_entry", "suggest_doctor"})
    registry.register("search_doctors_by_text", {"suggest_doctor"})
    return registry
