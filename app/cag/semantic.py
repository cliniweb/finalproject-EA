"""Semantic in-memory cache (ported from estimator CAG layer).

Two questions are considered the same when:

1. Their **bucket** matches exactly — ``doctor_id:node:model``. Questions to
   different doctors NEVER share a cache entry.
2. The cosine similarity of their embeddings is at least ``threshold``.

When ``log_only=True`` the cache logs potential hits but never serves them —
useful for calibrating the threshold before flipping it on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from app.services.console import say

log = structlog.get_logger()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


@dataclass
class _SemanticEntry:
    bucket: str
    embedding: list[float]
    reply: str
    expires_at: float


class SemanticCache:
    """In-memory vector-similarity cache using cosine similarity."""

    def __init__(
        self,
        *,
        vectorizer: Any,
        threshold: float = 0.90,
        ttl: int = 86400,
        log_only: bool = False,
    ) -> None:
        self.vectorizer = vectorizer
        self.threshold = threshold
        self.ttl = ttl
        self.log_only = log_only
        self._items: list[_SemanticEntry] = []

    @staticmethod
    def bucket_for(doctor_id: str, node: str, model: str) -> str:
        # doctor_id partitions the cache: an answer about one doctor must
        # never be served for another. The node and model partition further.
        return f"{doctor_id}:{node}:{model}"

    def lookup(self, question: str, bucket: str) -> str | None:
        embedding = self.vectorizer.embed(question)
        now = time.time()

        best_similarity = 0.0
        best_reply: str | None = None

        for entry in self._items:
            if entry.bucket != bucket:
                continue
            if now > entry.expires_at:
                continue
            similarity = _cosine_similarity(embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_reply = entry.reply

        log.info(
            "semantic_cache_lookup",
            bucket=bucket,
            similarity=round(best_similarity, 4),
            threshold=self.threshold,
        )
        say(
            f"🔍 Caché semántica: buscando en bucket '{bucket}' — "
            f"mejor similitud={best_similarity:.4f} (umbral={self.threshold})"
        )

        if best_similarity < self.threshold or best_reply is None:
            log.info("semantic_cache_miss", bucket=bucket, reason="below_threshold")
            say("❌ Caché semántica: MISS — ninguna pregunta parecida supera el umbral")
            return None

        if self.log_only:
            log.info(
                "semantic_cache_hit_log_only",
                bucket=bucket,
                similarity=round(best_similarity, 4),
            )
            say(
                f"👀 Caché semántica: HIT POTENCIAL (similitud={best_similarity:.4f}) "
                "pero modo LOG_ONLY activo — NO se sirve"
            )
            return None

        log.info("semantic_cache_hit", bucket=bucket, similarity=round(best_similarity, 4))
        say(f"🎯 LLAMADA IGUAL!! se usó la caché SEMÁNTICA (similitud={best_similarity:.4f})")
        return best_reply

    def store(self, question: str, reply: str, bucket: str) -> None:
        embedding = self.vectorizer.embed(question)
        self._items.append(
            _SemanticEntry(
                bucket=bucket,
                embedding=embedding,
                reply=reply,
                expires_at=time.time() + self.ttl,
            )
        )
        log.info("semantic_cache_stored", bucket=bucket, ttl=self.ttl)
        say(f"💾 Caché semántica: pregunta+respuesta GUARDADAS en bucket '{bucket}' (total={len(self._items)})")

    def clear(self) -> None:
        self._items.clear()
