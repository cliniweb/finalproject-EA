"""Retriever — embeds the query, searches the store, applies a similarity floor.

The floor (``min_score``) is the red-flag threshold: chunks below it are noise,
and returning zero chunks is a *signal* (the answer is not in the knowledge
base) that the generation node uses to refuse instead of hallucinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from app.rag.chunking import Chunk
from app.rag.store import InMemoryVectorStore
from app.services.console import say

log = structlog.get_logger()


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(
        self,
        *,
        store: InMemoryVectorStore,
        vectorizer: Any,
        top_k: int = 4,
        min_score: float = 0.25,
    ) -> None:
        self.store = store
        self.vectorizer = vectorizer
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str, doctor_id: str) -> list[RetrievedChunk]:
        embedding = self.vectorizer.embed(query)
        results = self.store.search(embedding, doctor_id=doctor_id, top_k=self.top_k)
        kept = [
            RetrievedChunk(chunk=c, score=s) for c, s in results if s >= self.min_score
        ]
        log.info(
            "retrieval_done",
            doctor_id=doctor_id,
            candidates=len(results),
            kept=len(kept),
            top_score=round(results[0][1], 4) if results else None,
        )
        if kept:
            say(
                f"📚 RAG: recuperados {len(kept)}/{len(results)} fragmentos del perfil "
                f"de '{doctor_id}' (mejor score={results[0][1]:.4f}, piso={self.min_score})"
            )
        else:
            say(
                f"🚫 RAG: NINGÚN fragmento supera el piso ({self.min_score}) para "
                f"'{doctor_id}' — la respuesta NO está en la base de conocimiento"
            )
        return kept
