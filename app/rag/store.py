"""In-memory vector store for RAG chunks.

Same design decision as the CAG layer: no external vector DB. Entries are
partitioned by ``doctor_id`` so retrieval never crosses doctors. Documented
limitation: the index dies with the process (single-process deployment).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from app.rag.chunking import Chunk

log = structlog.get_logger()


@dataclass
class _StoredChunk:
    chunk: Chunk
    embedding: np.ndarray  # float32, L2-normalised


class InMemoryVectorStore:
    """Cosine-similarity search over normalised embeddings."""

    def __init__(self) -> None:
        self._items: list[_StoredChunk] = []

    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._items.append(_StoredChunk(chunk=chunk, embedding=vec))

    def add_many(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, emb in zip(chunks, embeddings):
            self.add(chunk, emb)
        log.info("chunks_indexed", count=len(chunks))

    def has_doctor(self, doctor_id: str) -> bool:
        return any(item.chunk.doctor_id == doctor_id for item in self._items)

    def search(
        self, query_embedding: list[float], doctor_id: str, top_k: int = 4
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k most similar chunks for this doctor with scores."""
        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        scored: list[tuple[Chunk, float]] = []
        for item in self._items:
            if item.chunk.doctor_id != doctor_id:
                continue
            similarity = float(np.dot(query, item.embedding))
            scored.append((item.chunk, similarity))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def clear(self, doctor_id: str | None = None) -> None:
        if doctor_id is None:
            self._items.clear()
        else:
            self._items = [i for i in self._items if i.chunk.doctor_id != doctor_id]

    def __len__(self) -> int:
        return len(self._items)
