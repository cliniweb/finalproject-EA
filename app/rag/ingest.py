"""RAG ingestion — chunk the doctor profile (and optional FAQ), embed, index.

Runs once per doctor at first contact (lazy indexing). Embeddings are batched
in one API call per doctor for cost.
"""

from __future__ import annotations

import structlog

from app.rag.chunking import Chunk, chunk_doctor_profile, chunk_faq_text
from app.rag.store import InMemoryVectorStore

log = structlog.get_logger()


class RagIngestService:
    def __init__(self, *, store: InMemoryVectorStore, vectorizer) -> None:
        self.store = store
        self.vectorizer = vectorizer

    def ingest_doctor(
        self, doctor_id: str, doctor_data: dict, faq_text: str | None = None
    ) -> int:
        """Index a doctor's knowledge. Returns the number of chunks indexed.

        Idempotent: if the doctor is already indexed, this is a no-op.
        """
        if self.store.has_doctor(doctor_id):
            log.info("ingest_skipped", doctor_id=doctor_id, reason="already_indexed")
            return 0

        chunks: list[Chunk] = chunk_doctor_profile(doctor_id, doctor_data)
        if faq_text:
            chunks += chunk_faq_text(doctor_id, faq_text)

        if not chunks:
            log.warning("ingest_empty", doctor_id=doctor_id)
            return 0

        embeddings = [self.vectorizer.embed(c.text) for c in chunks]
        self.store.add_many(chunks, embeddings)
        log.info("ingest_done", doctor_id=doctor_id, chunks=len(chunks))
        return len(chunks)
