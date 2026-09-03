"""Tests for the RAG layer (chunking, store, retriever, ingest)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rag.chunking import Chunk, chunk_doctor_profile, chunk_faq_text
from app.rag.ingest import RagIngestService
from app.rag.retriever import Retriever
from app.rag.store import InMemoryVectorStore


_DOCTOR = {
    "nombrePersona": "Ana García",
    "especialidades": ["Pediatría", "Neonatología"],
    "localidades": [{"localidad": {"nombre": "Centro", "direccion": "Calle 1"}}],
    "seguros": ["MAPFRE", "ASSA"],
    "empty": None,
}


# --- Chunking ------------------------------------------------------------------


def test_profile_chunking_one_chunk_per_section():
    chunks = chunk_doctor_profile("doc1", _DOCTOR)
    sections = {c.metadata.get("section") for c in chunks}
    assert "especialidades" in sections
    assert "seguros" in sections
    assert "summary" in sections
    # Empty fields are skipped
    assert "empty" not in sections


def test_profile_chunks_carry_doctor_id():
    chunks = chunk_doctor_profile("doc1", _DOCTOR)
    assert all(c.doctor_id == "doc1" for c in chunks)


def test_profile_chunk_text_is_human_readable():
    chunks = chunk_doctor_profile("doc1", _DOCTOR)
    espec = next(c for c in chunks if c.metadata.get("section") == "especialidades")
    assert "Pediatría" in espec.text
    assert "Especialidades" in espec.text


def test_faq_chunking_splits_paragraphs():
    text = "Primer párrafo sobre horarios.\n\nSegundo párrafo sobre seguros."
    chunks = chunk_faq_text("doc1", text)
    assert len(chunks) >= 1
    assert all(c.doctor_id == "doc1" for c in chunks)


def test_faq_chunking_caps_size():
    text = "x" * 5000
    chunks = chunk_faq_text("doc1", text)
    assert all(len(c.text) <= 1500 for c in chunks)


# --- Vector store -----------------------------------------------------------------


def _emb(vec: list[float]) -> list[float]:
    return vec


def test_store_search_returns_most_similar():
    store = InMemoryVectorStore()
    c1 = Chunk(doctor_id="doc1", source="s1", text="pediatría")
    c2 = Chunk(doctor_id="doc1", source="s2", text="seguros")
    store.add(c1, [1.0, 0.0, 0.0])
    store.add(c2, [0.0, 1.0, 0.0])

    results = store.search([1.0, 0.1, 0.0], doctor_id="doc1", top_k=2)
    assert results[0][0].source == "s1"  # most similar first
    assert results[0][1] > results[1][1]


def test_store_partitions_by_doctor():
    store = InMemoryVectorStore()
    store.add(Chunk(doctor_id="doc1", source="s1", text="a"), [1.0, 0.0])
    store.add(Chunk(doctor_id="doc2", source="s2", text="b"), [1.0, 0.0])

    results = store.search([1.0, 0.0], doctor_id="doc1", top_k=10)
    assert len(results) == 1
    assert results[0][0].doctor_id == "doc1"


def test_store_has_doctor():
    store = InMemoryVectorStore()
    assert store.has_doctor("doc1") is False
    store.add(Chunk(doctor_id="doc1", source="s", text="t"), [1.0])
    assert store.has_doctor("doc1") is True


def test_store_clear_by_doctor():
    store = InMemoryVectorStore()
    store.add(Chunk(doctor_id="doc1", source="s", text="t"), [1.0])
    store.add(Chunk(doctor_id="doc2", source="s", text="t"), [1.0])
    store.clear("doc1")
    assert store.has_doctor("doc1") is False
    assert store.has_doctor("doc2") is True


# --- Retriever ------------------------------------------------------------------


def test_retriever_applies_min_score_floor():
    store = InMemoryVectorStore()
    store.add(Chunk(doctor_id="doc1", source="s1", text="a"), [1.0, 0.0])
    store.add(Chunk(doctor_id="doc1", source="s2", text="b"), [-1.0, 0.0])  # opposite

    vectorizer = SimpleNamespace(embed=lambda text: [1.0, 0.0])
    retriever = Retriever(store=store, vectorizer=vectorizer, top_k=5, min_score=0.5)

    results = retriever.retrieve("query", doctor_id="doc1")
    assert len(results) == 1  # opposite vector is below floor
    assert results[0].chunk.source == "s1"


def test_retriever_returns_empty_when_nothing_relevant():
    store = InMemoryVectorStore()
    store.add(Chunk(doctor_id="doc1", source="s1", text="a"), [0.0, 1.0])
    vectorizer = SimpleNamespace(embed=lambda text: [1.0, 0.0])
    retriever = Retriever(store=store, vectorizer=vectorizer, top_k=5, min_score=0.5)
    assert retriever.retrieve("query", doctor_id="doc1") == []


# --- Ingest -----------------------------------------------------------------------


def test_ingest_indexes_doctor_once():
    store = InMemoryVectorStore()
    vectorizer = SimpleNamespace(embed=lambda text: [0.5, 0.5])
    ingest = RagIngestService(store=store, vectorizer=vectorizer)

    count1 = ingest.ingest_doctor("doc1", _DOCTOR)
    assert count1 > 0
    # Idempotent: second call is a no-op
    count2 = ingest.ingest_doctor("doc1", _DOCTOR)
    assert count2 == 0
    assert len(store) == count1


def test_ingest_with_faq_text():
    store = InMemoryVectorStore()
    vectorizer = SimpleNamespace(embed=lambda text: [0.5, 0.5])
    ingest = RagIngestService(store=store, vectorizer=vectorizer)

    count = ingest.ingest_doctor("doc1", _DOCTOR, faq_text="Horario: lunes a viernes.")
    profile_only = len(chunk_doctor_profile("doc1", _DOCTOR))
    assert count > profile_only
