"""Tests for the CAG layer (exact + semantic caches)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from app.cag import ExactCache, SemanticCache


# --- ExactCache ---------------------------------------------------------------


@pytest.fixture
def exact() -> ExactCache:
    return ExactCache(ttl=60)


def test_exact_key_is_deterministic():
    args = dict(system="s", user_message="u", model="m")
    assert ExactCache.make_key(**args) == ExactCache.make_key(**args)


def test_exact_key_changes_with_inputs():
    base = dict(system="s", user_message="u", model="m")
    key = ExactCache.make_key(**base)
    assert ExactCache.make_key(**{**base, "system": "s2"}) != key
    assert ExactCache.make_key(**{**base, "user_message": "u2"}) != key
    assert ExactCache.make_key(**{**base, "model": "m2"}) != key


def test_exact_set_get_roundtrip(exact: ExactCache):
    key = ExactCache.make_key(system="s", user_message="u", model="m")
    exact.set(key, "respuesta cacheada")
    assert exact.get(key) == "respuesta cacheada"


def test_exact_miss_returns_none(exact: ExactCache):
    assert exact.get("cliniai:nonexistent") is None


def test_exact_expired_entry_returns_none(exact: ExactCache):
    key = ExactCache.make_key(system="s", user_message="u", model="m")
    exact._items[key] = (time.time() - 1, "vieja")
    assert exact.get(key) is None


def test_exact_ttl_seconds(exact: ExactCache):
    key = ExactCache.make_key(system="s", user_message="u", model="m")
    exact.set(key, "x")
    ttl = exact.ttl_seconds(key)
    assert ttl is not None
    assert 0 < ttl <= 60


# --- SemanticCache --------------------------------------------------------------


def _build_semantic(*, threshold: float = 0.90, log_only: bool = False) -> SemanticCache:
    fake_vectorizer = SimpleNamespace(embed=lambda text: [0.1] * 16)
    return SemanticCache(
        vectorizer=fake_vectorizer, threshold=threshold, ttl=60, log_only=log_only
    )


def test_bucket_partitions_by_doctor():
    assert SemanticCache.bucket_for("doc1", "doctor_info", "gpt-4o-mini") != \
           SemanticCache.bucket_for("doc2", "doctor_info", "gpt-4o-mini")


def test_bucket_partitions_by_model():
    assert SemanticCache.bucket_for("doc1", "doctor_info", "gpt-4o-mini") != \
           SemanticCache.bucket_for("doc1", "doctor_info", "gpt-4o")


def test_semantic_lookup_empty_returns_none():
    cache = _build_semantic()
    assert cache.lookup("¿Qué especialidad tiene?", "doc1:doctor_info:m") is None


def test_semantic_store_then_identical_lookup_hits():
    cache = _build_semantic()
    bucket = "doc1:doctor_info:m"
    cache.store("¿Qué especialidad tiene?", "Es pediatra.", bucket)
    # Identical embedding (fake vectorizer always returns [0.1]*16) → similarity 1.0
    assert cache.lookup("¿Qué especialidad tiene?", bucket) == "Es pediatra."


def test_semantic_lookup_wrong_bucket_misses():
    cache = _build_semantic()
    cache.store("¿Qué especialidad tiene?", "Es pediatra.", "doc1:doctor_info:m")
    assert cache.lookup("¿Qué especialidad tiene?", "doc2:doctor_info:m") is None


def test_semantic_log_only_never_serves():
    cache = _build_semantic(log_only=True)
    bucket = "doc1:doctor_info:m"
    cache.store("pregunta", "respuesta", bucket)
    assert cache.lookup("pregunta", bucket) is None


def test_semantic_expired_entry_misses():
    cache = _build_semantic()
    bucket = "doc1:doctor_info:m"
    cache.store("pregunta", "respuesta", bucket)
    cache._items[0].expires_at = time.time() - 1
    assert cache.lookup("pregunta", bucket) is None


def test_semantic_clear():
    cache = _build_semantic()
    cache.store("q", "r", "b")
    cache.clear()
    assert len(cache._items) == 0
