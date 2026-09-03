"""Exact-match in-memory cache for LLM responses (ported from estimator CAG layer).

The key is a SHA-256 of the system prompt + user message + model. Any change in
the doctor profile (which is embedded in the system prompt) implicitly
invalidates the cache without manual flushing.
"""

from __future__ import annotations

import hashlib
import json
import time

import structlog

from app.services.console import say

log = structlog.get_logger()


class ExactCache:
    """In-memory exact-match cache with TTL timestamps."""

    def __init__(self, ttl: int = 86400):
        self.ttl = ttl
        self._items: dict[str, tuple[float, str]] = {}

    @staticmethod
    def make_key(*, system: str, user_message: str, model: str) -> str:
        payload = json.dumps(
            {"system": system, "user_message": user_message, "model": model},
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"cliniai:{digest}"

    def get(self, key: str) -> str | None:
        entry = self._items.get(key)
        if entry is None:
            log.info("exact_cache_miss", key_prefix=key[:20])
            say(f"❌ Caché exacta: MISS (clave {key[:20]}…) — pregunta nueva")
            return None
        expires_at, reply = entry
        if time.time() > expires_at:
            del self._items[key]
            log.info("exact_cache_miss", key_prefix=key[:20], reason="expired")
            say(f"⏰ Caché exacta: entrada EXPIRADA (clave {key[:20]}…) — se descarta")
            return None
        log.info("exact_cache_hit", key_prefix=key[:20])
        say(f"🎯 LLAMADA IGUAL!! se usó la caché EXACTA (clave {key[:20]}…) — 0 llamadas al LLM")
        return reply

    def set(self, key: str, reply: str) -> None:
        self._items[key] = (time.time() + self.ttl, reply)
        log.info("exact_cache_stored", key_prefix=key[:20], ttl=self.ttl)
        say(f"💾 Caché exacta: respuesta GUARDADA (TTL={self.ttl}s, total={len(self._items)} entradas)")

    def clear(self) -> None:
        self._items.clear()

    def ttl_seconds(self, key: str) -> float | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        remaining = entry[0] - time.time()
        return remaining if remaining > 0 else None
