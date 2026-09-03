"""Session store — Redis-backed when available, in-memory fallback otherwise.

The public API (get_or_create / save / delete) is unchanged. On the first
session operation the store probes Redis (if REDIS_ENABLED and the ``redis``
package is installed); when the probe fails it degrades to the original
in-memory dict and logs the reason once.
"""

from __future__ import annotations

import structlog

from app.config import get_settings
from app.domain.schemas import ChatState
from app.services.console import say

log = structlog.get_logger()

_sessions: dict[str, ChatState] = {}

# None until probed; stays None when Redis is unavailable (memory fallback).
_redis_client = None
_redis_probed = False


def _get_redis():
    """Probe Redis once; return a live client or None."""
    global _redis_client, _redis_probed
    if _redis_probed:
        return _redis_client
    _redis_probed = True

    settings = get_settings()
    if not settings.REDIS_ENABLED:
        log.info("session_store_backend", backend="memory", reason="redis_disabled")
        say("🗄️ Sesiones: Redis deshabilitado — usando memoria")
        return None
    try:
        import redis  # optional dependency
    except ImportError:
        log.warning("session_store_backend", backend="memory", reason="redis_package_missing")
        say("🗄️ Sesiones: paquete redis no instalado — usando memoria")
        return None
    try:
        say(f"🗄️ Sesiones: >>> probando conexión Redis en {settings.REDIS_URL} (timeout=2s)…")
        client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        _redis_client = client
        log.info("session_store_backend", backend="redis", url=settings.REDIS_URL)
        say("🗄️ Sesiones: <<< Redis conectado OK")
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "session_store_backend",
            backend="memory",
            reason="redis_unreachable",
            error=str(exc)[:200],
        )
        say(f"🗄️ Sesiones: <<< Redis inaccesible ({str(exc)[:150]}) — usando memoria")
    return _redis_client


def _key(session_id: str) -> str:
    return f"cliniai:session:{session_id}"


def get_or_create(session_id: str, doctor_id: str) -> ChatState:
    say(f"🗄️ Sesiones: get_or_create('{session_id}')…")
    r = _get_redis()
    if r is not None:
        try:
            say(f"🗄️ Sesiones: >>> Redis GET {_key(session_id)}…")
            raw = r.get(_key(session_id))
            say(f"🗄️ Sesiones: <<< Redis GET {'HIT (' + str(len(raw)) + ' bytes)' if raw else 'MISS'}")
            if raw:
                return ChatState.model_validate_json(raw)
            state = ChatState(session_id=session_id, doctor_id=doctor_id)
            save(state)
            return state
        except Exception as exc:  # noqa: BLE001
            log.warning("redis_get_failed", error=str(exc)[:200])
            say(f"⚠️ Sesiones: Redis GET falló ({str(exc)[:150]}) — usando memoria")

    if session_id not in _sessions:
        say(f"🗄️ Sesiones: creando NUEVA sesión en memoria '{session_id}'")
        _sessions[session_id] = ChatState(session_id=session_id, doctor_id=doctor_id)
    else:
        say(f"🗄️ Sesiones: sesión existente en memoria '{session_id}'")
    return _sessions[session_id]


def save(state: ChatState) -> None:
    r = _get_redis()
    if r is not None:
        try:
            settings = get_settings()
            say(f"🗄️ Sesiones: >>> Redis SET {_key(state.session_id)}…")
            r.set(_key(state.session_id), state.model_dump_json(), ex=settings.REDIS_SESSION_TTL)
            say("🗄️ Sesiones: <<< Redis SET OK")
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("redis_save_failed", error=str(exc)[:200])
            say(f"⚠️ Sesiones: Redis SET falló ({str(exc)[:150]}) — guardando en memoria")
    _sessions[state.session_id] = state
    say(f"🗄️ Sesiones: guardada en memoria '{state.session_id}'")


def delete(session_id: str) -> None:
    r = _get_redis()
    if r is not None:
        try:
            r.delete(_key(session_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("redis_delete_failed", error=str(exc)[:200])
    _sessions.pop(session_id, None)
