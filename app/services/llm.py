"""LLM helpers — LiteLLM for routing, Instructor for structured extraction."""

from __future__ import annotations

import time
from typing import TypeVar, Type

import instructor
import litellm
import structlog
from openai import OpenAI
from pydantic import BaseModel

from app.config import get_settings
from app.services.console import say

log = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)

# Instructor client wrapping OpenAI (used for structured extraction)
_instructor_client: instructor.Instructor | None = None


def get_instructor_client() -> instructor.Instructor:
    global _instructor_client
    if _instructor_client is None:
        settings = get_settings()
        say(
            f"🤖 LLM: creando cliente OpenAI/Instructor "
            f"(timeout={settings.LLM_TIMEOUT}s, retries={settings.LLM_RETRIES}, "
            f"api_key={'CONFIGURADA (' + str(len(settings.OPENAI_API_KEY)) + ' chars)' if settings.OPENAI_API_KEY else '⚠️ VACÍA'})"
        )
        _instructor_client = instructor.from_openai(
            OpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.LLM_TIMEOUT,
                max_retries=settings.LLM_RETRIES,
            )
        )
        say("🤖 LLM: cliente Instructor creado OK")
    return _instructor_client


def extract(
    response_model: Type[T],
    system: str,
    user: str,
    model: str | None = None,
    temperature: float | None = None,
) -> T:
    """Call the LLM and extract a structured Pydantic object via Instructor.

    System-facing path: defaults to LLM_EXTRACT_TEMPERATURE (0.0) so routing,
    classification and judging are deterministic and precise.
    """
    settings = get_settings()
    model = model or settings.LLM_PRIMARY_MODEL
    temperature = temperature if temperature is not None else settings.LLM_EXTRACT_TEMPERATURE
    say(
        f"🤖 LLM extract: modelo={model}, schema={response_model.__name__}, "
        f"temp={temperature}, user_len={len(user)}"
    )
    log.debug(
        "llm_extract",
        model=model,
        temperature=temperature,
        response_model=response_model.__name__,
        system_preview=system[:150],
        user_preview=user[:150],
        user_len=len(user),
    )
    client = get_instructor_client()
    say(f"🤖 LLM extract: >>> enviando petición a OpenAI ({response_model.__name__})…")
    t0 = time.monotonic()
    try:
        result = client.chat.completions.create(
            model=model,
            response_model=response_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_retries=settings.LLM_RETRIES,
        )
    except Exception as exc:
        say(
            f"❌ LLM extract: FALLÓ tras {time.monotonic() - t0:.1f}s — "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )
        raise
    say(
        f"🤖 LLM extract: <<< respuesta en {time.monotonic() - t0:.1f}s — "
        f"{result.model_dump()}"
    )
    log.debug(
        "llm_extract_result",
        response_model=response_model.__name__,
        result=result.model_dump(),
    )
    return result


def chat(
    system: str,
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Free-form chat completion via LiteLLM (with automatic fallback).

    Patient-facing path: defaults to LLM_CHAT_TEMPERATURE (0.7) so replies
    sound warm and natural instead of robotic.
    """
    settings = get_settings()
    primary = model or settings.LLM_PRIMARY_MODEL
    temperature = temperature if temperature is not None else settings.LLM_CHAT_TEMPERATURE
    log.debug(
        "llm_chat",
        model=primary,
        temperature=temperature,
        system_preview=system[:150],
        message_count=len(messages),
    )
    try:
        say(f"🤖 LLM chat: >>> enviando petición a {primary} ({len(messages)} mensajes)…")
        t0 = time.monotonic()
        resp = litellm.completion(
            model=primary,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=temperature,
            timeout=settings.LLM_TIMEOUT,
            num_retries=settings.LLM_RETRIES,
        )
        content = resp.choices[0].message.content
        say(f"🤖 LLM chat: <<< respuesta en {time.monotonic() - t0:.1f}s — \"{(content or '')[:100]}\"")
        log.debug("llm_chat_result", model=primary, reply_preview=(content or "")[:150])
        return content
    except Exception as exc:
        say(
            f"⚠️ LLM chat: modelo primario FALLÓ ({type(exc).__name__}: {str(exc)[:150]}) — "
            f"reintentando con fallback {settings.LLM_FALLBACK_MODEL}…"
        )
        log.warning("llm_primary_failed", error=str(exc)[:200], fallback=settings.LLM_FALLBACK_MODEL)
        t0 = time.monotonic()
        resp = litellm.completion(
            model=settings.LLM_FALLBACK_MODEL,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=temperature,
            timeout=settings.LLM_TIMEOUT,
        )
        say(f"🤖 LLM chat: <<< fallback respondió en {time.monotonic() - t0:.1f}s")
        return resp.choices[0].message.content
