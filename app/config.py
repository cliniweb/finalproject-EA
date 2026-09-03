"""Application settings — loaded from .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    LLM_PRIMARY_MODEL: str = "gpt-4o-mini"
    LLM_FALLBACK_MODEL: str = "gpt-4o"
    LLM_TIMEOUT: int = 30
    LLM_RETRIES: int = 2
    # Temperature split: patient-facing chat replies are warmer/friendlier;
    # system-facing structured extraction (supervisor, intent, judge, choices)
    # is deterministic and precise.
    LLM_CHAT_TEMPERATURE: float = 0.7
    LLM_EXTRACT_TEMPERATURE: float = 0.0

    CLINIWEB_API_BASE: str = "https://api.cliniweb.com/api"
    # dominioEmpresa / nombreCuenta — must always be minimed-administracion
    CLINIWEB_ACCOUNT: str = "minimed-administracion"
    CLINIWEB_LANGUAGE: str = "es"
    # API Key required on all Cliniweb requests (shared separately by Cliniweb)
    CLINIWEB_API_KEY: str | None = None
    CLINIWEB_API_KEY_HEADER: str = "X-Api-Key"
    # SAFETY: fake host — booking links must never hit the real booking site
    # from this environment. Deliberately points to a non-existent page.
    CLINIWEB_BOOKING_BASE: str = "https://testers.cliniweb.com/info"

    # --- CAG layer (exact + semantic in-memory caches) ---
    CACHE_TTL: int = 86400
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SEMANTIC_CACHE_THRESHOLD: float = 0.90
    SEMANTIC_CACHE_TTL: int = 86400
    # When True the semantic cache LOGS potential hits but does NOT serve them
    # (threshold calibration mode).
    SEMANTIC_CACHE_LOG_ONLY: bool = False

    # --- RAG layer (in-memory vector store + retrieval) ---
    RAG_TOP_K: int = 4
    # Similarity floor: chunks below this are noise. Zero kept chunks = the
    # answer is not in the knowledge base and the node refuses to answer.
    RAG_MIN_SCORE: float = 0.25
    # Hallucination gate: LLM-judge verifies the answer is grounded in the
    # retrieved context before serving it.
    HALLUCINATION_GATE_ENABLED: bool = True
    HALLUCINATION_JUDGE_MODEL: str = "gpt-4o-mini"

    # --- Agents layer (supervisor + least privilege + HITL) ---
    # The supervisor decision space is 5 Literal options over a short digest,
    # so a cheap non-reasoning model is the right tool.
    SUPERVISOR_MODEL: str = "gpt-4o-mini"
    # Hard ceiling on routing steps — the bound that makes an LLM router safe
    # in a graph with cyclic return edges.
    SUPERVISOR_MAX_STEPS: int = 12
    # False = a denied tool call returns a denial envelope the flow survives;
    # True = raises PrivilegeViolation and fails loudly.
    PRIVILEGE_STRICT: bool = False
    # Human-in-the-loop: pause before the irreversible booking action and
    # require an explicit patient confirmation to resume.
    HITL_CONFIRM_BOOKING: bool = True

    # --- Session persistence (optional Redis; falls back to in-memory) ---
    REDIS_ENABLED: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"
    # Session expiry in Redis, seconds (24h)
    REDIS_SESSION_TTL: int = 86400

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"


@lru_cache
def get_settings() -> Settings:
    return Settings()
