"""Triage — hybrid red-flag detection (deterministic patterns + LLM verification).

Runs BEFORE intent classification. Deterministic regex list is cheap and
auditable; on a match, the LLM verifies to reduce false positives. If the
LLM verification itself fails, we fail SAFE (treat as emergency).
"""

from __future__ import annotations

import re

import structlog

from app.domain.schemas import RedFlagResult
from app.services import llm

log = structlog.get_logger()

# Deterministic, auditable red-flag patterns (Spanish). Each entry:
# (category, compiled regex). Keep patterns conservative — the LLM
# verification step filters false positives.
_RED_FLAG_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "dolor_toracico",
        re.compile(
            r"dolor\s+(en\s+el\s+|de\s+)?pecho.*(opresi|irradia|brazo|mand[ií]bula|sudor|falta\s+de\s+aire)"
            r"|(opresi[oó]n|presi[oó]n)\s+(en\s+el\s+)?pecho",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "disnea_subita",
        re.compile(
            r"(no\s+puedo|me\s+cuesta|dificultad\s+para)\s+respirar|falta\s+de\s+aire\s+(s[uú]bita|repentina|de\s+repente)|me\s+ahogo",
            re.IGNORECASE,
        ),
    ),
    (
        "focalizacion_neurologica",
        re.compile(
            r"(no\s+(puedo|siento))\s+(mover|el|la|un[a]?)\s*(brazo|pierna|lado|cara|mitad)"
            r"|cara\s+(ca[ií]da|torcida|dormida)"
            r"|(no\s+puedo|dificultad\s+para)\s+hablar"
            r"|p[eé]rdida\s+(s[uú]bita\s+)?de\s+(visi[oó]n|vista|fuerza|conciencia|conocimiento)"
            r"|peor\s+dolor\s+de\s+cabeza\s+de\s+mi\s+vida",
            re.IGNORECASE,
        ),
    ),
    (
        "sangrado_obstetrico",
        re.compile(
            r"(sangrado|sangro|hemorragia).*(embaraz|gestaci[oó]n|semanas)|embaraz.*(sangrado|sangro|hemorragia)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "fiebre_lactante",
        re.compile(
            r"(fiebre|calentura|temperatura).*(beb[eé]|reci[eé]n\s+nacid|lactante|(\d|un|dos|tres)\s*mes(es)?)"
            r"|(beb[eé]|reci[eé]n\s+nacid|lactante).*(fiebre|calentura)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "ideacion_suicida",
        re.compile(
            r"suicid|quitarme\s+la\s+vida|no\s+quiero\s+(seguir\s+)?vivi(r|endo)|hacerme\s+da[ñn]o|terminar\s+con\s+todo",
            re.IGNORECASE,
        ),
    ),
    (
        "sangrado_severo",
        re.compile(
            r"(vomito|vomitando|toso|tosiendo)\s+sangre|sangrado\s+(abundante|que\s+no\s+para)",
            re.IGNORECASE,
        ),
    ),
]

_VERIFY_SYSTEM = (
    "Eres un verificador de emergencias médicas para un chatbot de citas. "
    "Un filtro determinista detectó una posible bandera roja en el mensaje del paciente. "
    "Confirma si el mensaje describe una situación de emergencia que requiere atención "
    "inmediata (dolor torácico opresivo/irradiado, disnea súbita, focalización neurológica, "
    "sangrado obstétrico, fiebre en lactante, ideación suicida, sangrado severo). "
    "Marca is_emergency=false SOLO si claramente NO es una emergencia (p. ej., relata un "
    "episodio antiguo ya resuelto, o es una pregunta hipotética/administrativa)."
)

EMERGENCY_REPLY = (
    "⚠️ Por lo que me describe, su situación podría ser una EMERGENCIA médica. "
    "Este asistente no puede ayudarle en este caso: por favor llame ahora mismo al "
    "número de emergencias (911) o acuda de inmediato al servicio de urgencias más cercano. "
    "No espere una cita.\n\n"
    "Recuerde: este chat es solo un asistente para agendar citas; no es una consulta "
    "médica ni un servicio de triaje."
)


def check_red_flags(message: str) -> RedFlagResult | None:
    """Return a RedFlagResult if the message trips a red flag, else None.

    Hybrid: deterministic pattern match (auditable) → LLM verification.
    Fail-safe: if the LLM verification errors out, treat as emergency.
    """
    matched_category = None
    for category, pattern in _RED_FLAG_PATTERNS:
        if pattern.search(message):
            matched_category = category
            break
    if matched_category is None:
        return None

    log.info("red_flag_pattern_match", category=matched_category, message=message[:200])
    try:
        result: RedFlagResult = llm.extract(
            RedFlagResult,
            system=_VERIFY_SYSTEM,
            user=f"Categoría detectada: {matched_category}\nMensaje del paciente: {message}",
        )
    except Exception as exc:  # noqa: BLE001
        # Fail SAFE — assume emergency if we cannot verify.
        log.warning("red_flag_verify_failed_failsafe", error=str(exc)[:200])
        return RedFlagResult(
            is_emergency=True,
            category=matched_category,
            reason="Verificación LLM no disponible — se asume emergencia por seguridad.",
        )

    if result.is_emergency:
        result.category = result.category or matched_category
        log.warning(
            "red_flag_confirmed",
            category=result.category,
            reason=result.reason[:200],
        )
        return result

    log.info(
        "red_flag_discarded_by_llm",
        category=matched_category,
        reason=result.reason[:200],
    )
    return None
