"""Hallucination gate — LLM-judge that verifies the answer is grounded.

Ported concept from the estimator (Session 11). A cheap model checks that
every factual claim in the generated answer is supported by the retrieved
context. On failure the caller serves a refusal instead of the answer —
better no answer than a wrong one in a medical domain.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from app.services import llm
from app.services.console import say

log = structlog.get_logger()

_JUDGE_SYSTEM = (
    "Eres un verificador estricto. Recibirás un CONTEXTO y una RESPUESTA. "
    "Determina si TODAS las afirmaciones factuales de la RESPUESTA están "
    "respaldadas por el CONTEXTO. Frases de cortesía o invitaciones a agendar "
    "cita no cuentan como afirmaciones factuales."
)


class GroundingVerdict(BaseModel):
    grounded: bool = Field(description="True si toda afirmación factual está en el contexto")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Afirmaciones de la respuesta que NO aparecen en el contexto",
    )


def check_grounding(*, answer: str, context: str, model: str | None = None) -> GroundingVerdict:
    """Return the judge's verdict. Degrades to 'grounded' on judge failure —
    the gate must never take the service down."""
    try:
        verdict = llm.extract(
            GroundingVerdict,
            system=_JUDGE_SYSTEM,
            user=f"CONTEXTO:\n{context}\n\nRESPUESTA:\n{answer}",
            model=model,
        )
        log.info(
            "grounding_checked",
            grounded=verdict.grounded,
            unsupported=len(verdict.unsupported_claims),
        )
        if verdict.grounded:
            say("⚖️ Juez antialucinación: respuesta APROBADA — todo está fundamentado en el contexto")
        else:
            say(
                f"🚨 Juez antialucinación: respuesta RECHAZADA — "
                f"{len(verdict.unsupported_claims)} afirmación(es) sin respaldo: "
                f"{verdict.unsupported_claims[:2]}"
            )
        return verdict
    except Exception as exc:  # noqa: BLE001
        log.warning("grounding_judge_failed", error=str(exc)[:200])
        say("⚠️ Juez antialucinación: FALLÓ la verificación — se degrada a 'aprobada' para no tumbar el servicio")
        return GroundingVerdict(grounded=True)
