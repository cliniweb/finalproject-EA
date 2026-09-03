"""LLM-driven supervisor router (ported from estimator Session 14 pattern).

The supervisor owns the control flow but its decision space is a small
``Literal`` over a short factual digest of the state — so a cheap
non-reasoning model is the right tool. A hard step ceiling
(``SUPERVISOR_MAX_STEPS``) makes an LLM router safe in a graph with cyclic
return edges: the graph can never loop forever.
"""

from __future__ import annotations

from typing import Literal

import structlog
from pydantic import BaseModel, Field

from app.config import get_settings
from app.domain.schemas import ChatState, Node
from app.services import llm
from app.services.console import say

log = structlog.get_logger()

_SYSTEM = (
    "Eres el supervisor de un chatbot médico de citas. Recibirás un RESUMEN "
    "del estado de la conversación y el último mensaje del paciente. "
    "Decide cuál agente debe actuar a continuación:\n"
    "- suggest_doctor: el paciente aún no tiene médico elegido y describe síntomas\n"
    "- doctor_info: el paciente pregunta información sobre el médico\n"
    "- location: hay que elegir/confirmar la localidad de la cita\n"
    "- datetime: hay que elegir fecha u hora de la cita\n"
    "- collect: hay que recopilar datos del paciente (nombre, síntomas, email)\n"
    "- confirm: todo listo, falta la confirmación final\n"
    "Elige exactamente uno."
)


class SupervisorDecision(BaseModel):
    next_agent: Literal["suggest_doctor", "doctor_info", "location", "datetime", "collect", "confirm"]
    reason: str = Field(description="Una frase explicando la decisión")


def _digest(state: ChatState) -> str:
    """Short factual digest — the supervisor sees facts, not the transcript."""
    return (
        f"- Médico seleccionado: {'SÍ' if state.doctor_id else 'NO'}\n"
        f"- Localidad elegida: {state.localidad_name or 'NO'}\n"
        f"- Slots disponibles mostrados: {'SÍ' if state.available_slots else 'NO'}\n"
        f"- Horario ofrecido pendiente de respuesta SÍ/NO: {state.offered_slot or 'NO'}\n"
        f"- Hora de cita elegida: {state.slot_datetime or 'NO'}\n"
        f"- Datos del paciente completos: {'SÍ' if state.patient else 'NO'}\n"
        f"- Turnos transcurridos: {len(state.messages) // 2}"
    )


def decide_next(state: ChatState, last_user: str) -> Node:
    """Ask the supervisor model which agent acts next. Falls back to a
    deterministic rule if the model call fails — the router must never take
    the conversation down."""
    settings = get_settings()

    # Hard ceiling: beyond max steps, force the flow forward deterministically.
    steps = len(state.messages) // 2
    if steps >= settings.SUPERVISOR_MAX_STEPS:
        log.warning("supervisor_step_ceiling", steps=steps, session=state.session_id)
        nxt = _deterministic_next(state)
        say(
            f"🧭 Supervisor: TECHO DE PASOS alcanzado ({steps}) — "
            f"avance forzado por regla determinista → {nxt.value}"
        )
        return nxt

    try:
        decision = llm.extract(
            SupervisorDecision,
            system=_SYSTEM,
            user=f"RESUMEN DEL ESTADO:\n{_digest(state)}\n\nÚLTIMO MENSAJE DEL PACIENTE:\n{last_user}",
            model=settings.SUPERVISOR_MODEL,
        )
        log.info(
            "supervisor_decision",
            next_agent=decision.next_agent,
            reason=decision.reason[:120],
            session=state.session_id,
        )
        say(f"🧭 Supervisor decidió → {decision.next_agent} (razón: {decision.reason[:100]})")
        return Node(decision.next_agent)
    except Exception as exc:  # noqa: BLE001
        log.warning("supervisor_failed", error=str(exc)[:200])
        nxt = _deterministic_next(state)
        say(f"⚠️ Supervisor LLM FALLÓ — fallback determinista → {nxt.value}")
        return nxt


def _deterministic_next(state: ChatState) -> Node:
    """Rule-based fallback mirroring the booking funnel order."""
    if not state.doctor_id:
        return Node.SUGGEST_DOCTOR
    if not state.localidad_id:
        return Node.LOCATION
    if not state.slot_datetime:
        return Node.DATETIME
    if not state.patient:
        return Node.COLLECT
    return Node.CONFIRM
