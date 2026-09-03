"""Node: final confirmation and booking URL generation.

Session 14 patterns applied:
- **Human-in-the-loop**: the booking URL is the irreversible action of this
  system. The graph PAUSES (``awaiting_human=True``) and only proceeds on an
  explicit positive confirmation from the patient in the *next* turn.
- **Least privilege**: ``build_booking_url`` is registered as callable only
  from this node; every call is audited with an args digest.
"""

from __future__ import annotations

import structlog

from app.config import get_settings
from app.dependencies import get_tool_registry
from app.domain.schemas import ChatState, Node
from app.services import cliniweb

log = structlog.get_logger()

_POSITIVE = {"sí", "si", "yes", "confirmo", "correcto", "ok", "dale", "adelante", "claro"}
_NEGATIVE = {"no", "cancela", "cancelar", "espera", "corregir", "cambiar"}


async def confirm_node(state: ChatState) -> ChatState:
    settings = get_settings()
    last_user = next(
        (m["content"] for m in reversed(state.messages) if m["role"] == "user"), ""
    ).strip().lower()

    # Guard: the supervisor may route here before patient data was collected.
    if state.patient is None:
        log.warning("confirm_without_patient", session=state.session_id)
        state.current_node = Node.COLLECT
        state.reply = ""
        return state

    # --- HITL gate: first entry pauses and asks; only an explicit yes proceeds
    if settings.HITL_CONFIRM_BOOKING and not state.awaiting_human:
        state.awaiting_human = True
        state.current_node = Node.CONFIRM
        state.reply = (
            f"Por favor confirme su cita:\n\n"
            f"• Nombre: {state.patient.full_name}\n"
            f"• Síntomas: {state.patient.symptoms}\n"
            f"• Email: {state.patient.email}\n"
            f"• Fecha/hora: {state.slot_datetime}\n"
            f"• Localidad: {state.localidad_name}\n\n"
            "¿Confirma la cita?"
        )
        state.quick_replies = ["SÍ", "NO"]
        log.info("hitl_pause", session=state.session_id)
        return state

    if any(word in last_user for word in _NEGATIVE):
        state.awaiting_human = False
        state.current_node = Node.COLLECT
        state.reply = "Entendido, volvamos a revisar los datos. ¿Qué desea corregir?"
        log.info("hitl_rejected", session=state.session_id)
        return state

    if any(word in last_user for word in _POSITIVE):
        # --- Least privilege: audited tool call, only this node may book ---
        registry = get_tool_registry()
        args = {
            "patient_name": state.patient.full_name,
            "slot_datetime": state.slot_datetime,
            "localidad_id": state.localidad_id,
        }
        allowed = registry.check("build_booking_url", Node.CONFIRM.value, args)
        if not allowed:
            # Denial envelope: the flow survives, the denial is in the audit log.
            state.reply = "No fue posible generar el enlace de reserva. Contacte a la clínica."
            state.current_node = Node.CONFIRM
            log.error("booking_denied_by_privileges", session=state.session_id)
            return state

        url = cliniweb.build_booking_url(
            patient_name=state.patient.full_name,
            symptoms=state.patient.symptoms,
            email=state.patient.email,
            slot_datetime=state.slot_datetime,
            localidad_id=state.localidad_id,
        )
        state.booking_url = url
        state.completed = True
        state.awaiting_human = False
        state.current_node = Node.DONE
        log.info("appointment_confirmed", session=state.session_id, url=url)
        state.reply = (
            "¡Cita confirmada! 🎉 Use el enlace de abajo para finalizar su reserva. "
            "Gracias por usar el asistente."
        )
        return state

    # Ambiguous answer while paused: stay paused, re-ask.
    state.current_node = Node.CONFIRM
    state.reply = "¿Confirma la cita? Por favor responda Sí o No."
    state.quick_replies = ["SÍ", "NO"]
    return state
