"""Node: collect patient name, symptoms and email one field at a time."""

from __future__ import annotations

import structlog

from app.domain.schemas import ChatState, PatientData, Node
from app.services import llm

log = structlog.get_logger()

_SYSTEM = (
    "Eres un asistente médico. Debes recopilar tres datos del paciente uno a uno: "
    "1) nombre completo, 2) síntomas o motivo de consulta, 3) correo electrónico. "
    "Cuando tengas los tres datos, extráelos en el formato solicitado. "
    "Si aún falta algún dato, pregunta únicamente por el siguiente. "
    "Responde siempre en español."
)

_EXTRACT_SYSTEM = (
    "Extrae los datos del paciente de la conversación: nombre completo, síntomas y correo electrónico. "
    "El texto está en español. Deja un campo VACÍO si el paciente no lo ha "
    "proporcionado explícitamente. Respuestas como 'SÍ', 'NO' u 'OK' son "
    "confirmaciones, NUNCA nombres, síntomas ni correos."
)

# Bare confirmation tokens — never valid values for any patient field.
_YESNO_TOKENS = {"no", "sí", "si", "yes", "ok", "dale", "claro", "confirmo"}


def _sanitize(value: str) -> str:
    return "" if value.strip().lower() in _YESNO_TOKENS else value.strip()


async def collect_node(state: ChatState) -> ChatState:
    last_user = next(
        (m["content"] for m in reversed(state.messages) if m["role"] == "user"), ""
    )

    # Try to extract all three fields from the conversation so far
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state.messages[-10:]
    )
    try:
        patient: PatientData = llm.extract(
            PatientData,
            system=_EXTRACT_SYSTEM,
            user=conversation_text,
        )
        # Nunca aceptar tokens de confirmación ('NO', 'SÍ'...) como datos reales.
        patient.full_name = _sanitize(patient.full_name)
        patient.symptoms = _sanitize(patient.symptoms)
        patient.email = _sanitize(patient.email)
        # El paciente ya describió sus síntomas al inicio del flujo — no hay que
        # pedírselos de nuevo si la extracción no los encontró en los últimos turnos.
        if not patient.symptoms and state.symptoms_hint:
            patient.symptoms = state.symptoms_hint
        # Validate that all fields are non-empty
        if patient.full_name and patient.symptoms and patient.email:
            state.patient = patient
            log.info("patient_data_collected", session=state.session_id)
            # Continue into confirm in this same turn — the confirm node owns
            # the HITL confirmation prompt (avoids asking twice).
            state.current_node = Node.CONFIRM
            state.reply = ""
            return state
    except Exception:
        pass  # Not all fields present yet — ask for the next one

    # Ask for the next missing field
    state.reply = llm.chat(system=_SYSTEM, messages=state.messages)
    state.current_node = Node.COLLECT
    return state
