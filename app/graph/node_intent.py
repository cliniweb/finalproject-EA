"""Node: classify patient intent."""

from __future__ import annotations

import json
import structlog

from app.domain.schemas import ChatState, IntentResult, Node
from app.services import llm
from app.services.triage import check_red_flags, EMERGENCY_REPLY

log = structlog.get_logger()

_SYSTEM = (
    "You are an intent classifier for a medical appointment chatbot. "
    "The patient may write in any language. Respond ONLY in Spanish. "
    "Classify the patient's message into one of: "
    "book_appointment, doctor_info, find_doctor, greeting, other. "
    "Use find_doctor when the patient describes symptoms, a health problem, "
    "distress, or someone needing medical help — even if vague or emotional. "
    "Use other ONLY when the message is clearly unrelated to health or appointments."
)


async def intent_node(state: ChatState) -> ChatState:
    last_user = next(
        (m["content"] for m in reversed(state.messages) if m["role"] == "user"), ""
    )
    log.debug(
        "intent_enter",
        session=state.session_id,
        last_user=last_user[:200],
        awaiting_human=state.awaiting_human,
        has_suggestions=bool(state.suggested_doctors),
        doctor_id=state.doctor_id or None,
        current_node=str(state.current_node),
    )

    # Triage: red-flag detection BEFORE any intent classification.
    # On a confirmed emergency, short-circuit the funnel entirely.
    red_flag = check_red_flags(last_user)
    if red_flag is not None:
        log.warning(
            "triage_emergency_shortcircuit",
            session=state.session_id,
            category=red_flag.category,
        )
        state.reply = EMERGENCY_REPLY
        state.current_node = Node.INTENT
        return state

    # HITL: if the graph is paused awaiting confirmation, route straight back
    # to confirm — the patient's message IS the human decision.
    if state.awaiting_human:
        log.debug("intent_shortcut_confirm", session=state.session_id, reason="awaiting_human")
        state.current_node = Node.CONFIRM
        state.reply = ""
        return state

    # A proactively offered slot is pending a SÍ/NO answer — route it
    # deterministically to datetime; the answer is not a new intent.
    if state.offered_slot:
        log.debug("intent_shortcut_offered_slot", session=state.session_id)
        state.current_node = Node.DATETIME
        state.reply = ""
        return state

    # Alternatives offered when the doctor has no availability — deterministic
    # button answers that require resetting parts of the funnel.
    normalized = last_user.strip().lower()
    if normalized == "elegir otra localidad":
        log.debug("intent_shortcut_change_location", session=state.session_id)
        state.localidad_id = None
        state.localidad_name = None
        state.locations_presented = False
        state.proactive_offer_done = False
        state.available_slots = []
        state.current_node = Node.LOCATION
        state.reply = ""
        return state
    if normalized == "elegir otro médico":
        log.debug("intent_shortcut_change_doctor", session=state.session_id)
        state.doctor_id = ""
        state.doctor_data = None
        state.localidades = []
        state.localidad_id = None
        state.localidad_name = None
        state.locations_presented = False
        state.proactive_offer_done = False
        state.available_slots = []
        state.current_node = Node.SUGGEST_DOCTOR
        state.reply = (
            "De acuerdo. Cuénteme qué síntomas tiene o qué especialista busca, "
            "y le sugeriré otros médicos disponibles."
        )
        return state

    # Doctor suggestion in progress: the message is a pick or a refinement.
    # This MUST take priority over the funnel shortcut — the patient may be
    # switching doctors mid-funnel (suggestions were re-shown on purpose).
    if state.suggested_doctors:
        log.debug(
            "intent_shortcut_suggest_doctor",
            session=state.session_id,
            reason="suggestions_pending",
            suggestion_count=len(state.suggested_doctors),
        )
        state.current_node = Node.SUGGEST_DOCTOR
        state.reply = ""
        return state

    # Booking funnel in progress: a doctor is chosen but the booking isn't
    # complete — the message belongs to the funnel, let the supervisor place it.
    if state.doctor_id and not state.completed:
        from app.agents.supervisor import decide_next

        log.debug(
            "intent_shortcut_funnel",
            session=state.session_id,
            reason="booking_in_progress",
        )
        state.current_node = decide_next(state, last_user)
        state.reply = ""
        return state

    result: IntentResult = llm.extract(
        IntentResult,
        system=_SYSTEM,
        user=last_user,
    )
    log.info("intent_classified", intent=result.intent, session=state.session_id)
    log.debug("intent_reason", session=state.session_id, reason=result.reason[:200])

    if result.intent == "greeting":
        if not state.doctor_id:
            state.reply = (
                "¡Hola! Soy su asistente médico. Cuénteme qué síntomas tiene o qué "
                "especialista busca, y le sugeriré médicos disponibles."
            )
            state.current_node = Node.INTENT
            return state
        doctor_title = _doctor_title(state.doctor_data)
        state.reply = (
            f"¡Hola! Soy el asistente de {doctor_title}. "
            "¿Desea hacer una cita o tiene alguna consulta sobre el médico?"
        )
        state.current_node = Node.INTENT

    elif result.intent == "doctor_info":
        state.current_node = Node.DOCTOR_INFO
        state.reply = ""  # doctor_info node will fill this

    elif result.intent == "find_doctor":
        state.current_node = Node.SUGGEST_DOCTOR
        state.reply = ""

    elif result.intent == "book_appointment":
        if not state.doctor_id:
            # No doctor yet: suggest one from the patient's description first.
            state.current_node = Node.SUGGEST_DOCTOR
            state.reply = ""
            return state
        # Supervisor decides WHERE in the booking funnel this message belongs
        # (location, datetime, collect or confirm) from a factual digest.
        from app.agents.supervisor import decide_next

        state.current_node = decide_next(state, last_user)
        state.reply = ""

    else:
        log.debug("intent_unhandled", session=state.session_id, intent=result.intent)
        state.reply = "Disculpe, no entendí su consulta. ¿Desea hacer una cita o tiene alguna pregunta sobre el médico?"
        state.current_node = Node.INTENT

    log.debug(
        "intent_exit",
        session=state.session_id,
        next_node=str(state.current_node),
        has_reply=bool(state.reply),
    )
    return state


def _doctor_title(doctor_data: dict | None) -> str:
    if not doctor_data:
        return "el médico"
    name = doctor_data.get("nombrePersona", "el médico")
    prefix = "la doctora" if doctor_data.get("sexo") == "F" else "el doctor"
    return f"{prefix} {name}"
