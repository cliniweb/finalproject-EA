"""Node: parse natural-language date input and fetch available slots."""

from __future__ import annotations

import json
from datetime import datetime

import structlog

from app.domain.schemas import ChatState, DateRangeResult, SlotChoice, YesNoResult, Node
from app.services import llm, cliniweb

log = structlog.get_logger()

_SYSTEM_DATE = (
    "Eres un asistente que convierte fechas en lenguaje natural a formato yyyy-MM-dd. "
    "Calcula correctamente 'esta semana', 'la próxima semana', 'mañana', etc. "
    f"Hoy es {{today}}. Muestra el razonamiento paso a paso."
)

_SYSTEM_SLOT = (
    "El paciente quiere elegir una hora de la siguiente lista de slots disponibles. "
    "Extrae el slot exacto que eligió en formato yyyy-MM-dd HH:mm."
)


async def datetime_node(state: ChatState) -> ChatState:
    last_user = next(
        (m["content"] for m in reversed(state.messages) if m["role"] == "user"), ""
    )
    today = datetime.now().strftime("%A %d de %B de %Y")

    # A single slot was proactively offered with a SÍ/NO control.
    if state.offered_slot:
        offered = state.offered_slot
        state.offered_slot = None
        try:
            answer: YesNoResult = llm.extract(
                YesNoResult,
                system=(
                    "El paciente respondió a la oferta de un horario de cita. "
                    "Determina si aceptó el horario propuesto."
                ),
                user=last_user,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("offer_answer_failed", error=str(exc)[:200])
            answer = YesNoResult(yes=False)
        if answer.yes:
            state.slot_datetime = offered[:16].replace("T", " ")
            log.info("offered_slot_accepted", slot=offered, session=state.session_id)
            state.current_node = Node.COLLECT
            state.reply = (
                f"Cita reservada para el {state.slot_datetime}. "
                "Ahora necesito algunos datos. ¿Cuál es su nombre completo?"
            )
            return state
        log.info("offered_slot_rejected", slot=offered, session=state.session_id)
        state.reply = "Entendido. ¿Qué día le convendría para su cita?"
        state.current_node = Node.DATETIME
        return state

    # If we already have slots, the patient is choosing one
    if state.available_slots:
        try:
            choice: SlotChoice = llm.extract(
                SlotChoice,
                system=f"{_SYSTEM_SLOT}\n\nSLOTS:\n{json.dumps(state.available_slots, ensure_ascii=False)}",
                user=last_user,
            )
            state.slot_datetime = choice.slot_datetime
            log.info("slot_chosen", slot=choice.slot_datetime, session=state.session_id)
            state.current_node = Node.COLLECT
            state.reply = (
                f"Cita reservada para el {choice.slot_datetime}. "
                "Ahora necesito algunos datos. ¿Cuál es su nombre completo?"
            )
        except Exception as exc:
            log.warning("slot_extraction_failed", error=str(exc)[:200])
            state.reply = "No pude identificar el horario. Por favor indique exactamente cuál desea."
        return state

    # Parse the date range from the patient's message
    try:
        date_result: DateRangeResult = llm.extract(
            DateRangeResult,
            system=_SYSTEM_DATE.format(today=today),
            user=last_user,
        )
        log.info(
            "date_parsed",
            start=date_result.date_start,
            end=date_result.date_end,
            session=state.session_id,
        )
        state.current_node = Node.FETCH_SLOTS
        # Store dates in dedicated state fields; keep reply empty so the
        # graph routes onward to fetch_slots instead of ending the turn.
        state.date_start = date_result.date_start
        state.date_end = date_result.date_end
        state.reply = ""
    except Exception as exc:
        log.warning("date_parsing_failed", error=str(exc)[:200])
        state.reply = "No pude interpretar la fecha. ¿Podría indicarme la fecha o rango de fechas?"

    return state
