"""Node: present clinic locations and extract the patient's choice."""

from __future__ import annotations

import json
import structlog

from datetime import date, timedelta

from app.domain.schemas import ChatState, LocationChoice, Node
from app.services import llm

log = structlog.get_logger()

_SYSTEM_EXTRACT = (
    "Extrae la localidad elegida por el paciente de la siguiente lista JSON. "
    "Devuelve el id y nombre exactos de la localidad seleccionada."
)


async def location_node(state: ChatState) -> ChatState:
    localidades_json = json.dumps(state.localidades, ensure_ascii=False, indent=2)
    last_user = next(
        (m["content"] for m in reversed(state.messages) if m["role"] == "user"), ""
    )

    # First visit: present the list, then wait for the patient's choice.
    if not state.locations_presented:
        doctor_name = (state.doctor_data or {}).get("nombrePersona") or "seleccionado"
        state.reply = (
            f"Perfecto, ha elegido a {doctor_name}. "
            "Ahora seleccione la localidad donde desea su cita:"
        )
        # Structured options so the UI renders clickable buttons.
        state.location_options = [
            {"id": loc["id"], "nombre": loc["nombre"]} for loc in state.localidades
        ]
        state.locations_presented = True
        state.current_node = Node.LOCATION
        return state

    # Try to extract a choice from the latest user message
    try:
        choice: LocationChoice = llm.extract(
            LocationChoice,
            system=f"{_SYSTEM_EXTRACT}\n\nLOCALIDADES:\n{localidades_json}",
            user=last_user,
        )
        state.localidad_id = choice.localidad_id
        state.localidad_name = choice.localidad_name
        log.info("location_chosen", localidad_id=choice.localidad_id, session=state.session_id)
        # Instead of asking for a date, proactively check the calendar for the
        # next two weeks and offer the first available slot (see fetch_slots).
        today = date.today()
        state.date_start = today.isoformat()
        state.date_end = (today + timedelta(days=14)).isoformat()
        state.current_node = Node.FETCH_SLOTS
        state.reply = ""  # continue into fetch_slots in this same turn
    except Exception as exc:
        log.warning("location_extraction_failed", error=str(exc)[:200])
        state.reply = "No pude identificar la localidad. Por favor elija una de la lista:"
        state.location_options = [
            {"id": loc["id"], "nombre": loc["nombre"]} for loc in state.localidades
        ]
        state.current_node = Node.LOCATION

    return state
