"""Node: call the Cliniweb API and present available slots to the patient."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import structlog

from app.domain.schemas import ChatState, Node
from app.services import cliniweb

log = structlog.get_logger()

# Días y meses en español — independiente del locale del sistema operativo.
_DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

# Ventana extendida cuando las próximas dos semanas no tienen turnos.
_EXTENDED_DAYS = 90


def _spanish_date(dt: datetime | date) -> str:
    return f"{_DIAS[dt.weekday()]} {dt.day} de {_MESES[dt.month - 1]} de {dt.year}"


def _spanish_time(dt: datetime) -> str:
    hour12 = dt.hour % 12 or 12
    ampm = "a. m." if dt.hour < 12 else "p. m."
    return f"{hour12}:{dt.minute:02d} {ampm}"


def _slot_datetime(slot: dict) -> str:
    return slot.get("fechaHora") or slot.get("fecha") or str(slot)


def _slot_label(dt_raw: str) -> str:
    try:
        dt = datetime.fromisoformat(dt_raw)
        return f"{_spanish_date(dt)} — {_spanish_time(dt)}"
    except Exception:
        return dt_raw


def _distinct_dates(slots: list[dict], limit: int = 6) -> list[str]:
    """Distinct available dates as Spanish labels, in order of appearance."""
    seen: list[str] = []
    labels: list[str] = []
    for slot in slots:
        day = _slot_datetime(slot)[:10]
        if day and day not in seen:
            seen.append(day)
            try:
                labels.append(_spanish_date(date.fromisoformat(day)))
            except Exception:
                labels.append(day)
        if len(seen) >= limit:
            break
    return labels


async def _fetch(state: ChatState, date_start: str, date_end: str) -> list[dict]:
    chosen = next(
        (loc for loc in state.localidades if loc.get("id") == state.localidad_id), {}
    )
    empresa_id = str(chosen.get("idEmpresa", ""))
    responsable_id = str(chosen.get("idPersona", ""))
    if not empresa_id or not responsable_id:
        raise LookupError("missing empresa/responsable ids for chosen localidad")
    return await cliniweb.fetch_available_slots(
        date_start=date_start,
        date_end=date_end,
        empresa_id=empresa_id,
        responsable_id=responsable_id,
        localidad_id=state.localidad_id,
    )


async def fetch_slots_node(state: ChatState) -> ChatState:
    date_start, date_end = state.date_start, state.date_end
    if not date_start or not date_end:
        state.reply = "Error interno al procesar la fecha. Por favor intente de nuevo."
        state.current_node = Node.DATETIME
        return state

    try:
        slots = await _fetch(state, date_start, date_end)
    except LookupError:
        # Defensivo: el perfil no traía los IDs o el localidad_id quedó obsoleto
        # (p. ej. sesión restaurada tras un cambio de perfil). Volver a preguntar
        # la localidad es la única recuperación segura — nunca adivinar IDs.
        log.warning(
            "fetch_slots_missing_ids",
            session=state.session_id,
            localidad_id=state.localidad_id,
        )
        state.reply = "Error interno al procesar la localidad. Por favor elija la localidad de nuevo."
        state.current_node = Node.LOCATION
        return state
    except Exception as exc:
        log.warning("fetch_slots_failed", error=str(exc)[:200])
        state.reply = "No pude obtener los horarios disponibles. ¿Desea intentar con otra fecha?"
        state.current_node = Node.DATETIME
        return state

    # --- Tier 1: proactive offer of the first available slot (SÍ/NO) ---
    if not state.proactive_offer_done:
        state.proactive_offer_done = True
        extended = False
        if not slots:
            # Nothing in the next two weeks — extend the search before giving up.
            try:
                today = date.today()
                slots = await _fetch(
                    state,
                    today.isoformat(),
                    (today + timedelta(days=_EXTENDED_DAYS)).isoformat(),
                )
                extended = True
            except Exception as exc:  # noqa: BLE001
                log.warning("fetch_slots_extended_failed", error=str(exc)[:200])
                slots = []
        if slots:
            first = _slot_datetime(slots[0])
            state.offered_slot = first
            state.available_slots = []
            state.current_node = Node.DATETIME
            log.info("slot_offered", slot=first, extended=extended, session=state.session_id)
            prefix = (
                "Revisé el calendario del médico. No hay turnos en las próximas dos "
                "semanas, pero el primer horario disponible es: "
                if extended
                else "Revisé el calendario del médico. El primer horario disponible es: "
            )
            state.reply = f"{prefix}**{_slot_label(first)}**. ¿Le sirve ese horario?"
            state.quick_replies = ["SÍ", "NO"]
            return state
        # No availability at all in the extended window — the doctor's calendar
        # is effectively closed. Offer alternatives instead of a dead end.
        log.info("doctor_no_availability", session=state.session_id, doctor_id=state.doctor_id)
        state.reply = (
            f"Lo siento, este médico no tiene turnos disponibles en los próximos "
            f"{_EXTENDED_DAYS} días. ¿Desea elegir otra localidad u otro médico?"
        )
        state.quick_replies = ["Elegir otra localidad", "Elegir otro médico"]
        state.current_node = Node.DATETIME
        return state

    if not slots:
        # --- Tier 3: no slots for the requested day — offer available dates ---
        try:
            today = date.today()
            window = await _fetch(
                state, today.isoformat(), (today + timedelta(days=_EXTENDED_DAYS)).isoformat()
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_slots_window_failed", error=str(exc)[:200])
            window = []
        dates = _distinct_dates(window)
        if dates:
            state.reply = (
                "No hay horarios disponibles para ese día. "
                "Estas son las fechas con disponibilidad:"
            )
            state.quick_replies = dates
        else:
            log.info("doctor_no_availability", session=state.session_id, doctor_id=state.doctor_id)
            state.reply = (
                f"Lo siento, este médico no tiene turnos disponibles en los próximos "
                f"{_EXTENDED_DAYS} días. ¿Desea elegir otra localidad u otro médico?"
            )
            state.quick_replies = ["Elegir otra localidad", "Elegir otro médico"]
        state.current_node = Node.DATETIME
        state.available_slots = []
        return state

    # --- Tier 2: present the slots for the day the patient asked for ---
    state.available_slots = slots
    state.current_node = Node.DATETIME  # Patient will choose a slot next
    log.info("slots_fetched", count=len(slots), session=state.session_id)
    state.reply = "Estos son los horarios disponibles. Seleccione uno:"
    # Clickable buttons — max 8 to keep the list manageable.
    state.quick_replies = [_slot_label(_slot_datetime(s)) for s in slots[:8]]
    return state
