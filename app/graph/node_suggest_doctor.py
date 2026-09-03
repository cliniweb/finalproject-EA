"""Node: suggest doctors from the patient's symptom description.

Flow:
1. No suggestions yet → extract symptoms/specialty, search /api/doctores,
   present a numbered list of matching doctors.
2. Suggestions shown → try to extract the patient's choice; on a clear pick,
   load that doctor's profile and hand off to the supervisor funnel.
"""

from __future__ import annotations

import json

import structlog

from app.dependencies import get_rag_ingest
from app.domain.schemas import ChatState, DoctorChoice, Node, SymptomExtraction
from app.services import cliniweb, llm

log = structlog.get_logger()

# Especialidades registradas en la cuenta (minimed-administracion). La cuenta NO
# tiene médicos de "Medicina General" — el equivalente disponible para síntomas
# generales de adultos es Medicina Interna. (Verificado contra la API y la BD.)
_AVAILABLE_SPECIALTIES = [
    "Medicina Interna",
    "Pediatría",
    "Cardiología",
    "Dermatología",
    "Ginecología y Obstetricia",
    "Urología",
    "Nefrología",
    "Oncología",
    "Geriatría",
    "Nutrición",
    "Ortopedia y Traumatología",
    "Cirugía General",
    "Cirugía Cardiovascular",
    "Cirugía Plástica",
]

_EXTRACT_SYSTEM = (
    "Eres un asistente de enrutamiento para citas médicas (NO haces diagnósticos). "
    "El paciente describe un problema de salud. Extrae los síntomas y sugiere la "
    "especialidad médica más probable para orientar la búsqueda, en español. "
    "Elige preferentemente entre las especialidades disponibles en esta clínica: "
    f"{', '.join(_AVAILABLE_SPECIALTIES)}. "
    "Es solo una sugerencia de enrutamiento; si hay ambigüedad en un adulto "
    "prefiere 'Medicina Interna' y en un niño 'Pediatría'."
)

_CHOICE_SYSTEM = (
    "El paciente vio una lista de médicos sugeridos y respondió. Determina si "
    "eligió claramente a uno de ellos (por número, nombre o descripción). "
    "Si no eligió a ninguno con claridad, marca chosen=false.\n\n"
    "MÉDICOS SUGERIDOS:\n{doctors}"
)


async def suggest_doctor_node(state: ChatState) -> ChatState:
    last_user = next(
        (m["content"] for m in reversed(state.messages) if m["role"] == "user"), ""
    )
    log.debug(
        "suggest_doctor_enter",
        session=state.session_id,
        current_node=str(state.current_node),
        last_user=last_user[:200],
        message_count=len(state.messages),
        has_suggestions=bool(state.suggested_doctors),
        suggestion_count=len(state.suggested_doctors or []),
        symptoms_hint=(state.symptoms_hint or "")[:120],
    )

    # --- Phase B: patient is answering a shown suggestion list ---
    if state.suggested_doctors:
        log.debug(
            "suggest_doctor_phase_b",
            session=state.session_id,
            suggested_ids=[d.get("doctor_id") for d in state.suggested_doctors],
            user_answer=last_user[:200],
        )
        try:
            choice: DoctorChoice = llm.extract(
                DoctorChoice,
                system=_CHOICE_SYSTEM.format(
                    doctors=json.dumps(state.suggested_doctors, ensure_ascii=False)
                ),
                user=last_user,
            )
            log.debug(
                "doctor_choice_extracted",
                session=state.session_id,
                chosen=choice.chosen,
                doctor_id=choice.doctor_id,
                raw_choice=choice.model_dump(),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("doctor_choice_failed", error=str(exc)[:200])
            choice = DoctorChoice(chosen=False)

        valid_ids = {d.get("doctor_id") for d in state.suggested_doctors}
        if choice.chosen and choice.doctor_id in valid_ids:
            log.debug(
                "doctor_choice_accepted",
                session=state.session_id,
                doctor_id=choice.doctor_id,
            )
            return await _select_doctor(state, choice.doctor_id)

        if choice.chosen and choice.doctor_id:
            # LLM hallucinated an id that is not in the shown list — re-prompt.
            log.warning(
                "doctor_choice_invalid_id",
                session=state.session_id,
                doctor_id=choice.doctor_id,
                valid_ids=sorted(i for i in valid_ids if i),
            )
            state.reply = (
                "No pude identificar a ese médico en la lista. "
                "Por favor seleccione uno de los médicos sugeridos."
            )
            state.current_node = Node.SUGGEST_DOCTOR
            return state

        # Not a choice — treat it as a new/refined symptom description.
        log.debug(
            "doctor_choice_rejected_fallback_to_search",
            session=state.session_id,
            chosen=choice.chosen,
            doctor_id=choice.doctor_id,
            note="clearing suggestions and re-running symptom extraction",
        )
        state.suggested_doctors = []

    # --- Phase A: extract symptoms and search ---
    log.debug("suggest_doctor_phase_a", session=state.session_id, input_text=last_user[:200])
    try:
        extraction: SymptomExtraction = llm.extract(
            SymptomExtraction, system=_EXTRACT_SYSTEM, user=last_user
        )
        state.symptoms_hint = extraction.symptom_text
        query = extraction.suspected_specialty
        log.debug(
            "symptom_extracted",
            session=state.session_id,
            symptom_text=(extraction.symptom_text or "")[:200],
            suspected_specialty=extraction.suspected_specialty,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("symptom_extraction_failed", error=str(exc)[:200])
        query = last_user

    # The search endpoint returns clinical concepts (specialties, diagnoses,
    # procedures...) mixed with — or instead of — real doctors. When only
    # concepts come back, resolve them via /perfiles-publicos/{tipo}/{concepto}
    # to obtain the actual doctor profiles.
    async def _search(text: str) -> tuple[list, list]:
        raw = await cliniweb.search_doctors_by_text(text)
        docs = cliniweb.simplify_search_results(raw)
        if not docs:
            for concept in cliniweb.extract_concepts(raw)[:3]:
                log.debug(
                    "doctor_search_resolving_concept",
                    session=state.session_id,
                    tipo=concept["tipo"],
                    concepto=concept["concepto"],
                    clase=concept["clase"],
                )
                try:
                    perfiles = await cliniweb.search_profiles_by_concept(
                        concept["tipo"], concept["concepto"]
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "concept_resolution_failed",
                        tipo=concept["tipo"],
                        concepto=concept["concepto"],
                        error=str(exc)[:200],
                    )
                    continue
                docs = cliniweb.simplify_profile_results(perfiles)
                if docs:
                    break
        return raw, docs

    try:
        log.debug("doctor_search_start", session=state.session_id, query=query[:120])
        medicos, suggestions = await _search(query)
        log.debug(
            "doctor_search_done",
            session=state.session_id,
            query=query[:120],
            raw_result_count=len(medicos or []),
            doctor_count=len(suggestions),
        )
        if not suggestions and query != last_user:
            log.debug(
                "doctor_search_retry_with_raw_text",
                session=state.session_id,
                fallback_query=last_user[:120],
            )
            medicos, suggestions = await _search(last_user)
            log.debug(
                "doctor_search_retry_done",
                session=state.session_id,
                raw_result_count=len(medicos or []),
                doctor_count=len(suggestions),
            )
        if not suggestions and query.lower() not in ("medicina interna", "medicina general"):
            # Fallback seguro: la cuenta no tiene Medicina General; Medicina
            # Interna es la especialidad general disponible para adultos.
            log.debug("doctor_search_fallback_internal_medicine", session=state.session_id)
            medicos, suggestions = await _search("Medicina Interna")
    except Exception as exc:  # noqa: BLE001
        log.warning("doctor_search_failed", error=str(exc)[:200])
        state.reply = (
            "Lo siento, no pude buscar médicos en este momento. "
            "Por favor intente de nuevo en unos minutos."
        )
        state.current_node = Node.INTENT
        return state

    log.debug(
        "search_results_simplified",
        session=state.session_id,
        raw_count=len(medicos or []),
        simplified_count=len(suggestions),
        suggestion_ids=[d.get("doctor_id") for d in suggestions],
    )
    if not suggestions:
        log.debug("no_suggestions_reprompting", session=state.session_id, query=query[:120])
        state.reply = (
            "No encontré médicos para esa descripción. Las especialidades "
            "disponibles en la clínica son: "
            f"{', '.join(_AVAILABLE_SPECIALTIES)}. "
            "¿Con cuál de ellas desea agendar, o prefiere describir sus síntomas de otra forma?"
        )
        state.current_node = Node.SUGGEST_DOCTOR
        return state

    state.suggested_doctors = suggestions
    state.reply = (
        "Según lo que me describe, estos médicos podrían ayudarle (es solo una "
        "sugerencia de orientación, no un diagnóstico). Seleccione uno de la lista:"
    )
    # Shown by the UI AFTER the clickable doctor list.
    state.reply_footnote = (
        "Si prefiere, también puedo buscarle un médico de Medicina General u otra "
        "especialidad.\n\n"
        "ℹ️ Recuerde: este asistente no realiza consultas médicas ni triaje. "
        "Si sus síntomas empeoran o son graves, acuda a urgencias."
    )
    state.current_node = Node.SUGGEST_DOCTOR
    log.info(
        "doctors_suggested",
        session=state.session_id,
        query=query[:80],
        count=len(suggestions),
    )
    return state


async def _select_doctor(state: ChatState, doctor_id: str) -> ChatState:
    """Load the chosen doctor's profile and continue into the booking funnel."""
    log.debug("select_doctor_start", session=state.session_id, doctor_id=doctor_id)
    try:
        raw = await cliniweb.fetch_doctor(doctor_id)
        log.debug(
            "doctor_profile_fetched",
            session=state.session_id,
            doctor_id=doctor_id,
            nombre=raw.get("nombrePersona"),
            id_empresa=raw.get("idEmpresa"),
            top_level_keys=sorted(raw.keys())[:20],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("doctor_profile_failed", doctor_id=doctor_id, error=str(exc)[:200])
        state.reply = (
            "No pude cargar el perfil de ese médico. "
            "¿Desea elegir otro de la lista?"
        )
        state.current_node = Node.SUGGEST_DOCTOR
        return state

    state.doctor_id = doctor_id
    state.doctor_data = cliniweb.simplify_doctor_data(raw)
    # Los IDs de empresa/responsable son por localidad — se resuelven después
    # de que el paciente elige la localidad (ver node_fetch_slots / extract_localidades).
    state.localidades = cliniweb.extract_localidades(raw)
    state.suggested_doctors = []
    # Reset del embudo: si el paciente cambió de médico a mitad del flujo, la
    # localidad, los horarios y la oferta proactiva pertenecen al médico anterior.
    state.localidad_id = None
    state.localidad_name = None
    state.locations_presented = False
    state.available_slots = []
    state.offered_slot = None
    state.proactive_offer_done = False
    state.slot_datetime = None
    state.date_start = None
    state.date_end = None
    log.debug(
        "doctor_data_simplified",
        session=state.session_id,
        doctor_id=doctor_id,
        localidad_count=len(state.localidades or []),
        doctor_data_keys=sorted(state.doctor_data.keys()),
    )

    ingest = get_rag_ingest()
    log.debug("rag_ingest_available", session=state.session_id, available=ingest is not None)
    if ingest is not None:
        try:
            ingest.ingest_doctor(doctor_id, state.doctor_data)
            log.debug("rag_ingest_done", session=state.session_id, doctor_id=doctor_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("rag_ingest_failed", error=str(exc)[:300])

    # Leave reply empty so the graph continues directly into the location
    # node in this same turn (the router ends the turn on a non-empty reply).
    state.reply = ""
    state.current_node = Node.LOCATION
    log.info("doctor_selected", session=state.session_id, doctor_id=doctor_id)
    return state
