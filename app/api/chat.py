"""FastAPI router — /chat endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import structlog

from datetime import datetime, timezone

from app.domain.schemas import ChatState, ConsentResult, Node
from app.services import session_store, cliniweb, llm
from app.services.console import say
from app.graph.graph import run_turn

log = structlog.get_logger()
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    # Optional: when omitted, the bot suggests doctors from the patient's symptoms.
    doctor_id: str = ""
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    completed: bool
    booking_url: str | None = None
    # Structured doctor suggestions so the UI can render a clickable list.
    suggestions: list[dict] = []
    # Structured clinic locations so the UI can render a clickable list.
    locations: list[dict] = []
    # Short reply options (e.g. SÍ/NO, dates) rendered as buttons by the UI.
    quick_replies: list[str] = []
    # Rendered by the UI after the suggestions list.
    reply_footnote: str = ""


class ConsentRequest(BaseModel):
    session_id: str
    # Optional: when omitted, the bot suggests doctors from the patient's symptoms.
    doctor_id: str = ""
    # True = the patient pressed "SÍ, acepto"; False = pressed "NO".
    accepted: bool


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    log.debug(
        "chat_request",
        session=req.session_id,
        doctor_id=req.doctor_id or None,
        message=req.message[:200],
    )
    say(f"📩 /chat: mensaje del paciente (sesión={req.session_id}): \"{req.message[:100]}\"")
    say("⏱️  [1/8] Cargando/creando sesión…")
    state = session_store.get_or_create(req.session_id, req.doctor_id)
    say(
        f"⏱️  [1/8] Sesión lista: nodo={state.current_node}, "
        f"mensajes={len(state.messages)}, completado={state.completed}, "
        f"consent_given={state.consent_given}, doctor_data={'SÍ' if state.doctor_data else 'NO'}"
    )
    log.debug(
        "session_loaded",
        session=req.session_id,
        current_node=str(state.current_node),
        message_count=len(state.messages),
        completed=state.completed,
        has_doctor_data=state.doctor_data is not None,
    )

    # Load doctor data on first turn AFTER consent (only when a doctor is already known).
    # No personal/health data processing (profile fetch, RAG indexing) may happen
    # before the patient grants consent.
    if state.consent_given and state.doctor_id and state.doctor_data is None:
        log.debug("initial_doctor_fetch", session=req.session_id, doctor_id=state.doctor_id)
        say(f"🩺 Primer turno con médico conocido — descargando perfil de '{state.doctor_id}' e indexando en RAG…")
        try:
            say(f"⏱️  [2/8] Llamando cliniweb.fetch_doctor('{state.doctor_id}')…")
            raw = await cliniweb.fetch_doctor(state.doctor_id)
            say("⏱️  [2/8] fetch_doctor OK — simplificando datos…")
            state.doctor_data = cliniweb.simplify_doctor_data(raw)
            # Los IDs de empresa/responsable son por localidad — se resuelven
            # después de que el paciente elige la localidad (ver extract_localidades).
            state.localidades = cliniweb.extract_localidades(raw)
            say(f"⏱️  [2/8] Perfil listo: {len(state.localidades)} localidad(es)")
        except Exception as exc:
            log.error("doctor_fetch_failed", error=str(exc)[:300])
            say(f"❌ [2/8] fetch_doctor FALLÓ: {str(exc)[:200]}")
            raise HTTPException(status_code=502, detail="No se pudo obtener el perfil del médico.")

        # RAG: index the doctor's knowledge base (idempotent, lazy, once per doctor)
        from app.dependencies import get_rag_ingest

        say("⏱️  [3/8] Obteniendo servicio de ingesta RAG…")
        ingest = get_rag_ingest()
        say(f"⏱️  [3/8] Ingesta RAG: {'disponible' if ingest is not None else 'NO disponible (omitida)'}")
        if ingest is not None:
            try:
                say(f"⏱️  [3/8] Indexando doctor '{state.doctor_id}' en RAG…")
                ingest.ingest_doctor(state.doctor_id, state.doctor_data)
                say("⏱️  [3/8] Ingesta RAG OK")
            except Exception as exc:  # noqa: BLE001
                log.warning("rag_ingest_failed", error=str(exc)[:300])
                say(f"⚠️ [3/8] Ingesta RAG falló (continuando): {str(exc)[:200]}")

    if state.completed:
        log.debug("chat_already_completed", session=req.session_id)
        return ChatResponse(
            session_id=req.session_id,
            reply="El proceso ya fue completado. Gracias.",
            completed=True,
            booking_url=state.booking_url,
        )

    # Append user message to history
    state.messages.append({"role": "user", "content": req.message})

    # Explicit consent gate — must be resolved before any other processing.
    # On the very first message we ASK for consent (never treat that message
    # as the answer); only subsequent messages are classified as the answer.
    if not state.consent_given:
        if not state.consent_requested:
            state.consent_requested = True
            state.reply = (
                "¡Hola! 👋 Antes de continuar necesito su consentimiento explícito "
                "para procesar sus datos personales y de salud con el fin de "
                "gestionar su cita médica.\n\n"
                "Use los botones \"SÍ, acepto\" / \"NO\" para responder."
            )
            state.messages.append({"role": "assistant", "content": state.reply})
            session_store.save(state)
            say(f"🛡️ Consentimiento SOLICITADO (sesión={req.session_id})")
            return ChatResponse(
                session_id=req.session_id,
                reply=state.reply,
                completed=False,
                booking_url=None,
            )
        say("⏱️  [4/8] Consentimiento pendiente — clasificando respuesta con el LLM…")
        say("⏱️  [4/8] >>> ANTES de llm.extract(ConsentResult) — si esto es lo último que ve, el LLM está colgado")
        # llm.extract is blocking — run it off the event loop so the API stays responsive.
        result = await asyncio.to_thread(
            llm.extract,
            ConsentResult,
            system=(
                "You classify whether a patient explicitly accepts consent to process "
                "their personal and health data for booking a medical appointment. "
                "Only clear, affirmative acceptance counts."
            ),
            user=req.message,
        )
        say(f"⏱️  [4/8] <<< DESPUÉS de llm.extract — accepted={result.accepted}")
        if not result.accepted:
            state.reply = (
                "Entiendo. Sin su consentimiento explícito no puedo continuar con la "
                "gestión de la cita. Si cambia de opinión, escriba \"sí, acepto\"."
            )
            state.messages.append({"role": "assistant", "content": state.reply})
            session_store.save(state)
            say(f"🛡️ Consentimiento RECHAZADO (sesión={req.session_id})")
            return ChatResponse(
                session_id=req.session_id,
                reply=state.reply,
                completed=False,
                booking_url=None,
            )
        state.consent_given = True
        state.consent_timestamp = datetime.now(timezone.utc).isoformat()
        log.info(
            "consent_recorded",
            session=req.session_id,
            timestamp=state.consent_timestamp,
        )
        say(f"🛡️ Consentimiento OTORGADO (sesión={req.session_id}, ts={state.consent_timestamp})")
        state.reply = (
            "¡Gracias! Consentimiento registrado. 🙌\n\n"
            "Cuénteme qué síntomas tiene o qué especialista busca, "
            "y le sugeriré médicos disponibles para agendar una cita."
        )
        state.messages.append({"role": "assistant", "content": state.reply})
        say("⏱️  [5/8] Guardando sesión tras consentimiento…")
        session_store.save(state)
        say("⏱️  [5/8] Sesión guardada — devolviendo respuesta de consentimiento")
        return ChatResponse(
            session_id=req.session_id,
            reply=state.reply,
            completed=False,
            booking_url=None,
        )

    # Run one turn through the LangGraph
    state.reply_footnote = ""  # per-turn; only set again if a node needs it
    state.location_options = []  # per-turn; only set again if a node needs it
    state.quick_replies = []  # per-turn; only set again if a node needs it
    say(f"⏱️  [6/8] >>> Entrando a run_turn (nodo actual={state.current_node})…")
    state = await run_turn(state)
    say(f"⏱️  [6/8] <<< run_turn terminó (nodo final={state.current_node}, completado={state.completed})")

    # Append assistant reply to history
    if state.reply:
        state.messages.append({"role": "assistant", "content": state.reply})

    say("⏱️  [7/8] Guardando sesión final…")
    session_store.save(state)
    say("⏱️  [7/8] Sesión guardada")
    log.debug(
        "chat_response",
        session=req.session_id,
        final_node=str(state.current_node),
        completed=state.completed,
        reply_preview=state.reply[:150],
        booking_url=state.booking_url,
    )
    say(f"📤 /chat: respuesta enviada al paciente: \"{state.reply[:100]}\"")
    if state.booking_url:
        say(f"🔗 /chat: URL de reserva generada → {state.booking_url}")
    say("⏱️  [8/8] Fin del turno — respuesta HTTP saliendo")

    return ChatResponse(
        session_id=req.session_id,
        reply=state.reply,
        completed=state.completed,
        booking_url=state.booking_url,
        # Only render the clickable doctor list when this turn actually ended
        # waiting for a pick — otherwise stale suggestions re-lock the UI.
        suggestions=(
            state.suggested_doctors
            if state.current_node == Node.SUGGEST_DOCTOR
            else []
        ),
        locations=state.location_options or [],
        quick_replies=state.quick_replies or [],
        reply_footnote=state.reply_footnote,
    )


@router.post("/consent", response_model=ChatResponse)
async def consent(req: ConsentRequest) -> ChatResponse:
    """Deterministic consent gate driven by the UI's SÍ/NO buttons.

    A button click is an unambiguous answer — no LLM classification needed.
    """
    state = session_store.get_or_create(req.session_id, req.doctor_id)
    # Terminal state: a completed booking can never be reopened or mutated.
    if state.completed:
        log.debug("consent_on_completed_session", session=req.session_id)
        return ChatResponse(
            session_id=req.session_id,
            reply="El proceso ya fue completado. Gracias.",
            completed=True,
            booking_url=state.booking_url,
        )
    state.consent_requested = True
    if req.accepted:
        state.consent_given = True
        state.consent_timestamp = datetime.now(timezone.utc).isoformat()
        log.info("consent_recorded", session=req.session_id, timestamp=state.consent_timestamp)
        say(f"🛡️ Consentimiento OTORGADO vía botón (sesión={req.session_id}, ts={state.consent_timestamp})")
        state.reply = (
            "¡Gracias! Consentimiento registrado. 🙌\n\n"
            "Cuénteme qué síntomas tiene o qué especialista busca, "
            "y le sugeríre médicos disponibles para agendar una cita."
        )
    else:
        log.info("consent_declined", session=req.session_id)
        say(f"🛡️ Consentimiento RECHAZADO vía botón (sesión={req.session_id})")
        state.reply = (
            "Entiendo. Sin su consentimiento explícito no puedo continuar con la "
            "gestión de la cita. Si cambia de opinión, pulse \"SÍ, acepto\"."
        )
    state.messages.append({"role": "assistant", "content": state.reply})
    session_store.save(state)
    return ChatResponse(
        session_id=req.session_id,
        reply=state.reply,
        completed=False,
        booking_url=None,
    )


@router.delete("/{session_id}")
async def reset_session(session_id: str) -> dict:
    session_store.delete(session_id)
    return {"deleted": session_id}
