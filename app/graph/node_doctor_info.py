"""Node: answer doctor info questions - RAG-grounded with CAG in front.

Full pipeline (CAG -> RAG -> gate):

1. Exact-match cache      - literal repeat of the question: free.
2. Semantic cache         - semantically same question: one embedding call.
3. RAG retrieval          - top-k chunks from the doctor's indexed knowledge,
                            with a similarity floor. Zero kept chunks = the
                            answer is NOT in the knowledge base -> refuse
                            instead of hallucinate.
4. Grounded generation    - answer ONLY from the retrieved context.
5. Hallucination gate     - LLM-judge verifies grounding; on failure the node
                            serves a refusal (medical domain: no answer beats
                            a wrong answer).
6. Store in both caches   - only gated, grounded answers are cached.
"""

from __future__ import annotations

import structlog

from app.config import get_settings
from app.dependencies import (
    get_exact_cache,
    get_retriever,
    get_semantic_cache,
)
from app.domain.schemas import ChatState, Node
from app.services import llm
from app.services.console import say

log = structlog.get_logger()

_SYSTEM_TMPL = (
    "Eres un asistente medico que responde preguntas sobre un medico. "
    "Usa UNICAMENTE la informacion del siguiente CONTEXTO. "
    "Si el contexto no contiene la respuesta, di honestamente que no tienes esa informacion. "
    "Responde siempre en espanol. No menciones el contexto ni fuentes internas. "
    "Si el paciente quiere hacer una cita o preguntar por horarios disponibles, "
    "indicale que lo puede hacer a continuacion.\n\n"
    "CONTEXTO:\n{context}"
)

_REFUSAL = (
    "Lo siento, no tengo esa informacion sobre el medico. "
    "Puedo ayudarle con otra consulta o desea agendar una cita?"
)


async def doctor_info_node(state: ChatState) -> ChatState:
    settings = get_settings()
    model = settings.LLM_PRIMARY_MODEL

    last_user = next(
        (m["content"] for m in reversed(state.messages) if m["role"] == "user"), ""
    )

    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()
    retriever = get_retriever()

    say(f"🩺 Nodo doctor_info: pregunta sobre '{state.doctor_id}': \"{last_user[:80]}\"")

    # --- Tier 1: exact-match cache ---
    # Key on doctor_id + question + model (context varies per retrieval, so it
    # cannot be part of the key; the doctor_id partition gives correctness).
    exact_key = exact_cache.make_key(
        system=f"doctor_info:{state.doctor_id}", user_message=last_user, model=model
    )
    cached = exact_cache.get(exact_key)
    if cached is not None:
        state.reply = cached
        state.current_node = Node.INTENT
        log.info("doctor_info_answered", session=state.session_id, source="exact_cache")
        say("✅ doctor_info respondido desde CACHÉ EXACTA — gratis, 0 llamadas al LLM")
        return state

    # --- Tier 2: semantic cache ---
    bucket = None
    if semantic_cache is not None:
        bucket = semantic_cache.bucket_for(state.doctor_id, Node.DOCTOR_INFO.value, model)
        cached = semantic_cache.lookup(last_user, bucket)
        if cached is not None:
            exact_cache.set(exact_key, cached)
            state.reply = cached
            state.current_node = Node.INTENT
            log.info("doctor_info_answered", session=state.session_id, source="semantic_cache")
            say("✅ doctor_info respondido desde CACHÉ SEMÁNTICA — solo costó 1 embedding")
            return state

    # --- Tier 3: RAG retrieval ---
    if retriever is None:
        state.reply = _REFUSAL
        state.current_node = Node.INTENT
        log.warning("doctor_info_no_retriever", session=state.session_id)
        return state

    retrieved = retriever.retrieve(last_user, doctor_id=state.doctor_id)
    if not retrieved:
        # Below-floor similarity across the board: the knowledge base does not
        # contain the answer. Refuse - do not let the LLM improvise.
        state.reply = _REFUSAL
        state.current_node = Node.INTENT
        log.info("doctor_info_answered", session=state.session_id, source="refusal_no_context")
        say("🛑 doctor_info: sin contexto RAG — se RECHAZA responder para no alucinar")
        return state

    context = "\n\n".join(
        f"[{rc.chunk.source}] {rc.chunk.text}" for rc in retrieved
    )
    system = _SYSTEM_TMPL.format(context=context)

    # --- Tier 4: grounded generation ---
    say(f"🤖 doctor_info: generando respuesta con el LLM ({model}) sobre {len(retrieved)} fragmentos…")
    # Baja temperatura: la respuesta debe ceñirse al contexto (el juez rechaza
    # afirmaciones sin respaldo) — precisión antes que calidez aquí.
    answer = llm.chat(system=system, messages=state.messages, temperature=0.2)

    # --- Tier 5: hallucination gate ---
    if settings.HALLUCINATION_GATE_ENABLED:
        from app.rag.quality import check_grounding

        verdict = check_grounding(
            answer=answer, context=context, model=settings.HALLUCINATION_JUDGE_MODEL
        )
        if not verdict.grounded:
            log.warning(
                "doctor_info_gate_blocked",
                session=state.session_id,
                unsupported=verdict.unsupported_claims[:3],
            )
            state.reply = _REFUSAL
            state.current_node = Node.INTENT
            return state

    # --- Tier 6: cache the gated answer ---
    state.reply = answer
    state.current_node = Node.INTENT
    exact_cache.set(exact_key, answer)
    if semantic_cache is not None and bucket is not None:
        semantic_cache.store(last_user, answer, bucket)

    log.info(
        "doctor_info_answered",
        session=state.session_id,
        source="rag_llm",
        chunks_used=len(retrieved),
    )
    say(f"✅ doctor_info respondido vía RAG+LLM ({len(retrieved)} fragmentos) y guardado en ambas cachés")
    return state
