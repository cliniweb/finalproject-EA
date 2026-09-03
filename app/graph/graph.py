"""LangGraph graph definition for the CliniAI chatbot."""

from __future__ import annotations

import structlog
from langgraph.graph import StateGraph, END

from app.domain.schemas import ChatState, Node
from app.graph.node_intent import intent_node
from app.graph.node_suggest_doctor import suggest_doctor_node
from app.graph.node_doctor_info import doctor_info_node
from app.graph.node_location import location_node
from app.graph.node_datetime import datetime_node
from app.graph.node_fetch_slots import fetch_slots_node
from app.graph.node_collect import collect_node
from app.graph.node_confirm import confirm_node
from app.services.console import say

log = structlog.get_logger()


def _route(state: ChatState) -> str:
    """Edge router — returns the next node name based on current_node."""
    if state.completed:
        log.debug("route_end_completed", session=state.session_id)
        say("🏁 Grafo: conversación COMPLETADA — fin del flujo")
        return END
    # A non-empty reply means the node finished this turn and is waiting for
    # the patient's next message — end the graph invocation here instead of
    # re-entering the same node in an infinite loop.
    if state.reply:
        log.debug(
            "route_end_reply_ready",
            session=state.session_id,
            current_node=str(state.current_node),
            reply_preview=state.reply[:120],
        )
        say(f"⏸️ Grafo: turno terminado en '{state.current_node.value}' — esperando al paciente")
        return END
    log.debug(
        "route_continue",
        session=state.session_id,
        next_node=state.current_node.value,
    )
    say(f"➡️ Grafo: continúa hacia el nodo '{state.current_node.value}' en el mismo turno")
    return state.current_node.value


def build_graph() -> StateGraph:
    g = StateGraph(ChatState)

    # Register nodes
    g.add_node(Node.INTENT.value,         intent_node)
    g.add_node(Node.SUGGEST_DOCTOR.value, suggest_doctor_node)
    g.add_node(Node.DOCTOR_INFO.value,    doctor_info_node)
    g.add_node(Node.LOCATION.value,    location_node)
    g.add_node(Node.DATETIME.value,    datetime_node)
    g.add_node(Node.FETCH_SLOTS.value, fetch_slots_node)
    g.add_node(Node.COLLECT.value,     collect_node)
    g.add_node(Node.CONFIRM.value,     confirm_node)

    # Entry point
    g.set_entry_point(Node.INTENT.value)

    # Conditional edges from every node — all route via _route
    for node in Node:
        if node == Node.DONE:
            continue
        g.add_conditional_edges(node.value, _route)

    return g.compile()


# Module-level singleton
_graph = build_graph()


async def run_turn(state: ChatState) -> ChatState:
    """Run one user turn through the graph and return the updated state."""
    log.debug(
        "turn_start",
        session=state.session_id,
        entry_node=str(state.current_node),
        message_count=len(state.messages),
        doctor_id=state.doctor_id or None,
        completed=state.completed,
    )
    say(
        f"▶️ Grafo: inicia turno (sesión={state.session_id}, nodo={state.current_node.value}, "
        f"mensajes={len(state.messages)}, médico={state.doctor_id or 'ninguno'})"
    )
    result = await _graph.ainvoke(state)
    # LangGraph returns the state as a plain dict — convert back to ChatState.
    if not isinstance(result, ChatState):
        result = ChatState.model_validate(result)
    log.debug(
        "turn_end",
        session=state.session_id,
        final_node=str(result.current_node),
        completed=result.completed,
        reply_preview=result.reply[:120],
    )
    say(
        f"⏹️ Grafo: turno finalizado (nodo final={result.current_node.value}, "
        f"completado={'SÍ' if result.completed else 'NO'})"
    )
    return result
