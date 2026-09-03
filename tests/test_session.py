"""Tests for domain schemas and session store."""

from __future__ import annotations

from app.domain.schemas import ChatState, Node
from app.services import session_store


def test_session_created_on_first_access():
    state = session_store.get_or_create("test-001", "doc-123")
    assert state.session_id == "test-001"
    assert state.doctor_id == "doc-123"
    assert state.current_node == Node.INTENT
    assert state.completed is False


def test_session_reused_on_second_access():
    session_store.get_or_create("test-002", "doc-123")
    state2 = session_store.get_or_create("test-002", "doc-123")
    state2.localidad_id = "999"
    session_store.save(state2)
    assert session_store.get_or_create("test-002", "doc-123").localidad_id == "999"


def test_session_deleted():
    session_store.get_or_create("test-003", "doc-123")
    session_store.delete("test-003")
    fresh = session_store.get_or_create("test-003", "doc-123")
    assert fresh.localidad_id is None


def test_chat_state_message_append():
    state = ChatState(session_id="s1", doctor_id="d1")
    state.messages.append({"role": "user", "content": "Hola"})
    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "user"
