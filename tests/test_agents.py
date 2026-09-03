"""Tests for the agents layer (privileges, supervisor fallback, HITL flow)."""

from __future__ import annotations

import pytest

from app.agents.privileges import PrivilegeViolation, ToolRegistry
from app.agents.supervisor import _deterministic_next
from app.domain.schemas import ChatState, Node, PatientData


# --- ToolRegistry -----------------------------------------------------------------


def test_allowed_call_passes_and_is_audited():
    reg = ToolRegistry()
    reg.register("build_booking_url", {"confirm"})
    assert reg.check("build_booking_url", "confirm", {"a": 1}) is True
    assert len(reg.audit_log) == 1
    assert reg.audit_log[0].allowed is True
    assert len(reg.audit_log[0].args_digest) == 64  # SHA-256 hex


def test_denied_call_returns_false_in_default_mode():
    reg = ToolRegistry(strict=False)
    reg.register("build_booking_url", {"confirm"})
    assert reg.check("build_booking_url", "intent", {}) is False
    assert len(reg.denials()) == 1


def test_denied_call_raises_in_strict_mode():
    reg = ToolRegistry(strict=True)
    reg.register("build_booking_url", {"confirm"})
    with pytest.raises(PrivilegeViolation):
        reg.check("build_booking_url", "intent", {})


def test_unregistered_tool_is_denied():
    reg = ToolRegistry()
    assert reg.check("unknown_tool", "confirm", {}) is False


def test_args_digest_is_deterministic():
    reg = ToolRegistry()
    reg.register("t", {"n"})
    reg.check("t", "n", {"x": 1, "y": 2})
    reg.check("t", "n", {"y": 2, "x": 1})  # same args, different order
    assert reg.audit_log[0].args_digest == reg.audit_log[1].args_digest


def test_args_preview_is_capped():
    reg = ToolRegistry()
    reg.register("t", {"n"})
    reg.check("t", "n", {"data": "x" * 1000})
    assert len(reg.audit_log[0].args_preview) <= 200


# --- Supervisor deterministic fallback ------------------------------------------------


def _state(**kwargs) -> ChatState:
    return ChatState(session_id="s", doctor_id="d", **kwargs)


def test_fallback_routes_to_location_first():
    assert _deterministic_next(_state()) == Node.LOCATION


def test_fallback_routes_to_datetime_after_location():
    assert _deterministic_next(_state(localidad_id="1")) == Node.DATETIME


def test_fallback_routes_to_collect_after_slot():
    state = _state(localidad_id="1", slot_datetime="2025-01-01 10:00")
    assert _deterministic_next(state) == Node.COLLECT


def test_fallback_routes_to_confirm_when_complete():
    state = _state(
        localidad_id="1",
        slot_datetime="2025-01-01 10:00",
        patient=PatientData(full_name="Ana", symptoms="dolor", email="a@b.com"),
    )
    assert _deterministic_next(state) == Node.CONFIRM


# --- HITL confirm node flow -----------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_pauses_first_then_books_on_yes():
    from app.graph.node_confirm import confirm_node

    state = _state(
        localidad_id="1",
        localidad_name="Centro",
        slot_datetime="2025-01-01 10:00",
        patient=PatientData(full_name="Ana", symptoms="dolor", email="a@b.com"),
    )
    state.messages.append({"role": "user", "content": "quiero confirmar"})

    # First entry: HITL pause
    state = await confirm_node(state)
    assert state.awaiting_human is True
    assert state.completed is False
    assert "Confirma" in state.reply or "confirme" in state.reply.lower()

    # Patient says yes → booking proceeds
    state.messages.append({"role": "user", "content": "Sí"})
    state = await confirm_node(state)
    assert state.completed is True
    assert state.booking_url is not None
    assert state.awaiting_human is False


@pytest.mark.asyncio
async def test_confirm_rejection_returns_to_collect():
    from app.graph.node_confirm import confirm_node

    state = _state(
        localidad_id="1",
        localidad_name="Centro",
        slot_datetime="2025-01-01 10:00",
        patient=PatientData(full_name="Ana", symptoms="dolor", email="a@b.com"),
        awaiting_human=True,
    )
    state.messages.append({"role": "user", "content": "no, quiero cambiar el email"})

    state = await confirm_node(state)
    assert state.completed is False
    assert state.current_node == Node.COLLECT
    assert state.awaiting_human is False


@pytest.mark.asyncio
async def test_confirm_ambiguous_stays_paused():
    from app.graph.node_confirm import confirm_node

    state = _state(
        localidad_id="1",
        localidad_name="Centro",
        slot_datetime="2025-01-01 10:00",
        patient=PatientData(full_name="Ana", symptoms="dolor", email="a@b.com"),
        awaiting_human=True,
    )
    state.messages.append({"role": "user", "content": "mmm tal vez"})

    state = await confirm_node(state)
    assert state.completed is False
    assert state.current_node == Node.CONFIRM
