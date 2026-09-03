"""Tests for Cliniweb helper utilities (no network calls)."""

from __future__ import annotations

from app.services.cliniweb import (
    build_booking_url,
    extract_localidades,
    simplify_doctor_data,
    simplify_search_results,
)


def test_build_booking_url_contains_params():
    url = build_booking_url(
        patient_name="Juan Pérez",
        symptoms="dolor de cabeza",
        email="juan@example.com",
        slot_datetime="2024-11-15 10:00",
        localidad_id="12345",
    )
    assert "u=Juan" in url
    assert "l=12345" in url
    assert "t=2024" in url


def test_extract_localidades():
    doctor = {
        "localidades": [
            {"idEmpresa": 39965, "localidad": {"id": 100, "nombre": "Centro", "idPersona": 115986}},
            {"idEmpresa": 39965, "localidad": {"id": 200, "nombre": "Norte", "idPersona": 115986}},
            {"localidad": {}},  # incomplete — should be skipped
        ]
    }
    locs = extract_localidades(doctor)
    assert len(locs) == 2
    assert locs[0] == {
        "id": "100",
        "nombre": "Centro",
        "idEmpresa": "39965",
        "idPersona": "115986",
    }


def test_simplify_removes_noise():
    data = {
        "idDoctor": 999,
        "nombrePersona": "Ana García",
        "foto": "pic.jpg",
        "especialidades": ["Pediatría"],
        "empty_field": None,
    }
    clean = simplify_doctor_data(data)
    assert "nombrePersona" in clean
    assert "foto" not in clean
    assert "empty_field" not in clean


def test_simplify_search_results():
    medicos = [
        {
            "idPersona": 115986,
            "nombre": "Luisa F. Cuddy",
            "descripcion": "Ginecología",
            "url": "/doctoracuddy",
            "sexo": "F",
        },
        {"idPersona": 2, "nombre": "Sin Perfil", "descripcion": "Pediatría", "url": None},  # skipped
        {"idPersona": 3, "url": "/sinnombre"},  # skipped
    ]
    results = simplify_search_results(medicos)
    assert len(results) == 1
    assert results[0]["doctor_id"] == "doctoracuddy"
    assert results[0]["nombre"] == "Luisa F. Cuddy"
    assert results[0]["especialidad"] == "Ginecología"


def test_simplify_search_results_empty():
    assert simplify_search_results([]) == []
