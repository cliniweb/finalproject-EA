"""Tests for the eval harness itself — datasets well-formed, scoring logic
correct. These run offline (no API key, no LLM calls)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "evals" / "data"

VALID_INTENTS = {"book_appointment", "doctor_info", "greeting", "other"}


# --- Dataset integrity -----------------------------------------------------------


def test_intent_dataset_is_valid():
    cases = json.loads((DATA_DIR / "intent_cases.json").read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 20
    for case in cases:
        assert case["message"].strip()
        assert case["expected"] in VALID_INTENTS


def test_intent_dataset_covers_every_label():
    cases = json.loads((DATA_DIR / "intent_cases.json").read_text(encoding="utf-8"))["cases"]
    labels = {c["expected"] for c in cases}
    assert labels == VALID_INTENTS


def test_date_dataset_is_valid():
    data = json.loads((DATA_DIR / "date_cases.json").read_text(encoding="utf-8"))
    ref = datetime.strptime(data["reference_date"], "%Y-%m-%d")
    assert ref.strftime("%A") == data["reference_weekday"]
    for case in data["cases"]:
        start = datetime.strptime(case["expected_start"], "%Y-%m-%d")
        end = datetime.strptime(case["expected_end"], "%Y-%m-%d")
        assert start <= end
        assert start >= ref  # appointments are never in the past


def test_grounding_dataset_is_valid():
    data = json.loads((DATA_DIR / "grounding_cases.json").read_text(encoding="utf-8"))
    assert data["doctor_profile"].get("nombrePersona")
    expects = {c["expect"] for c in data["cases"]}
    assert expects == {"grounded", "refusal"}  # both behaviours are tested
    for case in data["cases"]:
        if case["expect"] == "grounded":
            assert case["must_mention"], "grounded cases need at least one expected fact"


# --- Scoring helpers ---------------------------------------------------------------


def test_refusal_detector():
    from evals.eval_grounding import _looks_like_refusal

    assert _looks_like_refusal("Lo siento, no tengo esa informacion.") is True
    assert _looks_like_refusal("No dispongo de ese dato.") is True
    assert _looks_like_refusal("La doctora es pediatra.") is False


def test_runner_knows_all_suites():
    from evals.run_all import SUITES

    assert set(SUITES) == {"intent", "dates", "grounding"}
    for module_name in SUITES.values():
        __import__(module_name, fromlist=["run"])  # importable

    # Every suite module exposes run() and a threshold
    for module_name in SUITES.values():
        module = __import__(module_name, fromlist=["run", "PASS_THRESHOLD"])
        assert callable(module.run)
        assert 0.0 < module.PASS_THRESHOLD <= 1.0
