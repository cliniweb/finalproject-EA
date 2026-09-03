"""Eval 1 — Intent classification accuracy.

Runs the real classifier (Instructor + primary model) over the labelled
dataset and reports accuracy plus a confusion table. The pass threshold is a
deliberate engineering knob: below it the graph misroutes too often to trust.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.domain.schemas import IntentResult
from app.services import llm

PASS_THRESHOLD = 0.85

_SYSTEM = (
    "You are an intent classifier for a medical appointment chatbot. "
    "Respond ONLY in Spanish. "
    "Classify the patient's message into one of: "
    "book_appointment, doctor_info, greeting, other."
)


def load_cases() -> list[dict]:
    path = Path(__file__).parent / "data" / "intent_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def run() -> dict:
    cases = load_cases()
    correct = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failures: list[dict] = []

    for case in cases:
        result: IntentResult = llm.extract(
            IntentResult, system=_SYSTEM, user=case["message"]
        )
        predicted = result.intent
        expected = case["expected"]
        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1
        else:
            failures.append(
                {"message": case["message"], "expected": expected, "predicted": predicted}
            )

    accuracy = correct / len(cases)
    report = {
        "eval": "intent_classification",
        "total": len(cases),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "threshold": PASS_THRESHOLD,
        "passed": accuracy >= PASS_THRESHOLD,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "failures": failures,
    }
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
