"""Eval 2 — Natural-language date parsing accuracy.

The datetime node's job is deterministic: turn Spanish date expressions into
yyyy-MM-dd ranges relative to "today". This eval freezes "today" to the
dataset's reference date, so results are reproducible on any day.

Scoring: a case passes only if BOTH start and end match exactly. Date bugs are
off-by-one-week bugs — partial credit hides them.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.schemas import DateRangeResult
from app.services import llm

PASS_THRESHOLD = 0.80

_SYSTEM_TMPL = (
    "Eres un asistente que convierte fechas en lenguaje natural a formato yyyy-MM-dd. "
    "Calcula correctamente 'esta semana', 'la próxima semana', 'mañana', etc. "
    "Hoy es {today} ({weekday}). Muestra el razonamiento paso a paso."
)


def load_dataset() -> dict:
    path = Path(__file__).parent / "data" / "date_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> dict:
    dataset = load_dataset()
    reference_date = dataset["reference_date"]
    reference_weekday = dataset["reference_weekday"]
    cases = dataset["cases"]

    system = _SYSTEM_TMPL.format(today=reference_date, weekday=reference_weekday)

    correct = 0
    failures: list[dict] = []

    for case in cases:
        result: DateRangeResult = llm.extract(
            DateRangeResult, system=system, user=case["message"]
        )
        ok = (
            result.date_start == case["expected_start"]
            and result.date_end == case["expected_end"]
        )
        if ok:
            correct += 1
        else:
            failures.append(
                {
                    "message": case["message"],
                    "expected": f"{case['expected_start']} → {case['expected_end']}",
                    "predicted": f"{result.date_start} → {result.date_end}",
                    "reasoning": result.reasoning[:200],
                }
            )

    accuracy = correct / len(cases)
    return {
        "eval": "date_parsing",
        "reference_date": reference_date,
        "total": len(cases),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "threshold": PASS_THRESHOLD,
        "passed": accuracy >= PASS_THRESHOLD,
        "failures": failures,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
