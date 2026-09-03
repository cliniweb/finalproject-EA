"""Run all evals and write a combined JSON report.

Usage:
    .venv\\Scripts\\python -m evals.run_all
    .venv\\Scripts\\python -m evals.run_all --only intent
    .venv\\Scripts\\python -m evals.run_all --only dates,grounding

Exit code 0 = every suite passed its threshold; 1 otherwise (CI-friendly).
Reports are written to evals/reports/report_<timestamp>.json so successive
runs are comparable — evidence of iteration, not a one-off score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SUITES = {
    "intent": "evals.eval_intent",
    "dates": "evals.eval_dates",
    "grounding": "evals.eval_grounding",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="Comma-separated subset: intent,dates,grounding")
    args = parser.parse_args()

    selected = [s.strip() for s in args.only.split(",") if s.strip()] or list(SUITES)

    results = []
    all_passed = True

    for name in selected:
        module_name = SUITES.get(name)
        if module_name is None:
            print(f"Unknown suite: {name}. Options: {', '.join(SUITES)}")
            return 2
        print(f"\n=== Running eval: {name} ===")
        module = __import__(module_name, fromlist=["run"])
        report = module.run()
        results.append(report)
        status = "PASS" if report["passed"] else "FAIL"
        print(f"  {status}  accuracy={report['accuracy']} (threshold {report['threshold']})")
        for failure in report.get("failures", [])[:5]:
            print(f"    failure: {json.dumps(failure, ensure_ascii=False)[:180]}")
        all_passed = all_passed and report["passed"]

    # Persist the combined report (timestamped) for run-over-run comparison.
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = reports_dir / f"report_{stamp}.json"
    out_path.write_text(
        json.dumps({"timestamp": stamp, "suites": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport written: {out_path}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
