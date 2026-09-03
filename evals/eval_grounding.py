"""Eval 3 — RAG grounding: faithfulness + correct refusal behaviour.

Exercises the REAL pipeline end-to-end (chunk → embed → retrieve → generate →
judge) against a fixture doctor profile:

- **grounded cases** must produce an answer that (a) passes the hallucination
  judge against the retrieved context and (b) mentions the expected facts.
- **refusal cases** ask questions whose answers are NOT in the profile: the
  system must refuse (retrieval floor or gate), never improvise.

The refusal half is the differentiator: it measures the property that matters
most in a medical domain — the system knows what it does not know.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.dependencies import OpenAITextVectorizer
from app.config import get_settings
from app.rag.ingest import RagIngestService
from app.rag.quality import check_grounding
from app.rag.retriever import Retriever
from app.rag.store import InMemoryVectorStore
from app.services import llm

PASS_THRESHOLD = 0.85
_DOCTOR_ID = "eval-doctor"

_SYSTEM_TMPL = (
    "Eres un asistente medico que responde preguntas sobre un medico. "
    "Usa UNICAMENTE la informacion del siguiente CONTEXTO. "
    "Si el contexto no contiene la respuesta, di honestamente que no tienes esa informacion. "
    "Responde siempre en espanol.\n\nCONTEXTO:\n{context}"
)

_REFUSAL_MARKERS = ["no tengo", "no dispongo", "no cuento con", "no aparece", "no tiene esa informac"]


def load_dataset() -> dict:
    path = Path(__file__).parent / "data" / "grounding_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def run() -> dict:
    settings = get_settings()
    dataset = load_dataset()

    # Build an isolated pipeline (own store — never touches the app singletons).
    vectorizer = OpenAITextVectorizer(
        model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY
    )
    store = InMemoryVectorStore()
    ingest = RagIngestService(store=store, vectorizer=vectorizer)
    ingest.ingest_doctor(_DOCTOR_ID, dataset["doctor_profile"])
    retriever = Retriever(
        store=store,
        vectorizer=vectorizer,
        top_k=settings.RAG_TOP_K,
        min_score=settings.RAG_MIN_SCORE,
    )

    correct = 0
    failures: list[dict] = []

    for case in dataset["cases"]:
        question = case["question"]
        retrieved = retriever.retrieve(question, doctor_id=_DOCTOR_ID)

        if not retrieved:
            answer = "REFUSAL(no-context)"
            refused = True
            grounded = True  # a refusal is trivially grounded
        else:
            context = "\n\n".join(f"[{rc.chunk.source}] {rc.chunk.text}" for rc in retrieved)
            answer = llm.chat(
                system=_SYSTEM_TMPL.format(context=context),
                messages=[{"role": "user", "content": question}],
            )
            refused = _looks_like_refusal(answer)
            verdict = check_grounding(
                answer=answer, context=context, model=settings.HALLUCINATION_JUDGE_MODEL
            )
            grounded = verdict.grounded or refused

        if case["expect"] == "grounded":
            mentions_ok = all(m.lower() in answer.lower() for m in case["must_mention"])
            ok = grounded and not refused and mentions_ok
        else:  # expect refusal
            ok = refused

        if ok:
            correct += 1
        else:
            failures.append(
                {
                    "question": question,
                    "expect": case["expect"],
                    "refused": refused,
                    "grounded": grounded,
                    "answer": answer[:300],
                }
            )

    accuracy = correct / len(dataset["cases"])
    return {
        "eval": "rag_grounding",
        "total": len(dataset["cases"]),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "threshold": PASS_THRESHOLD,
        "passed": accuracy >= PASS_THRESHOLD,
        "failures": failures,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
