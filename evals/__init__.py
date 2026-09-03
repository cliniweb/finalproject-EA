"""Evals package — objective, repeatable metrics for the CliniAI system.

Three eval suites, mirroring the layers of the system:

1. **Intent classification** (``eval_intent.py``) — accuracy over a labelled
   Spanish dataset. Measures the classifier that gates the whole graph.
2. **Date parsing** (``eval_dates.py``) — exact-match accuracy of natural
   language → yyyy-MM-dd extraction, relative to a frozen "today".
3. **RAG grounding** (``eval_grounding.py``) — faithfulness of generated
   answers against retrieved context, judged by the hallucination gate.

Run all: ``python -m evals.run_all``
"""
