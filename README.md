# RAG Evaluation Platform

Level: 7 — Intermediate RAG

Skills: Python, faithfulness, correctness

Score an answer against the context and a gold sentence. Both overlaps must be at least half of the relevant tokens or the case fails. The scorer does not call a model.

```bash
pip install -r requirements.txt
pytest -q
```
