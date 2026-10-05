# RAG Evaluation Platform

<!-- project-guide:start -->
## Project guide

[Project architecture](PROJECT_ARCHITECTURE.md) · [Interview questions and answers](INTERVIEW_QA.md)

Use the architecture document for the component diagram, implementation boundaries, and verification entry points. The interview guide includes source-backed answers and project walkthroughs.

### Implementation map

| Component | Responsibility |
| --- | --- |
| [`src/rageval/main.py`](src/rageval/main.py) | HTTP handlers: `GET /healthz`, `GET /metrics`, `POST /evaluate`, `POST /evaluate/batch`, `POST /gate` |
| [`src/rageval/score.py`](src/rageval/score.py) | Functions: `stem`, `words`, `ratio`, `context_precision`, `check_citations`, `evaluate`, `evaluate_batch` |
| [`requirements.txt`](requirements.txt) | Implementation or supporting configuration |
| [`src/rageval/__init__.py`](src/rageval/__init__.py) | Implementation or supporting configuration |
| [`tests/test_eval.py`](tests/test_eval.py) | Executable checks and regression examples |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | GitHub Actions job definitions |
| [`README.md`](README.md) | Project explanations or operating notes |

### Local setup and verification

From the repository root (the commands follow the checked-in manifests):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

To serve the FastAPI application locally, install the server separately if it is not already available:

```bash
python -m pip install uvicorn
PYTHONPATH=src python -m uvicorn rageval.main:app --reload
```

<!-- project-guide:end -->

Level: 7 — Intermediate RAG

Skills: Python, RAG metrics, citation checks, batch evaluation, a regression gate for CI

Scores a RAG answer against its retrieved contexts and a gold answer, using word overlap after removing stop words and a trailing plural `s`. It does not call a model, so a score never changes between runs.

| Metric | Meaning |
| --- | --- |
| `faithfulness` | Share of answer words found in the contexts. Low means the answer added unsupported detail. |
| `correctness` | Share of gold words found in the answer. |
| `context_precision` | Average precision of the context ranking. A context is relevant when it holds at least half the gold words. |
| `context_recall` | Share of gold words found anywhere in the contexts. |

Citations are markers like `[1]` that point at contexts by position. A marker is invalid if it points past the last context or if the cited context shares fewer than two words with the answer. With `require_citations`, an answer with no markers fails.

A case passes when faithfulness and correctness are at least 0.5 (override with `thresholds`) and no citation is invalid.

```bash
pip install -r requirements.txt
pytest -q
PYTHONPATH=src uvicorn rageval.main:app --reload
```

| Method and path | Use |
| --- | --- |
| `POST /evaluate` | One case: `answer`, `gold`, and `context` (a string) or `contexts` (a ranked list) |
| `POST /evaluate/batch` | Up to 500 cases with unique ids. Returns each result, mean metrics, `pass_rate`, and failing ids |
| `POST /gate` | Batch-evaluates `cases` and fails if any metric in `baseline` drops by more than `tolerance` (default 0.05) |

## Using the gate in CI

Store the summary from a known-good run as the baseline. On each prompt or retriever change, post the same cases to `/gate`. A `passed: false` response lists each regressed metric with its baseline, candidate value, and drop, which is enough to block the merge and say why.

```bash
curl -s -X POST localhost:8000/gate -H 'content-type: application/json' -d '{
  "cases": [{"id": "q1", "answer": "The platform refuses production.", "gold": "The platform refuses production.", "contexts": ["The platform refuses production deploys."]}],
  "baseline": {"correctness": 0.9, "pass_rate": 0.8},
  "tolerance": 0.05
}'
```

## Ops plane

Workspaces, tenant isolation, job approval, and audit live under `/v1`. Production apply is refused. See `docs/ARCHITECTURE.md`.
