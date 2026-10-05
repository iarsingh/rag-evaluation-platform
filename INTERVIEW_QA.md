# rag-evaluation-platform — interview questions and answers

[README](README.md) · [Project architecture](PROJECT_ARCHITECTURE.md)

Answers below use this repository’s files and implementation. They distinguish existing behavior from suggested extensions; source links let you verify each walkthrough.

## 1. What problem does rag-evaluation-platform address, and what can you demonstrate?

Scores a RAG answer against its retrieved contexts and a gold answer, using word overlap after removing stop words and a trailing plural `s`. It does not call a model, so a score never changes between runs.

I would demonstrate the linked implementation or examples and distinguish that evidence from any planned production features. Start with [`README.md`](README.md).

## 2. How is this repository organized?

- [`src/rageval/main.py`](src/rageval/main.py): Implementation or supporting configuration.
- [`src/rageval/score.py`](src/rageval/score.py): Implementation or supporting configuration.
- [`requirements.txt`](requirements.txt): Implementation or supporting configuration.
- [`src/rageval/__init__.py`](src/rageval/__init__.py): Implementation or supporting configuration.
- [`tests/test_eval.py`](tests/test_eval.py): Executable checks and regression examples.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml): GitHub Actions job definitions.
- [`README.md`](README.md): Project explanations or operating notes.

[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) contains the component diagram and the implementation walkthrough.

## 3. Can you walk through `evaluate` and explain the decision it makes?

The main walkthrough here is `evaluate(answer, gold, context, require_citations=False, thresholds=None)` in [`src/rageval/score.py`](src/rageval/score.py#L50).

```python
def evaluate(answer, gold, context, require_citations=False, thresholds=None):
    for name, value in (("answer", answer), ("gold", gold)):
        if not isinstance(value, str) or not value.strip():
            raise EvalError(f"{name} must be a non-empty string")
    contexts = [context] if isinstance(context, str) else context
    if not isinstance(contexts, list) or not contexts or not all(isinstance(c, str) for c in contexts):
        raise EvalError("context must be a string or a non-empty list of strings")
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    for name, value in limits.items():
        if name not in METRICS or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise EvalError(f"threshold {name} must be a known metric with a value from 0 to 1")

    answer_words = words(answer)
    gold_words = words(gold)
    context_words = set().union(*(words(c) for c in contexts))
    scores = {
        "faithfulness": ratio(context_words, answer_words),
        "correctness": ratio(answer_words, gold_words),
        "context_precision": context_precision(gold_words, contexts),
        "context_recall": ratio(context_words, gold_words),
```

This is an excerpt; follow the source link for the rest of the branches.

The implementation calls `EvalError`, `all`, `check_citations`, `context_precision`, `failed.append`, `isinstance`, `limits.items`, `ratio`, `round`. In an interview, trace those calls in execution order using a fixture input.

## 4. What responsibility does `evaluate_batch` have?

`evaluate_batch(cases, thresholds=None)` is defined in [`src/rageval/score.py`](src/rageval/score.py#L84).

Its return expressions include:

- `{'results': results, 'summary': summary, 'failing': [r['id'] for r in results if not r['passed']]}`

It uses `EvalError`, `all`, `case.get`, `evaluate`, `isinstance`, `len`, `results.append`, `round`. This is the code path I would compare against the caller to explain responsibility boundaries.

## 5. What input validation and failure behavior are implemented?

Explicit failure paths include:

- `HTTPException(status_code=422, detail=str(exc))` in [`src/rageval/main.py`](src/rageval/main.py#L12).
- `EvalError('context must be a string or a non-empty list of strings')` in [`src/rageval/score.py`](src/rageval/score.py#L56).
- `EvalError('cases must be a list of 1 to 500 items')` in [`src/rageval/score.py`](src/rageval/score.py#L86).
- `EvalError('every case needs a string id')` in [`src/rageval/score.py`](src/rageval/score.py#L89).
- `EvalError('case ids must be unique')` in [`src/rageval/score.py`](src/rageval/score.py#L91).
- `EvalError('tolerance must be from 0 to 1')` in [`src/rageval/score.py`](src/rageval/score.py#L103).
- `EvalError('baseline must map metric names to values')` in [`src/rageval/score.py`](src/rageval/score.py#L105).

I would test both the condition that reaches each exception and the caller that translates it. An explicit raise does not mean every malformed input or dependency failure is handled.

## 6. Which test would you use to demonstrate correctness?

[`tests/test_eval.py`](tests/test_eval.py#L16) contains `test_pass_and_fail`:

```python
def test_pass_and_fail():
    assert evaluate(answer=GOLD).json()["passed"] is True
    assert evaluate(answer="The cafeteria serves soup.").json()["passed"] is False
```

This is a concrete regression example from the repository. Its assertions establish that case; they do not establish behavior for every input or under production load.

## 7. What HTTP interface does the code expose?

- `GET /healthz` → `healthz` in [`src/rageval/main.py`](src/rageval/main.py#L16).
- `GET /metrics` → `metrics` in [`src/rageval/main.py`](src/rageval/main.py#L21).
- `POST /evaluate` → `post_evaluate` in [`src/rageval/main.py`](src/rageval/main.py#L26).
- `POST /evaluate/batch` → `post_batch` in [`src/rageval/main.py`](src/rageval/main.py#L38).
- `POST /gate` → `post_gate` in [`src/rageval/main.py`](src/rageval/main.py#L43).

These are literal decorators. Application/router prefixes, authentication, and middleware must be checked in the corresponding setup code.

## 8. Where does state live, and what happens with multiple workers?

Module-level containers include `STOP`, `DEFAULT_THRESHOLDS` in [`src/rageval/score.py`](src/rageval/score.py).

These containers belong to a Python process. Inspect which are constant fixtures and which are mutated. Mutable process state needs an explicit shared-storage or synchronization strategy before multiple workers can provide consistent behavior.

## 9. How would another engineer reproduce your walkthrough?

Start from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

These commands follow repository manifests; environment setup and command results still need to be checked on the target machine.

## 10. What does automation verify, and what does it not prove?

Inspect [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for triggers, permissions, and job commands. I would name the checks that those definitions run and show the latest run separately. A workflow definition alone does not establish a successful deployment, security review, or production SLO.

## 11. How would you present this project in a Forward Deployed Engineer interview?

Start with the user and operational problem described in [`README.md`](README.md). Explain one constraint that changes the implementation, show the linked code or example, and walk through a success case and a failure case. Agree on a measurable acceptance criterion before expanding the solution, and leave a handoff with data boundaries and rollback ownership. Any proposed production or business metric should be identified as a target until measured.

## 12. What is the input-to-output contract of `evaluate`?

In [`src/rageval/score.py`](src/rageval/score.py#L50), `evaluate(answer, gold, context, require_citations=False, thresholds=None)` receives the inputs. The function computes these intermediate values:

- `contexts = [context] if isinstance(context, str) else context`
- `limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}`
- `answer_words = words(answer)`
- `gold_words = words(gold)`
- `context_words = set().union(*(words(c) for c in contexts))`
- `scores = {'faithfulness': ratio(context_words, answer_words), 'correctness': ratio(answer_words, gold_words), 'context_precision': context_precision(gold_words, contexts), 'context_recall': ratio(context_words, gold_words)}`
- `citations_valid, citation_problems = check_citations(answer, contexts, require_citations)`

Its result is defined by:

- `{**{name: round(value, 4) for name, value in scores.items()}, 'citations_valid': citations_valid, 'citation_problems': citation_problems, 'failed': failed, 'passed': not failed}`

## 13. Which decision rules or boundary conditions should an interviewer challenge?

The implementation in [`src/rageval/score.py`](src/rageval/score.py#L50) branches on:

- `not isinstance(contexts, list) or not contexts or (not all((isinstance(c, str) for c in contexts)))`
- `citations_valid is False`
- `not isinstance(value, str) or not value.strip()`
- `name not in METRICS or not isinstance(value, (int, float)) or (not 0 <= value <= 1)`

A useful extension is a table-driven test that covers each condition just below, at, and above its boundary where applicable. These expressions are the current rules; changing them changes behavior and should be justified by the project’s acceptance criteria.
