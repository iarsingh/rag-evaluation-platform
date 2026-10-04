import re

STOP = {"the", "a", "an", "is", "of", "and", "to", "in", "what", "why", "does"}
METRICS = ("faithfulness", "correctness", "context_precision", "context_recall")
DEFAULT_THRESHOLDS = {"faithfulness": 0.5, "correctness": 0.5}
RELEVANT = 0.5
MARKER = re.compile(r"\[(\d+)\]")


class EvalError(ValueError):
    pass


def stem(token):
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def words(text):
    return {stem(token) for token in re.findall(r"[a-z0-9]+", MARKER.sub(" ", text).lower())} - STOP


def ratio(part, whole):
    return len(part & whole) / len(whole) if whole else 0.0


def context_precision(gold_words, contexts):
    hits = 0
    precisions = []
    for rank, context in enumerate(contexts, start=1):
        if ratio(words(context), gold_words) >= RELEVANT:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / len(precisions) if precisions else 0.0


def check_citations(answer, contexts, required):
    cited = [int(number) for number in MARKER.findall(answer)]
    if not cited:
        return (False if required else None), []
    problems = []
    answer_words = words(answer)
    for number in sorted(set(cited)):
        if not 1 <= number <= len(contexts):
            problems.append(f"[{number}] does not match a context")
        elif len(answer_words & words(contexts[number - 1])) < 2:
            problems.append(f"[{number}] does not support the answer")
    return not problems, problems


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
    }
    citations_valid, citation_problems = check_citations(answer, contexts, require_citations)
    failed = [name for name, limit in limits.items() if scores[name] < limit]
    if citations_valid is False:
        failed.append("citations")
    return {
        **{name: round(value, 4) for name, value in scores.items()},
        "citations_valid": citations_valid,
        "citation_problems": citation_problems,
        "failed": failed,
        "passed": not failed,
    }


def evaluate_batch(cases, thresholds=None):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 500:
        raise EvalError("cases must be a list of 1 to 500 items")
    ids = [case.get("id") if isinstance(case, dict) else None for case in cases]
    if not all(isinstance(case_id, str) and case_id for case_id in ids):
        raise EvalError("every case needs a string id")
    if len(set(ids)) != len(ids):
        raise EvalError("case ids must be unique")
    results = []
    for case in cases:
        result = evaluate(case.get("answer"), case.get("gold"), case.get("contexts", case.get("context")), case.get("require_citations", False), thresholds)
        results.append({"id": case["id"], **result})
    summary = {name: round(sum(r[name] for r in results) / len(results), 4) for name in METRICS}
    summary["pass_rate"] = round(sum(r["passed"] for r in results) / len(results), 4)
    return {"results": results, "summary": summary, "failing": [r["id"] for r in results if not r["passed"]]}


def gate(summary, baseline, tolerance=0.05):
    if not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 1:
        raise EvalError("tolerance must be from 0 to 1")
    if not isinstance(baseline, dict) or not baseline:
        raise EvalError("baseline must map metric names to values")
    regressions = []
    for name, before in baseline.items():
        if name not in summary:
            raise EvalError(f"unknown baseline metric: {name}")
        after = summary[name]
        if after < before - tolerance:
            regressions.append({"metric": name, "baseline": before, "candidate": after, "drop": round(before - after, 4)})
    return {"passed": not regressions, "regressions": regressions, "tolerance": tolerance}
