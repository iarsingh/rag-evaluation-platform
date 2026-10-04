import pytest
from fastapi.testclient import TestClient

from rageval.main import app

client = TestClient(app)
CONTEXT = "The platform refuses production deploys."
GOLD = "The platform refuses production."
OFF_TOPIC = "The cafeteria serves soup on Tuesdays."


def evaluate(**body):
    return client.post("/evaluate", json={"gold": GOLD, "context": CONTEXT, **body})


def test_pass_and_fail():
    assert evaluate(answer=GOLD).json()["passed"] is True
    assert evaluate(answer="The cafeteria serves soup.").json()["passed"] is False


def test_unsupported_detail_fails_faithfulness():
    answer = "The platform refuses production deploys on Fridays because the CEO hates weekends and coffee."
    payload = evaluate(answer=answer).json()
    assert payload["correctness"] == 1.0
    assert payload["faithfulness"] < 0.5
    assert payload["failed"] == ["faithfulness"]


def test_context_precision_rewards_relevant_context_first():
    first = evaluate(answer=GOLD, contexts=[CONTEXT, OFF_TOPIC]).json()
    second = evaluate(answer=GOLD, contexts=[OFF_TOPIC, CONTEXT]).json()
    assert first["context_precision"] == 1.0
    assert second["context_precision"] == 0.5


def test_context_recall_counts_gold_terms_across_contexts():
    payload = evaluate(answer=GOLD, contexts=["The platform is strict.", "It refuses production."]).json()
    assert payload["context_recall"] == 1.0
    partial = evaluate(answer=GOLD, contexts=["The platform is strict."]).json()
    assert partial["context_recall"] == pytest.approx(1 / 3, abs=1e-4)


def test_citations_must_point_at_supporting_context():
    good = evaluate(answer="The platform refuses production deploys [1].", contexts=[CONTEXT, OFF_TOPIC]).json()
    assert good["citations_valid"] is True
    wrong = evaluate(answer="The platform refuses production deploys [2].", contexts=[CONTEXT, OFF_TOPIC]).json()
    assert wrong["citations_valid"] is False
    assert wrong["passed"] is False
    missing = evaluate(answer="The platform refuses production [3].", contexts=[CONTEXT]).json()
    assert missing["citation_problems"] == ["[3] does not match a context"]


def test_required_citations_fail_when_absent():
    payload = evaluate(answer=GOLD, require_citations=True).json()
    assert payload["citations_valid"] is False
    assert "citations" in payload["failed"]


def test_batch_summary_and_failing_ids():
    cases = [
        {"id": "good", "answer": GOLD, "gold": GOLD, "contexts": [CONTEXT]},
        {"id": "bad", "answer": "Soup.", "gold": GOLD, "contexts": [CONTEXT]},
    ]
    payload = client.post("/evaluate/batch", json={"cases": cases}).json()
    assert payload["failing"] == ["bad"]
    assert payload["summary"]["pass_rate"] == 0.5
    assert payload["summary"]["correctness"] == 0.5


def test_gate_blocks_a_regression_and_allows_an_improvement():
    cases = [
        {"id": "good", "answer": GOLD, "gold": GOLD, "contexts": [CONTEXT]},
        {"id": "bad", "answer": "Soup.", "gold": GOLD, "contexts": [CONTEXT]},
    ]
    blocked = client.post("/gate", json={"cases": cases, "baseline": {"correctness": 0.9}}).json()
    assert blocked["passed"] is False
    assert blocked["regressions"][0]["metric"] == "correctness"
    allowed = client.post("/gate", json={"cases": cases[:1], "baseline": {"correctness": 0.9, "pass_rate": 0.8}}).json()
    assert allowed["passed"] is True


def test_invalid_requests_are_refused():
    assert client.post("/evaluate", json={"gold": GOLD, "context": CONTEXT}).status_code == 422
    assert evaluate(answer=GOLD, contexts=[]).status_code == 422
    assert evaluate(answer=GOLD, thresholds={"vibes": 0.5}).status_code == 422
    dupes = [{"id": "x", "answer": GOLD, "gold": GOLD, "contexts": [CONTEXT]}] * 2
    assert client.post("/evaluate/batch", json={"cases": dupes}).status_code == 422
    assert client.post("/gate", json={"cases": dupes[:1], "baseline": {"correctness": 0.9}, "tolerance": 2}).status_code == 422
