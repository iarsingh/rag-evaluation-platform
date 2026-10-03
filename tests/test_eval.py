from fastapi.testclient import TestClient
from rageval.main import app

def test_pass_and_fail():
    client = TestClient(app)
    context = "The platform refuses production deploys."
    gold = "The platform refuses production."
    good = client.post("/evaluate", json={"answer": gold, "gold": gold, "context": context}).json()
    assert good["passed"] is True
    bad = client.post("/evaluate", json={"answer": "The cafeteria serves soup.", "gold": gold, "context": context}).json()
    assert bad["passed"] is False
