from rageval.ops import router as ops_router
from fastapi import FastAPI, HTTPException

from rageval.score import METRICS, EvalError, evaluate, evaluate_batch, gate

app = FastAPI()
app.include_router(ops_router, prefix="/v1")


def guarded(call, *args):
    try:
        return call(*args)
    except EvalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return {"metrics": list(METRICS)}


@app.post("/evaluate")
def post_evaluate(body: dict):
    return guarded(
        evaluate,
        body.get("answer"),
        body.get("gold"),
        body.get("contexts", body.get("context")),
        body.get("require_citations", False),
        body.get("thresholds"),
    )


@app.post("/evaluate/batch")
def post_batch(body: dict):
    return guarded(evaluate_batch, body.get("cases"), body.get("thresholds"))


@app.post("/gate")
def post_gate(body: dict):
    report = guarded(evaluate_batch, body.get("cases"), body.get("thresholds"))
    verdict = guarded(gate, report["summary"], body.get("baseline"), body.get("tolerance", 0.05))
    return {**verdict, "summary": report["summary"], "failing": report["failing"]}
