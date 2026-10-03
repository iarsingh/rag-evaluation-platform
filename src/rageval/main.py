from fastapi import FastAPI
from rageval.score import evaluate

app = FastAPI()

@app.post("/evaluate")
def post_evaluate(body: dict):
    return evaluate(body["answer"], body["gold"], body["context"])
