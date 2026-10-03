import re

STOP = {"the", "a", "an", "is", "of", "and", "to", "in", "what", "why", "does"}

def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP

def evaluate(answer, gold, context):
    answer_words = words(answer)
    gold_words = words(gold)
    context_words = words(context)
    faithfulness = len(answer_words & context_words) / len(answer_words) if answer_words else 0
    correctness = len(answer_words & gold_words) / len(gold_words) if gold_words else 0
    return {
        "faithfulness": round(faithfulness, 4),
        "correctness": round(correctness, 4),
        "passed": faithfulness >= 0.5 and correctness >= 0.5,
    }
