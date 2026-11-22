import math


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_values = [math.exp(s - max_score) for s in scores]
    total = sum(exp_values)
    return [round(val / total, 4) for val in exp_values]