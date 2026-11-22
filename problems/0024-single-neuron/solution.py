import math


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def single_neuron_model(
    features: list[list[float]],
    labels: list[int],
    weights: list[float],
    bias: float,
) -> tuple[list[float], float]:
    if not features:
        return [], 0.0

    raw_probs: list[float] = []
    for x in features:
        z = sum(w * xi for w, xi in zip(weights, x)) + bias
        raw_probs.append(_sigmoid(z))

    mse_raw = sum((p - y) ** 2 for p, y in zip(raw_probs, labels)) / len(labels)
    probabilities = [round(p, 4) for p in raw_probs]
    mse = round(mse_raw, 4)
    return probabilities, mse