import math


def sigmoid(z: float) -> float:
    if z >= 0:
        result = 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        result = exp_z / (1.0 + exp_z)
    return round(result, 4)