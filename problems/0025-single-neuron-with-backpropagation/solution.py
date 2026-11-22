import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> tuple[np.ndarray, float, list[float]]:
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1, 1)
    w = np.asarray(initial_weights, dtype=float).reshape(-1, 1)
    b = float(initial_bias)

    n_samples = X.shape[0]
    mse_values: list[float] = []

    for _ in range(epochs):
        # 1. Forward pass over the entire batch
        z = X @ w + b
        preds = sigmoid(z)

        # 2. Compute MSE before updating parameters
        mse = float(np.mean((preds - y) ** 2))
        mse_values.append(round(mse, 4))

        # 3. Backpropagation: dL/dz = (2/N) * (preds - y) * preds * (1 - preds)
        dz = (2.0 / n_samples) * (preds - y) * preds * (1.0 - preds)
        dw = X.T @ dz
        db = float(np.sum(dz))

        # 4. Full-batch parameter update
        w -= learning_rate * dw
        b -= learning_rate * db

    updated_weights = np.round(w.flatten(), 4)
    updated_bias = round(b, 4)

    return updated_weights, updated_bias, mse_values