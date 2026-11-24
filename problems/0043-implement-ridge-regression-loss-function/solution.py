import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)

    predictions = X @ w
    mse = np.mean((y_true - predictions) ** 2)
    penalty = alpha * np.sum(w ** 2)

    return float(mse + penalty)