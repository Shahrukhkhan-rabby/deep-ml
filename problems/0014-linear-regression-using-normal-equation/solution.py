import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # Closed-form normal equation: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.pinv(X_arr.T @ X_arr) @ (X_arr.T @ y_arr)

    return [round(float(val), 4) for val in theta]