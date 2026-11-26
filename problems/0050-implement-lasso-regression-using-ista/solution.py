import numpy as np


def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.

    S(w, λ) = sign(w) * max(|w| - λ, 0)

    Args:
        w: Input array
        threshold: Threshold value λ

    Returns:
        Soft-thresholded array where values with |w| <= λ become 0
    """
    w = np.asarray(w, dtype=float)
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)


def l1_regularization_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on L2 weight change

    Returns:
        tuple: (weights, bias)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    for _ in range(max_iter):
        residuals = X @ weights + bias - y
        grad_w = (1.0 / n_samples) * (X.T @ residuals)
        grad_b = (1.0 / n_samples) * np.sum(residuals)

        w_temp = weights - learning_rate * grad_w
        bias = bias - learning_rate * grad_b

        new_weights = soft_threshold(w_temp, learning_rate * alpha)

        if np.linalg.norm(new_weights - weights) < tol:
            weights = new_weights
            break

        weights = new_weights

    return weights, float(bias)