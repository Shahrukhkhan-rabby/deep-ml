import numpy as np


def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.

    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')

    Returns:
        Optimized weights
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float).copy()

    m = X.shape[0]
    if m == 0:
        return w

    if method == 'batch':
        step_size = m
    elif method == 'stochastic':
        step_size = 1
    elif method == 'mini_batch':
        step_size = max(1, int(batch_size))
    else:
        raise ValueError(f"Unknown method: {method}. Expected 'batch', 'stochastic', or 'mini_batch'.")

    for _ in range(n_epochs):
        for start in range(0, m, step_size):
            X_b = X[start:start + step_size]
            y_b = y[start:start + step_size]
            k = X_b.shape[0]

            error = X_b @ w - y_b
            grad = (2.0 / k) * (X_b.T @ error)
            w = w - learning_rate * grad

    return w