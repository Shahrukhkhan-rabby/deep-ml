import numpy as np


def adam_optimizer(
    f,
    grad,
    x0,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    num_iterations=10,
):
    """
    Minimize an objective function using the Adam optimization algorithm.

    Args:
        f: Objective function (accepted for interface compatibility; unused here)
        grad: Function computing gradient of f at parameter vector x
        x0: Initial parameter values
        learning_rate: Step size alpha (default: 0.001)
        beta1: Exponential decay rate for first moment estimate (default: 0.9)
        beta2: Exponential decay rate for second moment estimate (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
        num_iterations: Number of optimization steps to perform (default: 10)

    Returns:
        np.ndarray of optimized parameters
    """
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, num_iterations + 1):
        g = np.asarray(grad(x), dtype=float)

        # Update biased first and second moment estimates
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)

        # Compute bias-corrected moment estimates
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x