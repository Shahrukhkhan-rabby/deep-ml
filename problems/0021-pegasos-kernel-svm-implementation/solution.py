import numpy as np


def pegasos_kernel_svm(
    data: np.ndarray,
    labels: np.ndarray,
    kernel: str = "linear",
    lambda_val: float = 0.01,
    iterations: int = 100,
    sigma: float = 1.0,
) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.

    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')

    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    X = np.asarray(data, dtype=float)
    y = np.asarray(labels, dtype=float)
    n_samples = X.shape[0]

    # Precompute kernel matrix K[j, i] = K(x_j, x_i)
    if kernel == "linear":
        K = X @ X.T
    elif kernel == "rbf":
        # ||x_j - x_i||^2 = ||x_j||^2 + ||x_i||^2 - 2 * (x_j . x_i)
        sq_norms = np.sum(X**2, axis=1, keepdims=True)
        dist_sq = np.maximum(sq_norms + sq_norms.T - 2.0 * (X @ X.T), 0.0)
        K = np.exp(-dist_sq / (2.0 * (sigma**2)))
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    alphas = np.zeros(n_samples, dtype=float)
    bias = 0.0

    for t in range(1, iterations + 1):
        eta = 1.0 / (lambda_val * t)
        for i in range(n_samples):
            # g_i = sum_j alpha_j * y_j * K(x_j, x_i) + bias
            g_i = float(np.dot(alphas * y, K[:, i]) + bias)

            if y[i] * g_i < 1.0:
                alphas[i] = alphas[i] + eta * (1.0 - lambda_val * alphas[i])
                bias = bias + eta * y[i]
            else:
                alphas[i] = alphas[i] - eta * lambda_val * alphas[i]

    rounded_alphas = [round(float(a), 4) for a in alphas]
    rounded_bias = round(float(bias), 4)
    return rounded_alphas, rounded_bias