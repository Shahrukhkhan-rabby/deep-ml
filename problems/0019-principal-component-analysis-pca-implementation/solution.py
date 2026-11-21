import numpy as np


def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.

    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return

    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    arr = np.asarray(data, dtype=float)
    n_samples = arr.shape[0]

    # 1. Standardize features (mean 0, population std 1)
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, ddof=0, keepdims=True)
    standardized = (arr - mean) / std

    # 2. Compute symmetric covariance matrix
    cov_matrix = (standardized.T @ standardized) / max(n_samples - 1, 1)

    # 3. Eigen-decompose symmetric matrix (ascending eigenvalues) and select top k descending
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1][:k]
    top_vectors = eigenvectors[:, sorted_indices]

    # 4. Canonical sign convention per eigenvector column
    for col in range(top_vectors.shape[1]):
        vec = top_vectors[:, col]
        for val in vec:
            if abs(val) > 1e-10:
                if val < 0:
                    top_vectors[:, col] = -vec
                break

    return np.round(top_vectors, 4)