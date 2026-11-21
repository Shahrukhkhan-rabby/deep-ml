import numpy as np


def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(data, dtype=float)

    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, ddof=0, keepdims=True)
    standardized = (arr - mean) / std

    min_val = np.min(arr, axis=0, keepdims=True)
    max_val = np.max(arr, axis=0, keepdims=True)
    minmax = (arr - min_val) / (max_val - min_val)

    return np.round(standardized, 4), np.round(minmax, 4)