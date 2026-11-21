from typing import List, Tuple
import numpy as np


def k_fold_cross_validation(
    n_samples: int, k: int = 5, shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.

    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices = np.arange(n_samples, dtype=int)
    if shuffle:
        np.random.shuffle(indices)

    base_size, remainder = divmod(n_samples, k)
    splits: List[Tuple[List[int], List[int]]] = []

    start = 0
    for fold_idx in range(k):
        fold_size = base_size + (1 if fold_idx < remainder else 0)
        end = start + fold_size

        test_indices = indices[start:end].tolist()
        train_indices = np.concatenate((indices[:start], indices[end:])).tolist()

        splits.append((train_indices, test_indices))
        start = end

    return splits