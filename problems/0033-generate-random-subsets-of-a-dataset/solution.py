import numpy as np
from tinygrad.tensor import Tensor


def get_random_subsets(
    X: Tensor, y: Tensor, n_subsets: int, replacements: bool = True
):
  """
    Generate random subsets of a dataset using Tinygrad Tensors.

    Args:
        X: 2D Tensor of features
        y: 1D Tensor of labels
        n_subsets: Number of subsets to generate
        replacements: If True, sample with replacement (subset size = n);
                      if False, sample without replacement (subset size = n // 2)

    Returns:
        List of n_subsets tuples, each tuple is (X_subset_list, y_subset_list)
    """
  X_np = X.numpy() if isinstance(X, Tensor) else np.asarray(X)
  y_np = y.numpy() if isinstance(y, Tensor) else np.asarray(y)

  n = X_np.shape[0]
  subset_size = n if replacements else n // 2

  subsets = []
  for _ in range(n_subsets):
    idx = np.random.choice(n, subset_size, replace=replacements)
    X_sub = X_np[idx].tolist()
    y_sub = y_np[idx].tolist()
    subsets.append((X_sub, y_sub))

  return subsets