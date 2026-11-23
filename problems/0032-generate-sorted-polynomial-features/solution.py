from itertools import combinations_with_replacement
import numpy as np
from tinygrad.tensor import Tensor


def polynomial_features(X: Tensor, degree: int) -> Tensor:
  """
    Generate all polynomial feature combinations up to the given degree,
    then sort the resulting features for each sample from lowest to highest.

    Args:
        X: 2D Tensor of shape (n_samples, n_features)
        degree: Maximum polynomial degree (inclusive)
    Returns:
        Tensor of shape (n_samples, n_poly_features) with rows sorted ascending
    """
  if not isinstance(X, Tensor):
    X = Tensor(X)

  n_samples, n_features = X.shape
  bias = Tensor.ones(n_samples, dtype=X.dtype)
  terms = [bias]

  for d in range(1, degree + 1):
    for comb in combinations_with_replacement(range(n_features), d):
      term = bias
      for col_idx in comb:
        term = term * X[:, col_idx]
      terms.append(term)

  stacked = Tensor.stack(terms, dim=1)
  sorted_features, _ = stacked.sort(dim=-1)
  return sorted_features