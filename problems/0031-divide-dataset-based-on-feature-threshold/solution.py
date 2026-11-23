from typing import List
import torch


def divide_on_feature(X, feature_i, threshold) -> List[torch.Tensor]:
  """
    Divide the tensor X into two subsets based on whether X[:, feature_i]
    is greater than or equal to the threshold.
    Return two tensors: one with samples that meet the condition,
    one with samples that do not.
    """
  if not isinstance(X, torch.Tensor):
    X = torch.as_tensor(X)

  mask = X[:, feature_i] >= threshold
  return [X[mask], X[~mask]]