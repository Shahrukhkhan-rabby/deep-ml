from typing import Union
import numpy as np
import torch


def accuracy_score(
    y_true: Union[torch.Tensor, list, "np.ndarray"],
    y_pred: Union[torch.Tensor, list, "np.ndarray"],
) -> float:
  """
    Compute the accuracy: fraction of matching elements in y_true and y_pred.
    Both inputs may be torch.Tensor, list, or numpy.ndarray.
    """
  if not isinstance(y_true, torch.Tensor):
    y_true = torch.as_tensor(y_true)
  if not isinstance(y_pred, torch.Tensor):
    y_pred = torch.as_tensor(y_pred)

  y_true = y_true.reshape(-1)
  y_pred = y_pred.reshape(-1)

  matches = (y_true == y_pred).to(dtype=torch.float64)
  return float(matches.mean().item())