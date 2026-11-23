from typing import Optional, Union
import numpy as np
import torch


def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, "np.ndarray"],
    Y: Optional[Union[torch.Tensor, list, "np.ndarray"]] = None,
) -> torch.Tensor:
  """
    Compute the Pearson correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """
  X_t = (
      X.to(dtype=torch.float32)
      if isinstance(X, torch.Tensor)
      else torch.as_tensor(X, dtype=torch.float32)
  )
  Y_t = (
      X_t
      if Y is None
      else (
          Y.to(dtype=torch.float32)
          if isinstance(Y, torch.Tensor)
          else torch.as_tensor(Y, dtype=torch.float32)
      )
  )

  X_centered = X_t - X_t.mean(dim=0, keepdim=True)
  Y_centered = Y_t - Y_t.mean(dim=0, keepdim=True)

  cov = X_centered.T @ Y_centered
  x_norm = torch.linalg.norm(X_centered, dim=0, keepdim=True).T
  y_norm = torch.linalg.norm(Y_centered, dim=0, keepdim=True)

  return cov / (x_norm @ y_norm)