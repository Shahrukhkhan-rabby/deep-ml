from typing import Union
import numpy as np
import torch


def make_diagonal(
    x: Union[torch.Tensor, list, "np.ndarray"],
) -> torch.Tensor:
  """Return a diagonal matrix whose diagonal elements are the 1-D values in `x`.
    If `x` is not a torch tensor it will be converted automatically.
    """
  if not isinstance(x, torch.Tensor):
    x = torch.as_tensor(x)

  x = x.flatten()
  return torch.diag_embed(x)