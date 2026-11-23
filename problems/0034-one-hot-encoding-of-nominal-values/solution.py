from tinygrad import dtypes
from tinygrad.tensor import Tensor


def to_categorical(x: Tensor, n_col=None) -> Tensor:
  """
    Perform one-hot encoding of nominal values using Tinygrad.

    Args:
        x: 1D Tensor of integer values
        n_col: Optional number of columns for the one-hot encoded output.
               If not provided, determined automatically from input.

    Returns:
        2D Tensor of shape (len(x), n_col) with one-hot encoding.
    """
  if not isinstance(x, Tensor):
    x = Tensor(x)

  x = x.flatten().cast(dtypes.int)
  if n_col is None:
    n_col = int(x.max().item()) + 1

  col_indices = Tensor.arange(n_col, dtype=dtypes.int).unsqueeze(0)
  row_values = x.unsqueeze(1)
  return (row_values == col_indices).cast(dtypes.float)