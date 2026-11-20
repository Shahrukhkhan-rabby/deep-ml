import numpy as np


def reshape_matrix(
    a: list[list[int | float]], new_shape: tuple[int, int]
) -> list[list[int | float]]:
  if not a or not a[0]:
    return []

  new_rows, new_cols = new_shape
  if new_rows <= 0 or new_cols <= 0:
    return []

  arr = np.asarray(a)
  if arr.size != new_rows * new_cols:
    return []

  reshaped_matrix = arr.reshape((new_rows, new_cols))
  return reshaped_matrix.tolist()