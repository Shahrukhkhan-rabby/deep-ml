import numpy as np


def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]],
) -> list[list[int | float]] | int:
  arr_A = np.asarray(A, dtype=float)
  arr_T = np.asarray(T, dtype=float)
  arr_S = np.asarray(S, dtype=float)

  # Ensure 2D non-empty inputs
  if arr_A.ndim != 2 or arr_T.ndim != 2 or arr_S.ndim != 2:
    return -1
  if arr_A.size == 0 or arr_T.size == 0 or arr_S.size == 0:
    return -1

  # Check squareness of T and S and shape compatibility with A
  rows_A, cols_A = arr_A.shape
  rows_T, cols_T = arr_T.shape
  rows_S, cols_S = arr_S.shape

  if rows_T != cols_T or rows_S != cols_S:
    return -1
  if rows_T != rows_A or cols_S != cols_A:
    return -1

  # Check invertibility (non-zero determinant within floating-point tolerance)
  if abs(np.linalg.det(arr_T)) < 1e-12 or abs(np.linalg.det(arr_S)) < 1e-12:
    return -1

  try:
    inv_T = np.linalg.inv(arr_T)
    transformed_matrix = inv_T @ arr_A @ arr_S
    return transformed_matrix.tolist()
  except np.linalg.LinAlgError:
    return -1