import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
  A_arr = np.asarray(A, dtype=float)
  b_arr = np.asarray(b, dtype=float)

  num_vars = b_arr.shape[0]
  x = np.zeros(num_vars, dtype=float)
  diag = np.diag(A_arr)

  if np.any(diag == 0):
    raise ValueError("Zero diagonal entry encountered; Jacobi requires A[i, i] != 0.")

  # Off-diagonal matrix: zero out the main diagonal so row dot products exclude i == j
  off_diag = A_arr.copy()
  np.fill_diagonal(off_diag, 0.0)

  for _ in range(n):
    x = (b_arr - off_diag @ x) / diag

  return [round(float(val), 4) for val in x]