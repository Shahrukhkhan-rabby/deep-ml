import numpy as np


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
  """Compute SVD of a 2x2 matrix using one Jacobi rotation.

  Args:
      A: A 2x2 numpy array

  Returns:
      Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
      - U: 2x2 orthogonal matrix
      - S: length-2 array of singular values
      - Vt: 2x2 orthogonal matrix (transpose of V)
  """
  A_arr = np.asarray(A, dtype=float)

  # 1. Form symmetric Gram matrix G = A^T @ A
  G = A_arr.T @ A_arr
  g00, g01, g11 = G[0, 0], G[0, 1], G[1, 1]

  # 2. Compute single Jacobi rotation angle: cot(2*theta) = (g00 - g11) / (2 * g01)
  if abs(g01) < 1e-15:
    c, s = 1.0, 0.0
  else:
    tau = (g00 - g11) / (2.0 * g01)
    if abs(tau) < 1e-15:
      t = 1.0
    elif tau > 0:
      t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
    else:
      t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
    c = 1.0 / np.sqrt(1.0 + t * t)
    s = t * c

  V = np.array([[c, -s], [s, c]], dtype=float)

  # 3. Project A onto V; orthogonal columns have norms equal to singular values
  AV = A_arr @ V
  s0 = float(np.linalg.norm(AV[:, 0]))
  s1 = float(np.linalg.norm(AV[:, 1]))

  # 4. Sort singular values descending and align V and U columns
  if s0 < s1:
    s0, s1 = s1, s0
    V = V[:, [1, 0]]
    AV = AV[:, [1, 0]]

  S = np.array([s0, s1], dtype=float)
  u0 = AV[:, 0] / s0
  u1 = AV[:, 1] / s1
  U = np.column_stack((u0, u1))
  Vt = V.T

  return U, S, Vt