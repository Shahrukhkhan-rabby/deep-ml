import numpy as np


def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    m = b.shape[0]

    # Forward elimination with partial pivoting
    for k in range(m):
        pivot_row = k + int(np.argmax(np.abs(A[k:, k])))

        if np.isclose(A[pivot_row, k], 0.0):
            raise ValueError("Matrix A is singular or nearly singular.")

        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        for i in range(k + 1, m):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Backward substitution
    x = np.zeros(m, dtype=float)
    for i in range(m - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])) / A[i, i]

    return x