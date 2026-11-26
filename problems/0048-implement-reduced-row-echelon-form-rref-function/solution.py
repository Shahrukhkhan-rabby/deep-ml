import numpy as np


def rref(matrix, tol=1e-10):
    """
    Convert a 2D matrix into Reduced Row Echelon Form (RREF).

    Args:
        matrix: Array-like of shape (m, n)
        tol: Numerical tolerance for treating near-zero values as zero

    Returns:
        np.ndarray of shape (m, n) in Reduced Row Echelon Form
    """
    A = np.array(matrix, dtype=float, copy=True)
    if A.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    m, n = A.shape
    pivot_row = 0

    for col in range(n):
        if pivot_row >= m:
            break

        # Find row with largest magnitude in current column at or below pivot_row
        max_row = pivot_row + int(np.argmax(np.abs(A[pivot_row:, col])))
        if np.abs(A[max_row, col]) <= tol:
            continue

        # Swap pivot row into place
        if max_row != pivot_row:
            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]

        # Normalize pivot row so leading entry is 1
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]

        # Eliminate all other entries in this column (above and below)
        for r in range(m):
            if r != pivot_row and np.abs(A[r, col]) > tol:
                A[r] = A[r] - A[r, col] * A[pivot_row]

        pivot_row += 1

    # Clean up tiny floating-point residuals and negative zeros (-0.0)
    A[np.abs(A) <= tol] = 0.0
    A = A + 0.0
    return A