import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.

    Args:
        A: 2x2 numpy array

    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    A = np.asarray(A, dtype=float)
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]

    # Combine entries into angle-sum and angle-difference components
    E = 0.5 * (a + d)
    F = 0.5 * (a - d)
    G = 0.5 * (c + b)
    H = 0.5 * (c - b)

    Q = np.hypot(E, H)
    R = np.hypot(F, G)

    # Signed singular values: s1 >= 0 and s1 * s2 = det(A)
    s1 = Q + R
    s2 = (a / s1) * d - (b / s1) * c if s1 > 0.0 else 0.0

    # alpha = theta_u - theta_v, beta = theta_u + theta_v
    alpha = np.arctan2(H, E)
    beta = np.arctan2(G, F)

    theta_u = 0.5 * (beta + alpha)
    theta_v = 0.5 * (beta - alpha)

    cu, su = np.cos(theta_u), np.sin(theta_u)
    cv, sv = np.cos(theta_v), np.sin(theta_v)

    U = np.array([[cu, -su], [su, cu]], dtype=float)
    V = np.array([[cv, sv], [-sv, cv]], dtype=float)

    # Flip sign of s2 and the corresponding row of V so s is non-negative
    if s2 < 0.0:
        s2 = -s2
        V[1, :] = -V[1, :]

    s = np.array([s1, abs(s2)], dtype=float)
    return U, s, V