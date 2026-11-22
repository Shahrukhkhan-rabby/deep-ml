def transform_basis(B: list[list[float]], C: list[list[float]]) -> list[list[float]]:
    n = len(C)
    if n == 0:
        return []

    # Build augmented matrix [C | I]
    aug = [
        [float(C[i][j]) for j in range(n)]
        + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]

    # Gauss-Jordan elimination with partial pivoting
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) == 0.0:
            raise ValueError("Matrix C is singular and cannot be inverted.")

        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    # Extract C_inv
    c_inv = [row[n:] for row in aug]

    # Multiply P = C_inv @ B
    m = len(B[0]) if B else 0
    P = [
        [
            sum(c_inv[i][k] * float(B[k][j]) for k in range(n))
            for j in range(m)
        ]
        for i in range(n)
    ]

    return P