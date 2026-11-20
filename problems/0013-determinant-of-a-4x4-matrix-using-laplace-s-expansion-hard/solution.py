def determinant_4x4(matrix: list[list[int | float]]) -> int | float:
  """Calculate the determinant of a square matrix (e.g., 4x4) via Laplace expansion."""
  n = len(matrix)
  if n == 0 or any(len(row) != n for row in matrix):
    raise ValueError("Input must be a non-empty square matrix.")

  # Base cases
  if n == 1:
    return matrix[0][0]
  if n == 2:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

  total = 0
  for col in range(n):
    element = matrix[0][col]
    # Skip recursive determinant call when multiplier is 0
    if element == 0:
      continue

    minor = [row[:col] + row[col + 1 :] for row in matrix[1:]]
    sign = 1 if col % 2 == 0 else -1
    total += sign * element * determinant_4x4(minor)

  return total