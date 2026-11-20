def matrix_dot_vector(
    matrix: list[list[float]], vector: list[float]
) -> list[float] | int:
  if not matrix or not matrix[0] or not vector:
    return -1

  vec_len = len(vector)
  result: list[float] = []

  for row in matrix:
    if len(row) != vec_len:
      return -1
    result.append(sum(a * b for a, b in zip(row, vector)))

  return result