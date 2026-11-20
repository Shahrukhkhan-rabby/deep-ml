def scalar_multiply(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
  if not matrix:
    return []

  return [[element * scalar for element in row] for row in matrix]