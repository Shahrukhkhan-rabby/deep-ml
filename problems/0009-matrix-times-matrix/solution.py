def matrixmul(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]] | int:
  if not a or not a[0] or not b or not b[0]:
    return -1

  rows_a, cols_a = len(a), len(a[0])
  rows_b, cols_b = len(b), len(b[0])

  # Ensure rows are rectangular and inner dimensions match
  if any(len(row) != cols_a for row in a) or any(
      len(row) != cols_b for row in b
  ):
    return -1
  if cols_a != rows_b:
    return -1

  # Transpose B so each column becomes a row for clean dot products
  b_cols = list(zip(*b))

  return [
      [sum(x * y for x, y in zip(row_a, col_b)) for col_b in b_cols]
      for row_a in a
  ]