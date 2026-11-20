def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
  if not matrix or not matrix[0]:
    return []

  num_rows = len(matrix)
  num_cols = len(matrix[0])

  if any(len(row) != num_cols for row in matrix):
    raise ValueError("Matrix must have rows of equal length.")

  if mode == "row":
    return [sum(row) / num_cols for row in matrix]
  elif mode == "column":
    return [
        sum(matrix[row][col] for row in range(num_rows)) / num_rows
        for col in range(num_cols)
    ]
  else:
    raise ValueError("Mode must be 'row' or 'column'.")