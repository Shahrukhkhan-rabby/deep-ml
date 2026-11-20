import math


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
  a, b = matrix[0][0], matrix[0][1]
  c, d = matrix[1][0], matrix[1][1]

  trace = a + d
  det = a * d - b * c
  discriminant = trace**2 - 4 * det

  # Guard against tiny negative floating-point noise on repeated eigenvalues
  if -1e-12 < discriminant < 0:
    discriminant = 0.0

  if discriminant < 0:
    raise ValueError(
        "Matrix has complex eigenvalues; expected real eigenvalues."
    )

  sqrt_disc = math.sqrt(discriminant)
  lambda_1 = (trace + sqrt_disc) / 2.0
  lambda_2 = (trace - sqrt_disc) / 2.0

  eigenvalues = [float(lambda_1), float(lambda_2)]
  eigenvalues.sort(reverse=True)
  return eigenvalues