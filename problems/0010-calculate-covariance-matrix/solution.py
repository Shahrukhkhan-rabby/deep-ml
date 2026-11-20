def calculate_covariance_matrix(
    vectors: list[list[float]],
) -> list[list[float]]:
  if not vectors or not vectors[0]:
    return []

  n_features = len(vectors)
  n_obs = len(vectors[0])

  if n_obs < 2:
    raise ValueError(
        "Each feature must have at least 2 observations to compute sample"
        " covariance."
    )
  if any(len(vec) != n_obs for vec in vectors):
    raise ValueError("All feature vectors must have the same length.")

  # Compute mean and mean-centered observations for each feature
  means = [sum(vec) / n_obs for vec in vectors]
  centered = [[val - means[i] for val in vec] for i, vec in enumerate(vectors)]

  denom = n_obs - 1
  cov_matrix: list[list[float]] = [
      [0.0] * n_features for _ in range(n_features)
  ]

  for i in range(n_features):
    for j in range(i, n_features):
      cov_ij = (
          sum(centered[i][k] * centered[j][k] for k in range(n_obs)) / denom
      )
      val = float(cov_ij)
      cov_matrix[i][j] = val
      cov_matrix[j][i] = val

  return cov_matrix