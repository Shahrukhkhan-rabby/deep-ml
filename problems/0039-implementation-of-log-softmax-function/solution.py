import numpy as np


def log_softmax(scores: np.ndarray) -> np.ndarray:
  scores = np.asarray(scores, dtype=np.float64)
  max_score = np.max(scores)
  shifted = scores - max_score
  return shifted - np.log(np.sum(np.exp(shifted)))