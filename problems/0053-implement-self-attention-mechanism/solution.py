import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
  """Compute Query, Key, Value matrices from input X and weight matrices."""
  Q = np.dot(X, W_q)
  K = np.dot(X, W_k)
  V = np.dot(X, W_v)
  return Q, K, V


def self_attention(Q, K, V):
  """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
  Q = np.asarray(Q, dtype=np.float64)
  K = np.asarray(K, dtype=np.float64)
  V = np.asarray(V, dtype=np.float64)

  d_k = K.shape[-1]

  # 1. Compute scaled dot-product scores: (seq_len, seq_len)
  scores = (Q @ K.T) / np.sqrt(d_k)

  # 2. Row-wise softmax with max subtraction for numerical stability
  scores_max = np.max(scores, axis=-1, keepdims=True)
  exp_scores = np.exp(scores - scores_max)
  attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

  # 3. Weighted sum of values: (seq_len, d_v)
  return attention_weights @ V