import torch
from torch.utils.data import TensorDataset, DataLoader


def batch_iterator(X, y=None, batch_size=64):
  """
    Batch iterator for a dataset represented as PyTorch tensors.

    Args:
        X: Tensor of shape (n_samples, n_features)
        y: Optional Tensor of shape (n_samples,)
        batch_size: Number of samples per batch

    Returns:
        List of Tensor batches (or [X_batch, y_batch] pairs if y is provided)
    """
  if not isinstance(X, torch.Tensor):
    X = torch.as_tensor(X)

  if y is not None:
    if not isinstance(y, torch.Tensor):
      y = torch.as_tensor(y)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return [[x_batch, y_batch] for x_batch, y_batch in loader]

  dataset = TensorDataset(X)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  return [batch[0] for batch in loader]