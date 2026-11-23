import math
from typing import Dict, List
import torch


def adaboost_fit(X, y, n_clf) -> List[Dict]:
    X_t = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    y_t = (
        y.to(dtype=X_t.dtype)
        if isinstance(y, torch.Tensor)
        else torch.as_tensor(y, dtype=X_t.dtype)
    ).reshape(-1)

    n_samples, n_features = X_t.shape
    w = torch.full((n_samples,), 1.0 / n_samples, dtype=torch.float64, device=X_t.device)

    clfs: List[Dict] = []
    eps_const = 1e-10

    for _ in range(n_clf):
        best_error = float("inf")
        best_polarity = 1
        best_threshold = 0.0
        best_feature_idx = 0

        for j in range(n_features):
            col = X_t[:, j]
            thresholds = torch.unique(col, sorted=True)

            for t_val in thresholds:
                polarity = 1
                # selection: strict '>' comparison
                pred = torch.where(col > t_val, torch.ones_like(y_t), -torch.ones_like(y_t))
                error = float(w[pred != y_t].sum().item())

                if error > 0.5:
                    polarity = -1
                    error = 1.0 - error

                if error < best_error:
                    best_error = error
                    best_polarity = polarity
                    best_threshold = float(t_val.item())
                    best_feature_idx = j

        alpha = 0.5 * math.log((1.0 - best_error) / (best_error + eps_const))

        # update: predict -1 where polarity * x < polarity * threshold, else +1
        col = X_t[:, best_feature_idx]
        h = torch.ones_like(y_t)
        h[best_polarity * col < best_polarity * best_threshold] = -1

        w = w * torch.exp(-alpha * y_t.to(torch.float64) * h.to(torch.float64))
        w = w / w.sum()

        clfs.append({
            "polarity": best_polarity,
            "threshold": best_threshold,
            "feature_index": best_feature_idx,
            "alpha": alpha,
        })

    return clfs