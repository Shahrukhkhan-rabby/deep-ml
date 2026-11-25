import numpy as np


def precision(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    predicted_positives = tp + fp
    if predicted_positives == 0:
        return 0.0

    return float(tp / predicted_positives)