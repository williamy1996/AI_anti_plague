import numpy as np


def smape(y_pred, y_true):
    sample_size = len(y_pred)
    numerator = np.abs(y_pred-y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true))/2
    return np.sum(np.divide(numerator, denominator))/sample_size
