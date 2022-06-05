import numpy as np

def softmax(x: np.ndarray, y_oh: np.ndarray = None, derivative: bool = False) -> np.ndarray:
  if derivative:
    y_pred = softmax(x)
    k = np.nonzero(y_pred * y_oh)
    pk = y_pred[k]
    y_pred[k] = pk * (1 - pk)
    return y_pred
  exp = np.exp(x)
  return exp / np.sum(exp, axis = 1, keepdims=True)

def neg_log_likelihood(y_oh: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> np.ndarray:
  k = np.nonzero(y_pred * y_oh)
  pk = y_pred[k]
  if derivative:
    y_pred[k] = (-1.0 / pk)
    return y_pred
  return np.mean(-np.log(pk))

def softmax_neg_log_likelihood(y_oh: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> np.ndarray:
  y_softmax = softmax(y_pred)
  if derivative:
    k = np.nonzero(y_pred * y_oh)
    d_log = neg_log_likelihood(y_oh, y_softmax, True)
    d_softmax = softmax(y_pred, y_oh, True)
    y_softmax[k] = d_log[k] * d_softmax[k]
    return y_softmax / y_softmax.shape[0]
  return neg_log_likelihood(y_oh, y_pred)