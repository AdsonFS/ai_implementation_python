import numpy as np

def mae(y: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> np.ndarray:
  if derivative:
    return np.where(y_pred > y, 1, -1) / y.shape[0]
  return np.mean(np.abs(y - y_pred))

def mse(y: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> np.ndarray:
  if derivative:
    return - (y - y_pred) / y.shape[0]
  return 0.5 * np.mean((y - y_pred) ** 2)
