import numpy as np
from functions.activation_function import sigmoid

def binary_cross_entropy(y: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> np.ndarray:
  if derivative:
    return - (y - y_pred) / (y_pred * (1 - y_pred) * y.shape[0])
  return - np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def sigmoid_binary_cross_entropy(y: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> np.ndarray:
  y_sigmoid = sigmoid(y_pred)
  if derivative:
    return - (y - y_sigmoid) / y.shape[0]
  return - np.mean(y * np.log(y_sigmoid) + (1 - y) * np.log(1 - y_sigmoid))