import numpy as np

def linear(x: np.ndarray, derivative: bool = False) -> np.ndarray:
  return np.ones_like(x) if derivative else x

def sigmoid(x: np.ndarray, derivative: bool = False) -> np.ndarray:
  if derivative:
    y = sigmoid(x)
    return y * (1 - y)
  return 1.0 / (1.0 + np.exp(-x))

def relu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
  if derivative:
    x = np.where(x <= 0, 0, 1)
  return np.maximum(0, x)

def tanh(x: np.ndarray, derivative: bool = False) -> np.ndarray:
  if derivative:
    y = tanh(x)
    return 1 - y ** 2
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def leaky_relu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
  alpha = 0.1
  if derivative:
    return np.where(x <= 0, alpha, 1)
  return np.where(x <= 0, alpha * x, x)

def elu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
  alpha = 1.0
  if derivative:
      y = elu(x)
      return np.where(x <= 0, y + alpha, 1)
  return np.where(x <= 0, alpha * (np.exp(x) - 1), x)
