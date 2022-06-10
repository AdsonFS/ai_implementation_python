import numpy as np
from typing import Callable

from functions.activation_function import linear

class Layer:
  def __init__(self, input_dim: int, output_dim: int, activation: Callable[[np.ndarray, bool], np.ndarray] = linear) -> None:
    self.input: np.ndarray
    self.weights: np.ndarray = np.random.rand(output_dim, input_dim)
    self.biases: np.ndarray = np.random.rand(1, output_dim)
    self.activation: Callable[[np.ndarray, bool], np.ndarray] = activation

    self._activ_inp: np.ndarray
    self._activ_out: np.ndarray

    self._d_weights: np.ndarray
    self._d_biases: np.ndarray
    

