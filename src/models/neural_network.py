from typing import List, Callable, Optional
import numpy as np

from functions.cost_function.regression import mse
from models.layer import Layer

class NeuralNetwork:
  def __init__(self, cost_function: Callable[[np.ndarray, np.ndarray, bool], np.ndarray] = mse, learning_rate: float = 1e-3) -> None:
    self.layers: List[Layer] = []
    self.cost_function = cost_function
    self.learning_rate = learning_rate
    self.epochs_plot: List[float] = []
    self.error_plot: List[float] = []
  
  def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, verbose: int = 10):
    for epoch in range(epochs + 1):
      y_pred: np.ndarray = self.__feedforward(x_train)
      self.__backpropagation(y_train, y_pred)

      if epoch % verbose == 0:
        loss_train: np.ndarray = self.cost_function(y_train, self.predict(x_train), False)
        # print(f'epoch: {epochs:=4}/{epoch} loss_train: {loss_train:.8f}')
        self.epochs_plot.append(epoch)
        # print(loss_train)
        self.error_plot.append(float(loss_train))
        # print(f'out: {y_pred}')

  def predict(self, x: np.ndarray) -> np.ndarray:
    return self.__feedforward(x)

  def __feedforward(self, x: np.ndarray) -> np.ndarray:
    self.layers[0].input = x
    for current_layer, next_layer in zip(self.layers, self.layers[1:] + [Layer(0, 0)]):
      current_layer._activ_inp = np.dot(current_layer.input, current_layer.weights.T) + current_layer.biases
      current_layer._activ_out = next_layer.input = current_layer.activation(current_layer._activ_inp, False)
    return self.layers[-1]._activ_out

  def __backpropagation(self, y: np.ndarray, y_pred: np.ndarray):
    last_delta: np.ndarray = self.cost_function(y, y_pred, True)
    for layer in reversed(self.layers):
      d_activate: np.ndarray = layer.activation(layer._activ_inp, True) * last_delta
      last_delta = np.dot(d_activate, layer.weights)
      layer._d_weights = np.dot(d_activate.T, layer.input)
      layer._d_biases = d_activate.sum(axis=0, keepdims=True)

    for layer in reversed(self.layers):
      layer.weights = layer.weights - self.learning_rate * layer._d_weights
      layer.biases = layer.biases - self.learning_rate * layer._d_biases
