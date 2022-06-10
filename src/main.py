from turtle import shape
from typing import Final
import numpy as np
import pandas as pd
# import _pickle as pkl
import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.datasets import make_blobs, make_circles, make_moons, make_classification
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# from functions.activation_function import sigmoid
# from functions.cost_function.binary_classification import binary_cross_entropy
from functions.activation_function import linear, sigmoid
from functions.cost_function.regression import mse
from models.layer import Layer
from models.neural_network import NeuralNetwork


listaColunas = np.array([[f'X{i}1', f'X{i}2'] for i in range(1, 10)]).reshape(1, -1)[0]
# print(listaColunas)
f = lambda x : '1' if x == 'A' else '0' 
df = pd.read_csv('data\data_edu.csv', converters={'Resultado': f})
x = np.array(df[listaColunas].astype(int))
y = np.array(df['Resultado'].astype(int)).reshape(-1, 1)
# print(y)

# x = np.array([[0.05, 0.10]])
# y = np.array([[0.01, 0.99]])

# w1 = np.array([[0.15, .20], [0.25, 0.30]])
# b1 = np.array([[0.35, 0.35]])
# w2 = np.array([[0.40, 0.45], [0.50, 0.55]])
# b2 = np.array([[0.60, 0.60]])


# montando a rede
input_dim: int = x.shape[1]
output_dim: int = y.shape[1]
nn: NeuralNetwork = NeuralNetwork(cost_function=mse, learning_rate= 1)
nn.layers.append(Layer(input_dim, 5, activation=sigmoid))
nn.layers.append(Layer(5, output_dim, activation=sigmoid))

nn.fit(x, y, epochs=6000, verbose=10)
 
plt.plot(nn.epochs_plot, nn.error_plot)
plt.show()

# for layer in nn.layers:
  # print(layer.weights)








