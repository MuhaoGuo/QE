import matplotlib.pyplot as plt
import scipy
import numpy as np
from qiskit.utils import algorithm_globals
from sklearn.metrics import zero_one_loss
from sklearn.utils import shuffle
from sympy import *
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.operator import Operator
from scipy.special import expit

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def loss0_1(y_true, y_pred):
  # l = 0 if y_true = y_pred
  # l = 1 if y_true != y_pred
  return zero_one_loss(y_true, y_pred)

def l1_loss(y_true, y_pred):
  return sum(abs(y_true-y_pred))/len(y_true)


import pandas as pd
from ourNN import OurNeuralNetwork
from Qkernel import run_Q_Kernel_method
from functools import reduce
from data_loader import data_loader
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# load dataset.
X_train, y_train, X_test, y_test = data_loader()
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# X_train = scaler.inverse_transform(X_train)

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
Train = TensorDataset(X_train, y_train)

# train our NN
print("......Init NN......")
ournn = OurNeuralNetwork(learn_rate=0.001, epochs=1000)


print("......Training.......")
w1_original = ournn.w1
w2_original = ournn.w2
b1_original = ournn.b1
b2_original = ournn.b2
print(ournn.w1)
print(ournn.w2)
print(ournn.b1)
print(ournn.b2)
ournn.train(Train)
print(ournn.w1)
print(ournn.w2)
print(ournn.b1)
print(ournn.b2)



#  以下是调用各个 核函数。
# # 1. #########################################################################################
# ##########################################################################################
print("=============== w1, w2, b1, b2 = w1_original, w2_original, b1_original, b2_original===================================")
def phi_function_general(x):
    '''
    # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
    # 𝜙(xi) = xi ,    𝜙(x1, x2) = (pi - x1) * (pi - x2)
    '''
    w1, w2, b1, b2 = w1_original, w2_original, b1_original, b2_original
    coeff = (w1 * x[0] + b1) if len(x) == 1 else reduce(lambda m, n: m * n, w2 * x + b2)
    return coeff

setting = {"rep": 1, "paulis": ['Z', 'ZZ'], "func": phi_function_general, 'alpha': 2}
setting = setting
run_Q_Kernel_method(X_train, y_train, X_test, y_test,
                    reps=setting["rep"],
                    alpha=setting["alpha"],
                    paulis=setting["paulis"],
                    func=setting["func"])
#
# # 2. #########################################################################################
# ##########################################################################################
print("================w1, w2, b1, b2 = 1, 0, -1, np.pi ====================")
def phi_function_general(x):
    '''
    # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
    # 𝜙(xi) = xi ,    𝜙(x1, x2) = (pi - x1) * (pi - x2)
    '''
    w1, w2, b1, b2 = 1, 0, -1, np.pi
    coeff = (w1 * x[0] + b1) if len(x) == 1 else reduce(lambda m, n: m * n, w2 * x + b2)
    return coeff
# def func_1(x: np.ndarray) -> float:
#     '''
#     # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
#     # 𝜙(xi) = xi ,    𝜙(x1, x2) = (pi - x1) * (pi - x2)
#     '''
#     coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, np.pi - x)
#     return coeff

setting = {"rep": 1, "paulis": ['Z', 'ZZ'], "func": phi_function_general, 'alpha': 2}
setting = setting
run_Q_Kernel_method(X_train, y_train, X_test, y_test,
                    reps=setting["rep"],
                    alpha=setting["alpha"],
                    paulis=setting["paulis"],
                    func=setting["func"])
#
# #  3. #########################################################################################
# ##########################################################################################
print("=============== w1, w2, b1, b2 = 1, 1, 1, 1===================================")


def phi_function_general(x):
    '''
    # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
    # 𝜙(xi) = xi ,    𝜙(x1, x2) = (pi - x1) * (pi - x2)
    '''
    w1, w2, b1, b2 = 1, 1, 1, 1
    coeff = (w1 * x[0] + b1) if len(x) == 1 else reduce(lambda m, n: m * n, w2 * x + b2)
    return coeff

setting = {"rep": 1, "paulis": ['Z', 'ZZ'], "func": phi_function_general, 'alpha': 2}
setting = setting
run_Q_Kernel_method(X_train, y_train, X_test, y_test,
                    reps=setting["rep"],
                    alpha=setting["alpha"],
                    paulis=setting["paulis"],
                    func=setting["func"])
#
# #  4. #########################################################################################
# ##########################################################################################
print("=============== w1, w2, b1, b2 = ournn.w1, ournn.w2, ournn.b1, ournn.b2 ===================================")
def phi_function_general(x):
    w1, w2, b1, b2 = ournn.w1, ournn.w2, ournn.b1, ournn.b2
    '''
    # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
    # 𝜙(xi) = xi ,    𝜙(x1, x2) = (pi - x1) * (pi - x2)
    '''
    coeff = (w1 * x[0] + b1) if len(x) == 1 else reduce(lambda m, n: m * n, w2 * x + b2)
    return coeff

setting = {"rep": 1, "paulis": ['Z', 'ZZ'], "func": phi_function_general, 'alpha': 2}
setting = setting
run_Q_Kernel_method(X_train, y_train, X_test, y_test,
                    reps=setting["rep"],
                    alpha=setting["alpha"],
                    paulis=setting["paulis"],
                    func=setting["func"])
#
