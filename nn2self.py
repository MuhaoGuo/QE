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

# 自测：==========================
train = pd.read_csv('./quantum_circuit_designed_dataset/TEST_phi1_repeat1_train.csv')
y_train, X_train_0, X_train_1 = np.array(train["y_train"]), train["feature1"], train["feature2"]
X_train = np.array([X_train_0, X_train_1]).T

test = pd.read_csv('./quantum_circuit_designed_dataset/TEST_phi1_repeat1_test.csv')
y_test, X_test_0, X_test_1 = np.array(test["y_test"]), test["feature1"], test["feature2"]
X_test = np.array([X_test_0, X_test_1]).T

ournn = OurNeuralNetwork(learn_rate = 0.001, epochs = 1000)

for i, x in enumerate(X_train):
    y_true = y_train[i]
    y_pred = ournn.feedforward(x)
    print("y_pred", y_pred)
    print("y_true", y_true)

w1_original = ournn.w1
w2_original = ournn.w2
b1_original = ournn.b1
b2_original = ournn.b2
print(ournn.w1)
print(ournn.w2)
print(ournn.b1)
print(ournn.b2)
ournn.train(X_train, y_train)
print(ournn.w1)
print(ournn.w2)
print(ournn.b1)
print(ournn.b2)

print("=============== w1, w2, b1, b2 = w1_original, w2_original, b1_original, b2_original===================================")
# w1, w2, b1, b2 = 1, 0, -1, np.pi
# w1, w2, b1, b2 = 1, 1, 1, 1
# w1, w2, b1, b2 = ournn.w1, ournn.w2, ournn.b1, ournn.b2
w1, w2, b1, b2 = w1_original, w2_original, b1_original, b2_original

def phi_function_general(x):
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




### ###### ###### ###### ###### ###### ###### ###### ###
print("================w1, w2, b1, b2 = 1, 0, -1, np.pi ====================")
w1, w2, b1, b2 = 1, 0, -1, np.pi
# w1, w2, b1, b2 = 1, 1, 1, 1
# w1, w2, b1, b2 = ournn.w1, ournn.w2, ournn.b1, ournn.b2
def phi_function_general(x):
    '''
    # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
    # 𝜙(xi) = xi ,    𝜙(x1, x2) = (pi - x1) * (pi - x2)
    '''
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



### ###### ###### ###### ###### ###### ###### ###### ###
print("=============== w1, w2, b1, b2 = 1, 1, 1, 1===================================")
# w1, w2, b1, b2 = 1, 0, -1, np.pi
w1, w2, b1, b2 = 1, 1, 1, 1
# w1, w2, b1, b2 = ournn.w1, ournn.w2, ournn.b1, ournn.b2
def phi_function_general(x):
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

### ###### ###### ###### ###### ###### ###### ###### ###
print("=============== w1, w2, b1, b2 = ournn.w1, ournn.w2, ournn.b1, ournn.b2 ===================================")
# w1, w2, b1, b2 = 1, 0, -1, np.pi
# w1, w2, b1, b2 = 1, 1, 1, 1
w1, w2, b1, b2 = ournn.w1, ournn.w2, ournn.b1, ournn.b2
def phi_function_general(x):
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




# train = pd.read_csv('./quantum_circuit_designed_dataset/phi1_repeat1_train.csv')
# y_train, X_train_0, X_train_1 = np.array(train["y_train"]), train["feature1"], train["feature2"]
# X_train = np.array([X_train_0, X_train_1]).T
#
# test = pd.read_csv('./quantum_circuit_designed_dataset/phi1_repeat1_test.csv')
# y_test, X_test_0, X_test_1 = np.array(test["y_test"]), test["feature1"], test["feature2"]
# X_test = np.array([X_test_0, X_test_1]).T
#
# y_train = [-1 if i == 0 else 1 for i in y_train]
# y_test = [-1 if i == 0 else 1 for i in y_test]
#
# # print(X_train)
# # print(y_train)
# # Train our neural network!
# from ourNN import OurNeuralNetwork
#
# network = OurNeuralNetwork(learn_rate = 0.00001, epochs = 1)
# print("======== initial===== ")
#
# print("w1", network.w1)
# print("w2", network.w2)
# print("w3", network.w3)
# print("w4", network.w4)
# print("b1", network.b1)
# print("b2", network.b2)
# print("b3", network.b3)
# print("b4", network.b4)
#
# # X_train, y_train = shuffle(X_train, y_train)
# network.train(X_train, y_train)
#
# print("======== end ===== ")
# print("w1", network.w1)
# print("w2", network.w2)
# print("w3", network.w3)
# print("w4", network.w4)
# print("b1", network.b1)
# print("b2", network.b2)
# print("b3", network.b3)
# print("b4", network.b4)
#
#
# # # Make some predictions
# # emily = np.array([-7, -3]) # 128 pounds, 63 inches
# # frank = np.array([20, 2])  # 155 pounds, 68 inches
# # print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
# # print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M


