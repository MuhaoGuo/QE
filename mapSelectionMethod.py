import random
import numpy as np
import pandas as pd
import scipy
from qiskit.utils import algorithm_globals
# https://qiskit.org/documentation/stable/0.26/_modules/qiskit/utils/algorithm_globals.html
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_machine_learning.datasets.dataset_helper import (features_and_labels_transform,)
import matplotlib.pyplot as plt
from adhoc_original import *
from Qkernel import *
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# def phi_function(x_1, x_2):
#     j_m = np.array([[1, 0], [0, 1]])   # 单位矩阵？
#     j_m = np.asarray(j_m)
#
#     s_z = np.array([[1, 0], [0, -1]])   # Z gate
#     z_m = np.asarray(s_z)
#
#     s_x = np.array([[0, 1], [1, 0]])   # X gate
#     x_m = np.asarray(s_x)
#
#     h_m = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # H GATE
#     h_2 = np.kron(h_m, h_m) # 2 个 H
#     h_m = np.asarray(h_m)
#     h_2 = np.asarray(h_2)
#
#     phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
#     # ########################
#     # # 只要x1 x2 单独的。-------------
#     # phi = ( x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m))
#
#     # # 只要 x1 x2  都是先z 后 i
#     # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(z_m, j_m))
#
#     # phi = (np.pi * x_1 * np.kron(z_m, j_m)  + np.pi* x_2 * np.kron(z_m, j_m))
#
#     # phi = ( np.pi* x_1 * np.kron(z_m, j_m) + np.pi* x_2 * np.kron(j_m, z_m) )
#
#     # # 只要 x1 x2  都是先i 后 z
#     # phi = (x_1 * np.kron(j_m, z_m)+ x_2 * np.kron(j_m, z_m)  )
#
#     # # X gate 一样
#     # phi = ( x_1 * np.kron(j_m, x_m)  + x_2 * np.kron(x_m, j_m))
#
#     # # 只要混合的 x1 x2 -----------
#     # phi = ( + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m) )
#
#     # # pi/2 * (1- x1)(1 - x2) * (z gate ,z gate )
#     # phi = ( + np.pi /2 * (1 - x_1)*(1 - x_2) * np.kron(z_m, z_m)  )
#
#     # # # pi * (cos x1)(cos x2) * (z gate ,z gate )
#     # phi = ( + np.pi * np.cos(x_1) * np.cos(x_2) * np.kron(z_m, z_m)   )
#
#     # phi = ( + np.pi * np.sin(x_1) * np.cos(x_2) * np.kron(z_m, z_m)   )
#
#     # phi = ( + np.pi * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m)  )
#
#     # phi = (+ np.pi * np.tan(x_1) * np.tan(x_2) * np.kron(z_m, z_m)   )
#
#     # phi = (+ np.pi * np.tan(x_1) * np.sin(x_2) * np.kron(z_m, z_m)  )
#
#     # phi = (+ np.pi * np.cos(x_1) / np.cos(x_2) * np.kron(z_m, z_m)  )
#
#     # phi = ( + np.pi/2 * np.cos(x_1) / np.cos(x_2) * np.kron(z_m, z_m))
#
#     # phi = ( + 0.5 * x_1 + 0.5 * x_2 * np.kron(z_m, z_m))
#
#     # phi = ( + np.pi * x_1 + np.pi * x_2 * np.kron(z_m, z_m))
#
#     # phi = (np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))
#
#     # phi = (100 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))
#
#     # phi = ( 10 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))
#
#     # phi = (0.1 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m)  )
#     # ### 2 种都要
#     # # x1 Z + x2 Z +  pi * (cos x1)(cos x2) * (z gate ,z gate )
#     # phi = ( + x_1 + x_2 + x_1 * x_2 * np.kron(z_m, z_m) )
#
#     # phi = ( x_1 * np.kron(z_m, j_m)  + x_2 * np.kron(j_m, z_m)  + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
#     return phi
# def phi_function2(x_1, x_2):
#     j_m = np.array([[1, 0], [0, 1]])   # 单位矩阵？
#     j_m = np.asarray(j_m)
#
#     s_z = np.array([[1, 0], [0, -1]])   # Z gate
#     z_m = np.asarray(s_z)
#
#     s_x = np.array([[0, 1], [1, 0]])   # X gate
#     x_m = np.asarray(s_x)
#
#     h_m = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # H GATE
#     h_2 = np.kron(h_m, h_m) # 2 个 H
#     h_m = np.asarray(h_m)
#     h_2 = np.asarray(h_2)
#
#     phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
#     # ########################
#     # # 只要x1 x2 单独的。-------------
#     # phi = ( x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m))
#
#     # # 只要 x1 x2  都是先z 后 i
#     # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(z_m, j_m))
#
#     # phi = (np.pi * x_1 * np.kron(z_m, j_m)  + np.pi* x_2 * np.kron(z_m, j_m))
#
#     # phi = ( np.pi* x_1 * np.kron(z_m, j_m) + np.pi* x_2 * np.kron(j_m, z_m) )
#
#     # # 只要 x1 x2  都是先i 后 z
#     # phi = (x_1 * np.kron(j_m, z_m)+ x_2 * np.kron(j_m, z_m)  )
#
#     # # X gate 一样
#     # phi = ( x_1 * np.kron(j_m, x_m)  + x_2 * np.kron(x_m, j_m))
#
#     # # 只要混合的 x1 x2 -----------
#     phi = ( + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m) )
#
#     # # pi/2 * (1- x1)(1 - x2) * (z gate ,z gate )
#     # phi = ( + np.pi /2 * (1 - x_1)*(1 - x_2) * np.kron(z_m, z_m)  )
#
#     # # # pi * (cos x1)(cos x2) * (z gate ,z gate )
#     # phi = ( + np.pi * np.cos(x_1) * np.cos(x_2) * np.kron(z_m, z_m)   )
#
#     # phi = ( + np.pi * np.sin(x_1) * np.cos(x_2) * np.kron(z_m, z_m)   )
#
#     # phi = ( + np.pi * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m)  )
#
#     # phi = (+ np.pi * np.tan(x_1) * np.tan(x_2) * np.kron(z_m, z_m)   )
#
#     # phi = (+ np.pi * np.tan(x_1) * np.sin(x_2) * np.kron(z_m, z_m)  )
#
#     # phi = (+ np.pi * np.cos(x_1) / np.cos(x_2) * np.kron(z_m, z_m)  )
#
#     # phi = ( + np.pi/2 * np.cos(x_1) / np.cos(x_2) * np.kron(z_m, z_m))
#
#     # phi = ( + 0.5 * x_1 + 0.5 * x_2 * np.kron(z_m, z_m))
#
#     # phi = ( + np.pi * x_1 + np.pi * x_2 * np.kron(z_m, z_m))
#
#     # phi = (np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))
#
#     # phi = (100 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))
#
#     # phi = ( 10 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))
#
#     # phi = (0.1 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m)  )
#     # ### 2 种都要
#     # # x1 Z + x2 Z +  pi * (cos x1)(cos x2) * (z gate ,z gate )
#     # phi = ( + x_1 + x_2 + x_1 * x_2 * np.kron(z_m, z_m) )
#
#     # phi = ( x_1 * np.kron(z_m, j_m)  + x_2 * np.kron(j_m, z_m)  + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
#
#     return phi
# def phi_function3(x_1, x_2):
#     j_m = np.array([[1, 0], [0, 1]])   # 单位矩阵？
#     j_m = np.asarray(j_m)
#
#     s_z = np.array([[1, 0], [0, -1]])   # Z gate
#     z_m = np.asarray(s_z)
#
#     s_x = np.array([[0, 1], [1, 0]])   # X gate
#     x_m = np.asarray(s_x)
#
#     h_m = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # H GATE
#     h_2 = np.kron(h_m, h_m) # 2 个 H
#     h_m = np.asarray(h_m)
#     h_2 = np.asarray(h_2)
#
#     phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
#     # ########################
#     # # 只要x1 x2 单独的。-------------
#     phi = ( x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m))
#
#     return phi

j_m = np.array([[1, 0], [0, 1]])  # 单位矩阵？
j_m = np.asarray(j_m)

s_z = np.array([[1, 0], [0, -1]])  # Z gate
z_m = np.asarray(s_z)

s_x = np.array([[0, 1], [1, 0]])  # X gate
x_m = np.asarray(s_x)

h_m = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # H GATE
h_2 = np.kron(h_m, h_m)  # 2 个 H
h_m = np.asarray(h_m)
h_2 = np.asarray(h_2)

Paulis ={
    "Z1": np.kron(z_m, j_m),
    "Z2": np.kron(j_m, z_m),
    "ZZ": np.kron(z_m, z_m),
    "X1": np.kron(x_m, j_m),
    "X2": np.kron(j_m, x_m),
    "XX": np.kron(x_m, x_m),
    "H1": np.kron(h_m, j_m), # 单 H gate, H gate 位于第1位
    "H2": np.kron(j_m, h_m), # 单 H gate, H gate 位于第2位
    "HH": np.kron(h_m, h_m), # 双 H gate
}

def func_self(x_1):
    return x_1

# def func_multiple(a, x_1):
#     return a * x_1
#
# def func_add(a, x_1):
#     return a + x_1

def func_sin(x_1):
    return np.sin(x_1)

def func_cos(x_1):
    return np.cos(x_1)

def func_tan(x_1):
    return np.tan(x_1)

def func_zero(x_1):
    return 0

BASIC = {
    # "zero": func_zero,
    "self": func_self,
    # "mul": func_multiple,
    # "add": func_add,
    "sin": func_sin,
    "cos": func_cos,
    "tan": func_tan,
}

reps = [1, 2, 3, 4, 5]
alphas = [0.1, 1, 2, 3, 5, 10]
# paulis_signle = ["X", "Y", "Z", "I"]
# paulis_double = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"] # 如果有I 的话，其实就是signle 门，这样其实就是repeat？

# def phi1(x1, x2)-> float:
#     '''
#     :return:  x1[Z] + x2[Z] + (pi-x1)(pi-x2)[ZZ]
#     '''
#     return BASIC["slf"](x1) * Paulis["Z1"] + BASIC["slf"](x2) * Paulis["Z2"] + BASIC["add"](np.pi, -BASIC["slf"](x1))
# def phi2(x1, x2)-> float:
#     '''
#     :return: x1[Z] + x2[Z]
#     '''
#     return BASIC["slf"](x1) * Paulis["Z1"] + BASIC["slf"](x2) * Paulis["Z2"]
# # def phi3(x1, x2)-> float:
# #     '''
# #     :return: pi x1[Z] + pi x2[Z]
# #     '''
# #     return BASIC["mul"](x1, np.pi) * Paulis["Z1"] + BASIC["mul"](x2, np.pi) * Paulis["Z2"]
# def phi4(x1, x2)-> float:
#     '''
#     :return: (pi-x1)(pi-x2)[ZZ]
#     '''
#     return BASIC['add'](np.pi, -BASIC["slf"](x1)) * BASIC['add'](np.pi, -BASIC["slf"](x2)) * Paulis["ZZ"]
# def phi5(x1, x2)-> float:
#     '''
#     :return: (1-x1)(1-x2)[ZZ]
#     '''
#     return BASIC['add'](1, -BASIC["slf"](x1)) * BASIC['add'](1, -BASIC["slf"](x2)) * Paulis["ZZ"]
# def phi6(x1, x2)-> float:
#     '''
#     :return: cos(x1)cos(x2)[ZZ]
#     '''
#     return BASIC['cos'](x1) * BASIC['cos'](x2) * Paulis["ZZ"]
# def phi7(x1, x2)-> float:
#     '''
#     :return: sin(x1)cos(x2)[ZZ]
#     '''
#     return BASIC['sin'](x1) * BASIC['cos'](x2) * Paulis["ZZ"]
# def phi8(x1, x2)-> float:
#     '''
#     :return: sin(x1)sin(x2)[ZZ]
#     '''
#     return BASIC['sin'](x1) * BASIC['sin'](x2) * Paulis["ZZ"]

O_SET = {'sin', 'cos', 'tan', 'self'}
S_SET = {"X", "Y", "Z", "I"}
D_SET = {"XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"}

def auto_phi(x1, x2, a1, a2, a3, a4, b1, b2, b3, b4, O, S, D) -> np.array:
    '''
    :param x1: feature1
    :param x2: feature2
    :param a1:
    :param a2:
    :param a3:
    :param b1:
    :param b2:
    :param b3:
    :param O: operation
    :param S: single gate
    :param D: double gate
    :return: phi.shape (4, 4) array
    #  a1 O(x1) + b1 + a2 O(x2) + b2 + (a3 O(x1) + b3)(a4 O(x2) + b4)
    '''
    phi = \
        (a1 * BASIC[O](x1) + b1) * Paulis[S] + \
        (a2 * BASIC[O](x2) + b2) * Paulis[S] + \
        (a3 * BASIC[O](x1) + b3) * (a4 * BASIC[O](x2) + b4) * Paulis[D]
    return phi

'''
def phi_function():
    phi_map = {}
    j_m = np.array([[1, 0], [0, 1]])   # 单位矩阵？
    j_m = np.asarray(j_m)

    s_z = np.array([[1, 0], [0, -1]])   # Z gate
    z_m = np.asarray(s_z)

    s_x = np.array([[0, 1], [1, 0]])   # X gate
    x_m = np.asarray(s_x)

    h_m = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # H GATE
    h_2 = np.kron(h_m, h_m) # 2 个 H
    h_m = np.asarray(h_m)
    h_2 = np.asarray(h_2)
    # （1）
    def phi_map1(x_1,x_2):
        phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
        return phi
    phi_map["map1"] = phi_map1
    # ########################
    # （2）
    # # 只要x1 x2 单独的。-------------
    def phi_map2(x_1,x_2):
        phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m))
        return phi
    phi_map["map2"] = phi_map2

    # # 只要 x1 x2  都是先z 后 i
    # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(z_m, j_m))

    #
    # phi = (np.pi * x_1 * np.kron(z_m, j_m)  + np.pi* x_2 * np.kron(z_m, j_m))

    #（3）gap 0.1
    def phi_map3(x_1,x_2):
        return (np.pi * x_1 * np.kron(z_m, j_m) + np.pi* x_2 * np.kron(j_m, z_m) )
    phi_map["map3"] = phi_map3

    # # 只要 x1 x2  都是先i 后 z
    # phi = (x_1 * np.kron(j_m, z_m)+ x_2 * np.kron(j_m, z_m)  )

    # # X gate 一样
    # phi = ( x_1 * np.kron(j_m, x_m)  + x_2 * np.kron(x_m, j_m))

    # （4）gap 0.1
    # # 只要混合的 x1 x2 -----------
    def phi_map4(x_1, x_2):
        return ( + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m) )
    phi_map["map4"] = phi_map4

    # （5）gap 0.1
    # # pi/2 * (1- x1)(1 - x2) * (z gate ,z gate )
    def phi_map5(x_1,x_2):
        return ( + np.pi /2 * (1 - x_1)*(1 - x_2) * np.kron(z_m, z_m)  )
    phi_map["map5"] = phi_map5

    # （6）gap 0.1
    # # # pi * (cos x1)(cos x2) * (z gate ,z gate )
    def phi_map6(x_1,x_2):
        return ( + np.pi * np.cos(x_1) * np.cos(x_2) * np.kron(z_m, z_m)   )
    phi_map["map6"] = phi_map6


    # （7）gap 0.1    pi * (sin x1)(cos x2) * (z gate ,z gate )
    def phi_map7(x_1,x_2):
        return ( + np.pi * np.sin(x_1) * np.cos(x_2) * np.kron(z_m, z_m)   )
    phi_map["map7"] = phi_map7

    # phi = ( + np.pi * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m)  )

    # phi = (+ np.pi * np.tan(x_1) * np.tan(x_2) * np.kron(z_m, z_m)   )

    # phi = (+ np.pi * np.tan(x_1) * np.sin(x_2) * np.kron(z_m, z_m)  )

    # phi = (+ np.pi * np.cos(x_1) / np.cos(x_2) * np.kron(z_m, z_m)  )

    # phi = ( + np.pi/2 * np.cos(x_1) / np.cos(x_2) * np.kron(z_m, z_m))

    # phi = ( + 0.5 * x_1 + 0.5 * x_2 * np.kron(z_m, z_m))

    # phi = ( + np.pi * x_1 + np.pi * x_2 * np.kron(z_m, z_m))

    # phi = (np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))

    # phi = (100 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))
    # (8)
    def phi_map8(x_1,x_2):
        return ( 10 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))
    phi_map["map8"] = phi_map8

    # ### 2 种都要
    # (9)
    def phi_map9(x_1, x_2):
        return (x_1 + x_2 + x_1 * x_2 * np.kron(z_m, z_m) )
    phi_map["map9"] = phi_map9

    # # (10)
    # def phi_map10(x_1,x_2):
    #     return ( x_1 * np.kron(z_m, j_m)  + x_2 * np.kron(j_m, z_m)  + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
    # phi_map["map10"] = phi_map10

    return phi_map
phi_maps = phi_function()
'''

# class AutoDesignQC():
#     def __init__(self):
#         return
#
#     def fit(self, X_train, y_train):
#         # 准备阶段
#         parity = [ 1, -1, -1, 1]
#         n = 2
#         d_m = np.diag(parity)  # 对角矩阵
#         basis = algorithm_globals.random.random((2 ** n, 2 ** n)) + \
#                 1j * algorithm_globals.random.random((2 ** n, 2 ** n))
#         # 可以多次 选择。
#         basis = np.asarray(basis).conj().T @ np.asarray(basis)  # basis 的 共轭转置 与 basis 每一项 都相乘
#
#         [s_a, u_a] = np.linalg.eig(
#             basis)  # 不是e指数！！。而是 Compute the eigenvalues and right eigenvectors of a square array.
#         idx = s_a.argsort()[::-1]  # eigenvalues 从大到小 排列 的index
#         s_a = s_a[idx]
#         u_a = u_a[:, idx]  # u_a： 就是 V: a random  unitary V,  BELONG TO SU(4) .  eigenvectors 内层按  idx 排列，外层不变 文章里所谓的 V
#         m_m = (np.asarray(u_a)).conj().T @ np.asarray(d_m) @ np.asarray(u_a)  # u_a共轭转置 * 对角矩阵 * u_a
#         psi_plus = np.transpose(np.ones(2)) / np.sqrt(2)
#         psi_0 = [[1]]
#         for k in range(n):
#             psi_0 = np.kron(np.asarray(psi_0), np.asarray(psi_plus))
#
#         # 训练阶段。训练的参数是 a 和 b
#         score =0
#         for i in range(len(X_train)):
#             x1, x2 = X_train[i][0], X_train[i][1]
#             y = y_train[i]
#             phi = auto_phi(x1, x2, a1, a2, a3, a4, b1, b2, b3, b4, O, S, D)  # 调用 某一个 phi_map1 函数
#             # U_phi(x) feature map 的 i 及 矩阵实现
#             u_u = scipy.linalg.expm(1j * phi)  # pylint: disable=no-member  计算 e^A  的值，其中A为一个任意（应该是任意？）维数的方阵。
#
#             ######### 重复次数 真正的feature map #####################################################################################################################
#             # 如： 重复2 次 -- 真正的feature map： ZZ feature map 实现。 U_phi(x) H^2 U_phi(x) H^2 ｜0> ^n
#             psi = np.transpose(psi_0)
#             for r in range(repeat):
#                 psi = np.asarray(u_u) @ h_2 @ psi
#
#             ###### expection. ##################################################################################################################################################################
#             #  m_m 是一种operator/hibert space ： real of 《psi｜ m_m ｜psi》  temp 是 期望。 为什么取 real？
#             temp = np.real(
#                 psi.conj().T @ m_m @ psi).item()  # 为什么取 real值 ？因为只有   operator是 hermintion 时，特征值才是real，导致期望是real。当然，这里operator应该是real
#             ########################################################################################################################################################################
#
#             if abs(temp) >= 0.3:
#                 # s += abs(temp * y)
#                 score += abs(np.sign(temp * y))
#
#         score = score / len(y_train)
#         return self

def mapSelectMethod(dataset, a1, a2, a3, a4, b1, b2, b3, b4, O, S, D, repeat) -> float:
    n = 2
    count = 100
    steps = 2 * np.pi / count  # 2 pi / 6  每次旋转 steps 弧度？ 分成 6份。
    # print("steps", steps)        # steps 1.0471975511965976
    s_z = np.array([[1, 0], [0, -1]])  # Z gate
    z_m = np.asarray(s_z)
    j_m = np.array([[1, 0], [0, 1]])  # 单位矩阵？
    j_m = np.asarray(j_m)
    h_m = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # H GATE
    h_2 = np.kron(h_m, h_m)
    h_3 = np.kron(h_m, h_2)
    h_m = np.asarray(h_m)
    h_2 = np.asarray(h_2)
    h_3 = np.asarray(h_3)
    f_a = np.arange(2 ** n)  # 辅助用吧 ， my_array 外层个数和 f_a 一样
    my_array = [[0 for _ in range(n)] for _ in range(2 ** n)]  # my_array 外层个数和 f_a 一样
    for arindex, _ in enumerate(my_array):  # my_array 外层循环。
        temp_f = bin(f_a[arindex])[2:].zfill(
            n)  # returns the binary equivalent string of a given integer.  adds zeros (0) at the beginning of the string, until it reaches the specified length.
        for findex in range(n):  # my_array 内层循环
            my_array[arindex][findex] = int(temp_f[findex])

    my_array = np.asarray(my_array)
    my_array = np.transpose(my_array)

    # Define decision functions
    maj = (-1) ** (2 * my_array.sum(axis=0) > n)  # n =3 才用到
    parity = (-1) ** (my_array.sum(axis=0))  # parity奇偶性 [ 1 -1 -1  1],    -1 的 (0, 1,1,2) 次方  n =2 用到

    # d_m： 就是 f: parity  function  d_m 是 zz 吧， as  the  parity  function
    d_m = None
    if n == 2:
        d_m = np.diag(parity)  # 对角矩阵
    elif n == 3:
        d_m = np.diag(maj)

    # algorithm_globals.random: "Return a numpy np.random.Generator
    # seed = 12345
    # algorithm_globals.random_seed = seed

    ######### 创造测量 期望 矩阵 #####################################################################################################################
    basis = algorithm_globals.random.random((2 ** n, 2 ** n)) + \
            1j * algorithm_globals.random.random((2 ** n, 2 ** n))
    # 可以多次 选择。
    basis = np.asarray(basis).conj().T @ np.asarray(basis)  # basis 的 共轭转置 与 basis 每一项 都相乘

    [s_a, u_a] = np.linalg.eig(basis)  # 不是e指数！！。而是 Compute the eigenvalues and right eigenvectors of a square array.
    idx = s_a.argsort()[::-1]  # eigenvalues 从大到小 排列 的index
    s_a = s_a[idx]
    u_a = u_a[:, idx]  # u_a： 就是 V: a random  unitary V,  BELONG TO SU(4) .  eigenvectors 内层按  idx 排列，外层不变 文章里所谓的 V
    m_m = (np.asarray(u_a)).conj().T @ np.asarray(d_m) @ np.asarray(u_a)  # u_a共轭转置 * 对角矩阵 * u_a

    psi_plus = np.transpose(np.ones(2)) / np.sqrt(2)
    # print("psi_plus", psi_plus)   #  [0.70710678 0.70710678]


    psi_0 = [[1]]
    for k in range(n):
        psi_0 = np.kron(np.asarray(psi_0), np.asarray(psi_plus))
    all_temp = []  ##


    score = 0
    X_train = dataset[0]
    y_train = dataset[1]

    for i in range(len(X_train)):
        x1, x2 = X_train[i][0], X_train[i][1]
        y = y_train[i]
        phi = auto_phi(x1, x2, a1, a2, a3, a4, b1, b2, b3, b4, O, S, D)  # 调用 某一个 phi_map1 函数
        # U_phi(x) feature map 的 i 及 矩阵实现
        u_u = scipy.linalg.expm(1j * phi)  # pylint: disable=no-member  计算 e^A  的值，其中A为一个任意（应该是任意？）维数的方阵。

        ######### 重复次数 真正的feature map #####################################################################################################################
        # 如： 重复2 次 -- 真正的feature map： ZZ feature map 实现。 U_phi(x) H^2 U_phi(x) H^2 ｜0> ^n
        psi = np.transpose(psi_0)
        for r in range(repeat):
            psi = np.asarray(u_u) @ h_2 @ psi

        ###### expection. ##################################################################################################################################################################
        #  m_m 是一种operator/hibert space ： real of 《psi｜ m_m ｜psi》  temp 是 期望。 为什么取 real？
        temp = np.real(
            psi.conj().T @ m_m @ psi).item()  # 为什么取 real值 ？因为只有   operator是 hermintion 时，特征值才是real，导致期望是real。当然，这里operator应该是real
        # print("temp", temp)
        all_temp.append(temp)
        ########################################################################################################################################################################

        if abs(temp) >= 0.3:
            # s += abs(temp * y)
            score += abs(np.sign(temp * y))

    score = score / len(y_train)
    print("score is", score)
    # all_scores.append(s)


param_distributions ={
    "O": ["sin", 'cos', 'tan', 'self'],
    "S": ["X", "Y", "Z", "I"],
    "D": ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"],
    "alpha": uniform(loc=-100, scale=200),
    "repeat": [1, 2, 3, 4, 5]
}

clf = RandomizedSearchCV(
    mapSelectMethod(kernel="rbf", class_weight="balanced"), param_distributions, n_iter=100
)



def run():
    # 产生data
    # repeat = 2
    # X_train, y_train, X_test, y_test, sample_total = ad_hoc_data(
    #     training_size = 25,
    #     test_size=10,
    #     n=2,
    #     gap=0.3,
    #     repeat=repeat,
    #     plot_data=True,
    #     one_hot=False,
    #     include_sample_total=True,
    # )

    # data example: #########
    train = pd.read_csv('./quantum_circuit_designed_dataset/phi1_repeat2_train.csv')
    y_train, X_train_0, X_train_1 = np.array(train["y_train"]), train["feature1"], train["feature2"]
    X_train = np.array([X_train_0, X_train_1]).T

    test = pd.read_csv('./quantum_circuit_designed_dataset/phi1_repeat2_test.csv')
    y_test, X_test_0, X_test_1 = np.array(test["y_test"]), test["feature1"], test["feature2"]
    X_test = np.array([X_test_0, X_test_1]).T

    y_train = [-1 if i == 0 else 1 for i in y_train]
    y_test = [-1 if i == 0 else 1 for i in y_test]

    dataset = (X_train, y_train)
    print("y_train", y_train)
    print("y_test", y_test)
    ######### ######### ######### ######### #########

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(X_train.T[0], X_train.T[1], c=y_train, cmap=plt.cm.coolwarm)
    ax1.scatter(X_test.T[0], X_test.T[1], marker='x', c=y_test, cmap=plt.cm.coolwarm)
    plt.show()

    mapSelectMethod(dataset, repeat=2)

    # # 选择map according rep
    # for reps in [1, 2, 3, 4, 5, 6]:
    #     print("======")
    #     print("reps is", reps)
    #     mapSelectMethod(dataset, repeat=reps)
    #     s_train, s_test, f1_train, f1_test = run_Q_Kernel_method(reps, X_train, y_train, X_test, y_test)
    #     print("s_train", s_train)
    #     print("s_test", s_test)

    # print("======")
    # reps = 2
    # mapSelectMethod(dataset, repeat=reps)
    # s_train, s_test, f1_train, f1_test = run_Q_Kernel_method(reps, X_train, y_train, X_test, y_test)
    # print("s_train", s_train)
    # print("s_test", s_test)
    #
    # print("======")
    # reps = 3
    # mapSelectMethod(dataset, repeat=reps)
    # s_train, s_test, f1_train, f1_test = run_Q_Kernel_method(reps, X_train, y_train, X_test, y_test)
    # print("s_train", s_train)
    # print("s_test", s_test)
    #
    # print("======")
    # reps = 4
    # mapSelectMethod(dataset, repeat=reps)
    # s_train, s_test, f1_train, f1_test = run_Q_Kernel_method(reps, X_train, y_train, X_test, y_test)
    # print("s_train", s_train)
    # print("s_test", s_test)
    #
    # reps = 5
    # mapSelectMethod(dataset, repeat=reps)
    # s_train, s_test, f1_train, f1_test = run_Q_Kernel_method(reps, X_train, y_train, X_test, y_test)
    # print("s_train", s_train)
    # print("s_test", s_test)
    #
    # reps = 6
    # mapSelectMethod(dataset, repeat=reps)
    # s_train, s_test, f1_train, f1_test = run_Q_Kernel_method(reps, X_train, y_train, X_test, y_test)
    # print("s_train", s_train)
    # print("s_test", s_test)

run()



