from sklearn.svm import SVC
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap,ZFeatureMap,PauliFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
seed = 1996
from functools import reduce

# 原来的 Q_Kernel_method 类
# class Q_Kernel_method(SVC):
#     # super(SVC, self).__init__()
#     def __init__(self, reps, paulis, func):
#         # self.Q_feature_map = ZZFeatureMap(feature_dimension=2,
#         #                                   reps=2,
#         #                                   entanglement='linear')
#         self.Q_feature_map = PauliFeatureMap(feature_dimension=2,
#                                              reps=reps,
#                                              paulis=paulis,
#                                              data_map_func = func
#                                              )
#
#         self.Q_backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
#                                          shots=1024,
#                                          seed_simulator=seed,
#                                          seed_transpiler=seed)
#
#         # print(nlocal)
#         self.svc = SVC(kernel='precomputed')
#         self.Q_kernel = QuantumKernel(feature_map=self.Q_feature_map, quantum_instance=self.Q_backend)
#
#     ### 训练集： 先fit 再 score_train；
#     def fit(self, X, y, delta=None):
#         self.Q_matrix_train = self.Q_kernel.evaluate(x_vec=X)
#         return self.svc.fit(self.Q_matrix_train, y)
#
#     def score_train(self, y_train):
#         s = self.svc.score(self.Q_matrix_train, y_train)
#         return s
#
#     def predict_train(self, ):
#         pred_y = self.svc.predict(self.Q_matrix_train)
#         return pred_y
#
#     # 测试集： 先 cal_test_matrix，再 score_test or predict_test
#     def cal_test_matrix(self, X_test, X_train):
#         self.Q_matrix_test = self.Q_kernel.evaluate(x_vec=X_test, y_vec=X_train)  #### 原来是 x_vec=X_test, y_vec=X_train
#         return
#
#     def score_test(self, y_test):
#         s = self.svc.score(self.Q_matrix_test, y_test)
#         return s
#
#     def predict_test(self, ):
#         pred_y = self.svc.predict(self.Q_matrix_test)
#         return pred_y
#
#     def predict_proba(self, X):
#         return self.svc.predict_proba(X)



# # 原来的
# def run_Q_Kernel_method(X_train, y_train, X_test, y_test, reps, paulis, func):
#     clf = Q_Kernel_method(reps, paulis, func)
#     clf.fit(X_train, y_train)
#     pre_y_train = clf.predict_train()        # fit 时，已经知道 trian_matrix 了，不需要输入了
#     s_train = clf.score_train(y_train)       # 普通score
#     f1_train = f1_score(y_train, pre_y_train)# F1 score
#
#     # 测试集操作：
#     clf.cal_test_matrix(X_test, X_train)
#     pre_y_test = clf.predict_test()          # 已经知道 cal_test_matrix 了，不需要输入了
#     s_test = clf.score_test(y_test)          # 普通score
#     f1_test = f1_score(y_test, pre_y_test)   # F1 score
#
#     print("s_test",s_test)
#
#     return s_train, s_test, f1_train, f1_test
#

class Q_Kernel_method(SVC):
    # super(SVC, self).__init__()
    def __init__(self, reps, alpha, paulis, func, kernel):
        # super(Q_Kernel_method, self).__init__(reps, paulis, func)
        SVC.__init__(self, kernel=kernel)
        # self.Q_feature_map = ZZFeatureMap(feature_dimension=2,
        #                                   reps=2,
        #                                   entanglement='linear')
        self.Q_feature_map = PauliFeatureMap(feature_dimension=2,
                                             reps = reps,
                                             alpha = alpha,
                                             paulis = paulis,
                                             data_map_func = func
                                             )
        self.Q_backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                         shots=1024,
                                         seed_simulator=seed,
                                         seed_transpiler=seed)

        # self.svc = SVC(kernel='precomputed')
        self.Q_kernel = QuantumKernel(feature_map = self.Q_feature_map, quantum_instance=self.Q_backend)

def run_Q_Kernel_method(X_train, y_train, X_test, y_test, reps, alpha, paulis, func):
    clf = Q_Kernel_method(reps, alpha, paulis, func, kernel='precomputed')

    ###### train #######
    # calculate kernel matrix of train
    Q_matrix_train = clf.Q_kernel.evaluate(x_vec=X_train)
    #  fit
    clf.fit(Q_matrix_train, y_train)
    # predict train
    pre_y_train = clf.predict(Q_matrix_train)     # fit 时，已经知道 trian_matrix 了，不需要输入了
    s_train = accuracy_score(pre_y_train, y_train)  # 普通score
    f1_train = f1_score(pre_y_train,y_train)     # F1 score

    ###### test #######
    # calculate kernel matrix of test
    Q_matrix_test = clf.Q_kernel.evaluate(x_vec=X_test, y_vec=X_train)
    # predict test
    pre_y_test = clf.predict(Q_matrix_test)
    s_test = accuracy_score(pre_y_test, y_test)
    f1_test = f1_score(pre_y_test, y_test, )
    print(s_test)
    return (s_train, f1_train, s_test, f1_test)

def func_1(x: np.ndarray) -> float:
    '''
    # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
    # 𝜙(xi) = xi ,    𝜙(x1, x2) = (pi - x1) * (pi - x2)
    '''
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, np.pi - x)
    return coeff

def func_2(x: np.ndarray) -> float:
    '''
    phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m))
    𝜙(xi) = xi
    '''
    coeff = x[0]
    return coeff

def func_3(x: np.ndarray) -> float:
    '''
    phi = (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m)
    𝜙(xi,x2) = (pi - x1) (pi - x2)
    '''
    coeff = reduce(lambda m, n: m * n, np.pi - x)
    return coeff

def func_4(x: np.ndarray) -> float:
    '''
    phi = (1 - x_1)*(1 - x_2) * np.kron(z_m, z_m))
    𝜙(xi,x2) = (1 - x1) (1 - x2)
    '''
    coeff = reduce(lambda m, n: m * n, 1 - x)
    return coeff

def func_5(x: np.ndarray) -> float:
    '''
    phi = np.cos(x_1) * np.cos(x_2) * np.kron(z_m, z_m)
    𝜙(xi,x2) = cos(x1) cos(x2)
    '''
    coeff = reduce(lambda m, n: m * n, np.cos(x))
    return coeff

def func_6(x: np.ndarray) -> float:
    '''
    phi =  np.sin(x_1) * np.cos(x_2) * np.kron(z_m, z_m)
    𝜙(xi,x2) = sin(x1) cos(x2)
    '''
    coeff = reduce(lambda m, n: np.sin(m) * np.cos(n), x)
    return coeff

def func_7(x: np.ndarray) -> float:
    '''
    phi =  np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))
    𝜙(xi,x2) = sin(x1) sin(x2)
    '''
    coeff = reduce(lambda m, n: np.sin(m) * np.sin(n), x)
    return coeff

def func_8(x: np.ndarray) -> float:
    '''
    phi =   x_1 + x_2 + x_1 * x_2 * np.kron(z_m, z_m)
    𝜙(xi) = xi,   𝜙(x1,x2) = x1 x2
    '''
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, x)
    return coeff


'''
# data example:
train = pd.read_csv('./quantum_circuit_designed_dataset/phi1_repeat2_train.csv')
y_train, X_train_0, X_train_1 = np.array(train["y_train"]), train["feature1"], train["feature2"]
X_train = np.array([X_train_0, X_train_1]).T

test = pd.read_csv('./quantum_circuit_designed_dataset/phi1_repeat2_test.csv')
y_test, X_test_0, X_test_1 = np.array(test["y_test"]), test["feature1"], test["feature2"]
X_test = np.array([X_test_0, X_test_1]).T

y_train = [-1 if i == 0 else 1 for i in y_train]
y_test = [-1 if i == 0 else 1 for i in y_test]

dataset = (X_train, y_train)
print(X_train)
print("y_train", y_train)
print("y_test", y_test)


setting1 = {"rep": 2, "paulis": ['Z','ZZ'], "func": func_1, 'alpha':2}
setting2 = {"rep": 2, "paulis": ['Z'], "func": func_2, 'alpha': 2}
setting3 = {"rep": 2, "paulis": ['Z'], "func": func_2, 'alpha': 2 * np.pi}
setting4 = {"rep": 2, "paulis": ['ZZ'], "func": func_3, 'alpha': 2}
setting5 = {"rep": 2, "paulis": ['ZZ'], "func": func_4, 'alpha': 2 * np.pi /2}
setting6 = {"rep": 2, "paulis": ['ZZ'], "func": func_5, 'alpha': 2 * np.pi}
setting7 = {"rep": 2, "paulis": ['ZZ'], "func": func_6, 'alpha': 2 * np.pi}
setting8 = {"rep": 2, "paulis": ['ZZ'], "func": func_7, 'alpha': 2 * 10}
setting9 = {"rep": 2, "paulis": ['ZZ'], "func": func_8, 'alpha': 2 * 2 * 8.5}
'''

# setting = setting2
# run_Q_Kernel_method(X_train, y_train, X_test, y_test,
#                     reps=setting["rep"],
#                     alpha=setting["alpha"],
#                     paulis=setting["paulis"],
#                     func=setting["func"])



# 自测：
# train = pd.read_csv('./quantum_circuit_designed_dataset/TEST_phi1_repeat1_train.csv')
# y_train, X_train_0, X_train_1 = np.array(train["y_train"]), train["feature1"], train["feature2"]
# X_train = np.array([X_train_0, X_train_1]).T
#
# test = pd.read_csv('./quantum_circuit_designed_dataset/TEST_phi1_repeat1_test.csv')
# y_test, X_test_0, X_test_1 = np.array(test["y_test"]), test["feature1"], test["feature2"]
# X_test = np.array([X_test_0, X_test_1]).T

# w1 = 1.0488137817112642
# w2 = -1.161519399339819
# b1 = 0.6909123022594433
# b2 = 2.3061988506302593

# w1, w2, b1, b2 = 1, 0, -1, np.pi
# def phi_function_general(x):
#     '''
#     # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
#     # 𝜙(xi) = xi ,    𝜙(x1, x2) = (pi - x1) * (pi - x2)
#     '''
#     coeff = (w1 * x[0] + b1) if len(x) == 1 else reduce(lambda m, n: m * n, w2 * x + b2)
#     return coeff
#
# setting = {"rep": 1, "paulis": ['Z', 'ZZ'], "func": phi_function_general, 'alpha': 2}
# setting = setting
# run_Q_Kernel_method(X_train, y_train, X_test, y_test,
#                     reps=setting["rep"],
#                     alpha=setting["alpha"],
#                     paulis=setting["paulis"],
#                     func=setting["func"])