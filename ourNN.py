import numpy as np
import scipy
from qiskit.utils import algorithm_globals
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
h_m = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # H GATE
h_2 = np.kron(h_m, h_m)
h_2 = np.asarray(h_2)

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

Paulis = {
    "Z1": np.kron(z_m, j_m),
    "Z2": np.kron(j_m, z_m),
    "ZZ": np.kron(z_m, z_m),
    "X1": np.kron(x_m, j_m),
    "X2": np.kron(j_m, x_m),
    "XX": np.kron(x_m, x_m),
    "H1": np.kron(h_m, j_m),  # 单 H gate, H gate 位于第1位
    "H2": np.kron(j_m, h_m),  # 单 H gate, H gate 位于第2位
    "HH": np.kron(h_m, h_m),  # 双 H gate
}


n = 2       # 2 dimension
repeat = 1  # repeat times
gap = 0

f_a = np.arange(2 ** n)
my_array = [[0 for _ in range(n)] for _ in range(2 ** n)]  # my_array 外层个数和 f_a 一样
for arindex, _ in enumerate(my_array):  # my_array 外层循环。
  temp_f = bin(f_a[arindex])[2:].zfill(
    n)  # returns the binary equivalent string of a given integer.  adds zeros (0) at the beginning of the string, until it reaches the specified length.
  for findex in range(n):  # my_array 内层循环
    my_array[arindex][findex] = int(temp_f[findex])

my_array = np.asarray(my_array)
my_array = np.transpose(my_array)

# 定义 f
parity = (-1) ** (my_array.sum(axis=0))  # parity奇偶性 [ 1 -1 -1  1],    -1 的 (0, 1,1,2) 次方  n =2 用到
d_m = np.diag(parity)
print("d_m", d_m)

# seed = 12345 #可以
seed = 12345
algorithm_globals.random_seed = seed
basis = algorithm_globals.random.random((2 ** n, 2 ** n)) + \
        1j * algorithm_globals.random.random((2 ** n, 2 ** n))

# 可以多次 选择。
basis = np.asarray(basis).conj().T @ np.asarray(basis)  # basis 的 共轭转置 与 basis 每一项 都相乘
[s_a, u_a] = np.linalg.eig(basis)  # 不是e指数！！。而是 Compute the eigenvalues and right eigenvectors of a square array.
idx = s_a.argsort()[::-1]  # eigenvalues 从大到小 排列 的index
s_a = s_a[idx]
u_a = u_a[:, idx]  # u_a： 就是 V: a random  unitary V,  BELONG TO SU(4) .  eigenvectors 内层按  idx 排列，外层不变 文章里所谓的 V
m_m = (np.asarray(u_a)).conj().T @ np.asarray(d_m) @ np.asarray(u_a)
print("m_m:", m_m)


psi_plus = np.transpose(np.ones(2)) / np.sqrt(2)
psi_0 = [[1]]
for k in range(n):
  psi_0 = np.kron(np.asarray(psi_0), np.asarray(psi_plus))

# print("psi_0",psi_0)
zero2 = np.array([[1, 0, 0, 0]]).T

def d_pre_d_A(A, H_2, M, zero_2):  #
  t_0 = (H_2).dot(zero_2)
  t_1 = (A).dot(t_0)
  t_2 = (M).dot(t_1)
  gradient = (np.outer(t_2, t_0) + np.outer((M.T).dot(t_1), t_0))
  return gradient

def d_A_d_phi_real(phi):
  t_0 = (1 / 2)
  gradient = -((t_0 * np.einsum('ik, jl', np.eye(4, 4), phi.T)) + (t_0 * np.einsum('ik, jl', phi, np.eye(4, 4))))
  # print(gradient.shape)
  return gradient

def d_A_d_phi_im(phi):
  t_0 = (1 / 6)
  T_1 = (phi).dot(phi)
  T_2 = (T_1).dot(phi)
  gradient = -((((t_0 * np.einsum('ik, jl', np.eye(4, 4), T_2.T)) + (t_0 * np.einsum('ik, jl', phi, T_1.T))) + (
            t_0 * np.einsum('ik, jl', T_1, phi.T))) + (t_0 * np.einsum('ik, jl', T_2, np.eye(4, 4))))
  # print(gradient.shape)
  return gradient

def complex_array_to_real_array(C) -> np.array:
  '''
  :param C = A + Bj:
  :return: Y = [[A, -B],[B, A]]
  '''
  real = np.real(C)
  imag = np.imag(C)

  row1 = list(real[0]) + list(-imag[0])
  row2 = list(real[1]) + list(-imag[1])
  row3 = list(real[2]) + list(-imag[2])
  row4 = list(real[3]) + list(-imag[3])

  row5 = list(imag[0]) + list(real[0])
  row6 = list(imag[1]) + list(real[1])
  row7 = list(imag[2]) + list(real[2])
  row8 = list(imag[3]) + list(real[3])

  res = np.array([row1, row2, row3, row4, row5, row6, row7, row8])
  # print("res", res.shape)
  return res

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self, learn_rate, epochs):
    self.learn_rate = learn_rate
    self.epochs = epochs

    self.w1 = np.random.normal() # 1
    self.w2 = np.random.normal() # 1
    self.b1 = np.random.normal() # 0
    self.b2 = np.random.normal() # 0

    # self.w1 = 1
    # self.w2 = -1
    # self.b1 = 0
    # self.b2 = np.pi

    # Weights
    # self.w1 = 1 #np.random.normal() #1
    # self.w2 = 1 #np.random.normal()# 1
    # self.w3 = 1 #np.random.normal() #  -1 #
    # self.w4 = 1 #np.random.normal() # -1
    # Biases
    # self.b1 = 1 #np.random.normal() # 0
    # self.b2 = 1 #np.random.normal() # 0
    # self.b3 = 1 #np.random.normal() # np.pi
    # self.b4 = 1 #np.random.normal() #np.pi#

    # Weights
    # self.w1 = 1#np.random.normal() #1
    # self.w2 = 1#np.random.normal()# 1
    # self.w3 = -1#np.random.normal() #  -1 #
    # self.w4 = -1#np.random.normal() # -1
    # # Biases
    # self.b1 = 0#np.random.normal() # 0
    # self.b2 = 0#np.random.normal() # 0
    # self.b3 = np.pi#np.random.normal() # np.pi
    # self.b4 = np.pi#np.random.normal() #np.pi#

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    # phi = (self.w1 * x[0] + self.b1) * Paulis['Z1'] + \
    #       (self.w2 * x[1] + self.b2) * Paulis['Z2'] + \
    #       (self.w3 * x[0] + self.b3) * (self.w4 * x[1] + self.b4) * Paulis['ZZ']

    phi = (self.w1 * x[0] + self.b1) * Paulis['Z1'] + \
          (self.w1 * x[1] + self.b1) * Paulis['Z2'] + \
          (self.w2 * x[0] + self.b2) * (self.w2 * x[1] + self.b2) * Paulis['ZZ']

    A_out = scipy.linalg.expm(1j * phi)
    psi_0 = np.array([[0.5, 0.5, 0.5, 0.5]])

    # print("A_out", A_out.shape)
    # print("A_out", A_out)
    # print("psi_0", psi_0)

    # todo repeat 1 次 now
    repeat = 1
    if repeat == 1:
      psi = np.asarray(A_out) @ np.transpose(psi_0)
    # print("psi", psi)
    # for r in range(repeat):
    #   psi = np.asarray(A_out) @ h_2 @ psi
    y_pred = np.real(psi.conj().T @ m_m @ psi).item()  # temp
    # print("y_pred feedforward",y_pred)

    gap = 0

    # y_pred = 1 if y_pred > gap else -1 if y_pred < -gap else 0
    # if y_pred > gap:
    #   y_pred = 1
    # elif y_pred < -gap:
    #   y_pred = -1
    # else:
    #   y_pred = 0

    # (1) 适用sign函数，导致梯度为0
    # y_pred = np.sign(y_pred)
    return y_pred

  def loss_function(self, y_true, y_pred):
    loss = nn.MSELoss(reduction="mean")
    output = loss(y_true, y_pred)
    return output

  def train(self, Train):
    myloader = DataLoader(dataset=Train, batch_size=10, shuffle=True, drop_last=False)
    losses_epochs_list = []
    self.epochs = 2000
    self.learn_rate = 0.01
    for j in range(self.epochs):    # 一个 epoch
      loss_one_epoch_list = []
      for batch_data in myloader:   # 一个 batch
        X_train_batch, y_train_batch = batch_data[0], batch_data[1]
        X_train_batch,  y_train_batch = X_train_batch.numpy(), y_train_batch.numpy()
        # forward process for every data point 中间过程变量都要用到，所以不能直接调用 self.forward
        # 问题：一个batch的数据，变量怎么存储？
        phi_batch = []
        A_out_batch = []
        y_pred_batch = []

        for i, X in enumerate(X_train_batch):
          phi = (self.w1 * X[0] + self.b1) * Paulis['Z1'] + \
                (self.w1 * X[1] + self.b1) * Paulis['Z2'] + \
                (self.w2 * X[0] + self.b2) * (self.w2 * X[1] + self.b2) * Paulis['ZZ']
          A_out = scipy.linalg.expm(1j * phi)

          psi_0 = np.array([[0.5, 0.5, 0.5, 0.5]])
          repeat = 1
          if repeat == 1:
            psi = np.asarray(A_out) @ np.transpose(psi_0)
          y_pred = np.real(psi.conj().T @ m_m @ psi).item()

          phi_batch.append(phi)
          A_out_batch.append(A_out)
          y_pred_batch.append(y_pred)

        phi_batch = np.array(phi_batch)
        A_out_batch = np.array(A_out_batch)
        y_pred_batch = np.array(y_pred_batch)

        # loss of a batch
        loss_batch = self.loss_function(torch.Tensor(y_train_batch), torch.Tensor(y_pred_batch))
        # print("loss_batch", loss_batch)
        loss_one_epoch_list.append(loss_batch)

        ### back propagation:
        # todo 计算  d(L)/d(ypred) -- mean # scalar
        d_L_d_ypred = 2 * (y_pred_batch - y_train_batch).mean()

        # todo 计算 d(y_pre)/d(A) -- mean # (4, 4)  # y_pre 是 scalar, A 是（4，4）
        A_out_batch_mean = A_out_batch.mean(axis=0)
        d_pre_d_A_res = d_pre_d_A(A=A_out_batch_mean, H_2=h_2, M=m_m, zero_2=zero2)

        # todo 计算 d(A)/d(phi)  -- mean : 级数展开，实虚部分开. 过大 # A 是（4，4）phi是 (4,4)
        # 实部
        phi = phi_batch.mean(axis=0)
        d_A_d_phi_real_res = d_A_d_phi_real(phi).astype(complex)   # (4,4,4)
        # d_A_d_phi_real_res = np.trace(d_A_d_phi_real_res,)       # option1: 求trace (4,4,4,4) -> (4,4)
        d_A_d_phi_real_res = d_A_d_phi_real_res.mean(axis=(0,1))   # option2: 求mean  (4,4,4,4) -> (4,4)
        # 虚部
        d_A_d_phi_im_res = d_A_d_phi_im(phi).astype(complex)       # (4,4,4)
        # d_A_d_phi_im_res = np.trace(d_A_d_phi_im_res)            # option1: 求trace (4,4,4,4) -> (4,4)
        d_A_d_phi_im_res = d_A_d_phi_im_res.mean(axis=(0,1))       # option2: 求mean (4,4,4,4) -> (4,4)
        # 实部 + 虚部  # (4,4)
        d_A_d_phi_res = d_A_d_phi_real_res + 1j * d_A_d_phi_im_res

        #todo 计算 d(phi)/d(w) and  d(phi)/d(b)   # (4,4)
        X_mean = X_train_batch.mean(axis=0)
        d_phi_d_w1 = X_mean[0] * Paulis['Z1'] + X_mean[1] * Paulis['Z2']
        d_phi_d_b1 = Paulis['Z1'] + Paulis['Z2']
        d_phi_d_w2 = (2 * X_mean[0] * X_mean[1] * self.w2 + self.b2 * (X_mean[0] + X_mean[1]) ) * Paulis['ZZ']
        d_phi_d_b2 = (self.w2 * (X_mean[0] + X_mean[1]) + 2 * self.b2) * Paulis['ZZ']

        # todo 合起来: 计算 d(L) for d(w)/d(b)  # (4, 4)
        # (2)  2个w 2个b 的phi
        d_L_d_w1 = np.array(d_L_d_ypred * d_pre_d_A_res * d_A_d_phi_res * d_phi_d_w1)
        d_L_d_w2 = np.array(d_L_d_ypred * d_pre_d_A_res * d_A_d_phi_res * d_phi_d_w2)
        d_L_d_b1 = np.array(d_L_d_ypred * d_pre_d_A_res * d_A_d_phi_res * d_phi_d_b1)
        d_L_d_b2 = np.array(d_L_d_ypred * d_pre_d_A_res * d_A_d_phi_res * d_phi_d_b2)
        # # 复数矩阵 --> 实数矩阵
        d_L_d_w1 = complex_array_to_real_array(d_L_d_w1)
        d_L_d_w2 = complex_array_to_real_array(d_L_d_w2)
        d_L_d_b1 = complex_array_to_real_array(d_L_d_b1)
        d_L_d_b2 = complex_array_to_real_array(d_L_d_b2)
        # # 矩阵的迹 实数
        d_L_d_w1 = np.trace(d_L_d_w1)
        d_L_d_w2 = np.trace(d_L_d_w2)
        d_L_d_b1 = np.trace(d_L_d_b1)
        d_L_d_b2 = np.trace(d_L_d_b2)

        # # # 实数的 np.tanh 避免爆炸
        # d_L_d_w1 = np.tanh(d_L_d_w1)
        # d_L_d_w2 = np.tanh(d_L_d_w2)
        # d_L_d_b1 = np.tanh(d_L_d_b1)
        # d_L_d_b2 = np.tanh(d_L_d_b2)
        # # update
        self.w1 -= self.learn_rate * d_L_d_w1
        self.w2 -= self.learn_rate * d_L_d_w2
        self.b1 -= self.learn_rate * d_L_d_b1
        self.b2 -= self.learn_rate * d_L_d_b2
      loss_one_epoch = np.array(loss_one_epoch_list).sum()
      if j % 50 == 0:
        print("Epoch {} Loss:{}".format(j, loss_one_epoch))
      losses_epochs_list.append(loss_one_epoch)

    print("losses_epochs_list", losses_epochs_list)

    # drawing:
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot([_+1 for _ in range(self.epochs)], losses_epochs_list)
    plt.show()
    return