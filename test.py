from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
import numpy as np
# todo 构建 ypred 关于 psi 表达式：  NO
'''
psi = MatrixSymbol('psi', 4, 1)
psi_dagger = Matrix(Dagger(psi))
m_m = Matrix(m_m)
print("psi.shape", psi.shape)
print("psi_dagger.shape", psi_dagger.shape)
print("m_m.shape", m_m.shape)

ypred = psi_dagger * m_m * psi
print("ypred", ypred)
print("shape ypred", shape(ypred))


# todo 构建 ypred 关于 A 表达式：
# zero2 = Matrix(np.array([[1, 0, 0, 0]]).T)
# print("zero2.shape", zero2.shape)
# A = MatrixSymbol('A', n ** 2, n ** 2)
# h_2 = Matrix(h_2)
# psi = A * h_2
#
# for r in range(repeat - 1):
#   psi = psi * psi
# psi = psi * zero2
# print("psi.shape", psi.shape)
# psi_dagger = Matrix(Dagger(psi))
# print("psi_dagger.shape", psi_dagger.shape)
# m_m = Matrix(m_m)
# ypred = psi_dagger * m_m * psi
# print("shape ypred", shape(ypred))
# print("ypred", ypred)

# todo 构建 ypred 关于 phi 的 表达式：
# phi = MatrixSymbol('phi', n ** 2, n ** 2)
# print("phi.shape", phi.shape)
# A = Matrix(eye(4)) + I * phi #- phi * phi * 0.5 # - I * (phi*phi*phi)/6   # 近似
# print("A", A)
# print("A.shape", A.shape)
#
# zero2 = Matrix(np.array([[1, 0, 0, 0]]).T)
# h_2 = Matrix(h_2)
# psi = A * h_2
# for r in range(repeat - 1):
#   psi = psi * psi
# psi = psi * zero2
# print("psi.shape", psi.shape)
# psi_dagger = Matrix(Dagger(psi))
# print("psi_dagger.shape", psi_dagger.shape)
# m_m = Matrix(m_m)
# ypred = psi_dagger * m_m * psi
# print("shape ypred", shape(ypred))
# print("ypred", ypred)
'''

#
# def complex_array_to_real_array(C)-> np.array:
#   real = np.real(C)
#   imag = np.imag(C)
#   print("real", real)
#   print("imag", imag)
#
#   row1 = list(real[0]) + list(-imag[0])
#   row2 = list(real[1]) + list(-imag[1])
#   row3 = list(imag[0]) + list(real[0])
#   row4 = list(imag[1]) + list(real[1])
#   res = np.array([row1, row2, row3, row4])
#   return res
#
# C = [[1, 1 + 9j], [2+1j, 2+2j]]
# print(complex_array_to_real_array(C))


# epochs = [1,2,3,4]
# losss =[1,2,3,4]
# fig = plt.figure(figsize=(5, 5))
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.plot(epochs, losss, )
# plt.show()


# y_true = [1, -1]
# y_pred = [1, -1]
#
# a = zero_one_loss(y_true, y_pred)
# print(a)



# fig = plt.figure()
# plt.subplot(1, 1, 1)
# names = ["before", "after"]
# width = 0.3
# acc = [0.6, 0.65]
# plt.bar(names, acc, width=width)
# plt.title("Before and After Parameter Optimization")
# plt.ylabel("Test Accuracy")
#
# plt.grid()
# plt.show()


'''
Question:
You are climbing a staircase. It takes n (for example, n =100) steps to reach the top.  
time complexity: O(n)
space complexity: O(n)
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

f(n) = 1, for n = 1
f(n) = 2, for n = 2
f(n) = f(n-1) + f(n-2), for n >= 2
'''



from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
import numpy as np
from torch import nn
X = np.arange(0, 20).reshape(10,2)
X = torch.Tensor(X)
print(X)
y = [0 for i in range(10)]
y = torch.Tensor(y)
print(y)

ds = TensorDataset(X, y)

myloader = DataLoader(dataset=ds, batch_size=2, shuffle=False, drop_last = False)
for data in myloader:
    print(data)



loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()
