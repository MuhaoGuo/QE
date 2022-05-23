
"""
ad hoc dataset
"""
import random
import numpy as np
import scipy
from qiskit.utils import algorithm_globals
# https://qiskit.org/documentation/stable/0.26/_modules/qiskit/utils/algorithm_globals.html
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_machine_learning.datasets.dataset_helper import (features_and_labels_transform,)
import pandas as pd

def phi_function(x_1, x_2):
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
    phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m) + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
    # ########################
    # （2）
    # # 只要x1 x2 单独的。-------------
    # phi = ( x_1 * np.kron(z_m, j_m) + x_2 * np.kron(j_m, z_m))

    # # 只要 x1 x2  都是先z 后 i
    # phi = (x_1 * np.kron(z_m, j_m) + x_2 * np.kron(z_m, j_m))

    #
    # phi = (np.pi * x_1 * np.kron(z_m, j_m)  + np.pi* x_2 * np.kron(z_m, j_m))
    #（3）gap 0.1
    # phi = ( np.pi* x_1 * np.kron(z_m, j_m) + np.pi* x_2 * np.kron(j_m, z_m) )

    #
    # # 只要 x1 x2  都是先i 后 z
    # phi = (x_1 * np.kron(j_m, z_m)+ x_2 * np.kron(j_m, z_m)  )

    # # X gate 一样
    # phi = ( x_1 * np.kron(j_m, x_m)  + x_2 * np.kron(x_m, j_m))

    # （4）gap 0.1
    # # 只要混合的 x1 x2 -----------
    # phi = ( + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m) )

    # （5）gap 0.1
    # # pi/2 * (1- x1)(1 - x2) * (z gate ,z gate )
    # phi = ( + np.pi /2 * (1 - x_1)*(1 - x_2) * np.kron(z_m, z_m)  )

    # （6）gap 0.1
    # # # pi * (cos x1)(cos x2) * (z gate ,z gate )
    # phi = ( + np.pi * np.cos(x_1) * np.cos(x_2) * np.kron(z_m, z_m)   )

    # （7）gap 0.1
    # phi = ( + np.pi * np.sin(x_1) * np.cos(x_2) * np.kron(z_m, z_m)   )

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
    # phi = ( 10 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m))

    # phi = (0.1 * np.sin(x_1) * np.sin(x_2) * np.kron(z_m, z_m)  )

    # ### 2 种都要
    # (9)
    # # x1 Z + x2 Z +  pi * (cos x1)(cos x2) * (z gate ,z gate )
    # phi = ( + x_1 + x_2 + x_1 * x_2 * np.kron(z_m, z_m) )

    # (10)
    # phi = ( x_1 * np.kron(z_m, j_m)  + x_2 * np.kron(j_m, z_m)  + (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))

    return phi

def ad_hoc_data(training_size, test_size, n, gap, repeat, plot_data=False, one_hot=True, include_sample_total=False, ):
    """returns ad hoc dataset"""
    class_labels = [r"A", r"B"]
    count = 0
    if n == 2:
        count = 100  # 本来是100， sample_total 的大小 （count， count）
    # elif n == 3:
    #     count = 6     # 原来是20 coarseness of data separation （20，20，20）
    # else:
    #     raise ValueError(f"Supported values of 'n' are 2 and 3 only, but {n} is provided.")

    label_train = np.zeros(2 * (training_size + test_size))  # 这里的 label_train 其实就是 train + test ，且 *2 （class1 class2）
    # print("label_train", label_train)  # 2*(5+2)    ：[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    sample_train = []
    sample_a = [[0 for _ in range(n)] for _ in range(
        training_size + test_size)]  ## print("sample_a", sample_a)  # (7,2)  [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    sample_b = [[0 for _ in range(n)] for _ in range(
        training_size + test_size)]  ## print("sample_b", sample_b)  # (7,2)  [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    sample_total = [[[0 for _ in range(count)] for _ in range(count)] for _ in range(count)]  # 只是记录标签
    # print("sample_total", sample_total)  #(6,6,6) ：  这里默认是n=3， 如果n=2，其实后面变成2维了。  [[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], .... ]]

    # interactions = np.transpose(np.array([[1, 0], [0, 1], [1, 1]]))

    steps = 2 * np.pi / count  # 2 pi / 6  每次旋转 steps 弧度？ 分成 6份。
    # print("steps", steps)        # steps 1.0471975511965976

    # sx = np.array([[0, 1], [1, 0]])
    # X = np.asmatrix(sx)
    # sy = np.array([[0, -1j], [1j, 0]])
    # Y = np.asmatrix(sy)
    s_z = np.array([[1, 0], [0, -1]])  # Z gate
    # print("s_z", s_z)
    z_m = np.asarray(s_z)
    # print("z_m", z_m)

    j_m = np.array([[1, 0], [0, 1]])  # 单位矩阵？
    j_m = np.asarray(j_m)

    h_m = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # H GATE
    # h_m[[0.70710678  0.70710678]
    # [0.70710678 - 0.70710678]]

    h_2 = np.kron(h_m, h_m)
    # print("h_2", h_2)    # (4,4)  2 量子系统用到  (n=2 时)
    # h_2[[0.5  0.5  0.5   0.5]
    #     [0.5 -0.5 0.5   -0.5]
    #     [0.5  0.5 -0.5  -0.5]
    #     [0.5 -0.5 -0.5  0.5]]

    h_3 = np.kron(h_m, h_2)
    # print("h_3", h_3)    # (8,8)  3 量子系统用到  (n=3 时)

    h_m = np.asarray(h_m)
    h_2 = np.asarray(h_2)
    h_3 = np.asarray(h_3)

    f_a = np.arange(2 ** n)  # 辅助用吧 ， my_array 外层个数和 f_a 一样
    # print("f_a", f_a)   # n=2, [0 1 2 3]

    my_array = [[0 for _ in range(n)] for _ in range(2 ** n)]  # my_array 外层个数和 f_a 一样
    # print("my_array", my_array)  # (2^n, n) ->  (2^2, 2) :  [[0, 0], [0, 0], [0, 0], [0, 0]]

    # 最终 生成 my_array: [[0, 0], [0, 1], [1, 0], [1, 1]]， --> transport ： [[0 0 1 1], [0 1 0 1]]
    for arindex, _ in enumerate(my_array):  # my_array 外层循环。
        temp_f = bin(f_a[arindex])[2:].zfill(
            n)  # returns the binary equivalent string of a given integer.  adds zeros (0) at the beginning of the string, until it reaches the specified length.
        # print("f_a[arindex])", f_a[arindex])       # 0 1 2 3
        # print("bin(f_a[arindex])", bin(f_a[arindex]))  # 0b0  0b1  0b10  0b11
        # print("bin(f_a[arindex])[2:]", bin(f_a[arindex])[2:]) # 0 1 10 11
        # print("temp_f", temp_f)  # 00 01 10 11
        for findex in range(n):  # my_array 内层循环
            my_array[arindex][findex] = int(temp_f[findex])
            # print("int(temp_f[findex])", int(temp_f[findex])) # 如 0
    # print("my_array", my_array)    # [[0, 0], [0, 1], [1, 0], [1, 1]]
    my_array = np.asarray(my_array)
    my_array = np.transpose(my_array)
    # print("my_array", my_array)    # [[0 0 1 1], [0 1 0 1]]

    # Define decision functions
    maj = (-1) ** (2 * my_array.sum(axis=0) > n)  # n =3 才用到
    # print("2 * my_array.sum(axis=0) > n", 2 * my_array.sum(axis=0) > n) # [0 2 2 4]-> [False False False True]
    # print("maj", maj)         # [ 1  1  1 -1]
    # print("", (-1) ** False)  # 1
    # print("", (-1) ** True).  # -1
    parity = (-1) ** (my_array.sum(axis=0))  # parity奇偶性 [ 1 -1 -1  1],    -1 的 (0, 1,1,2) 次方  n =2 用到
    # print("parity", parity)
    # dict1 = (-1) ** (my_array[0])

    # d_m： 就是 f: parity  function  d_m 是 zz 吧， as  the  parity  function
    d_m = None
    if n == 2:
        d_m = np.diag(parity)  # 对角矩阵
    elif n == 3:
        d_m = np.diag(maj)

    # print("d_m", d_m)
    # d_m[
    # [1  0  0  0]
    # [0 -1  0  0]
    # [0  0 -1  0]
    # [0  0  0  1]]

    # algorithm_globals.random: "Return a numpy np.random.Generator
    seed = 12345
    algorithm_globals.random_seed = seed

    basis = algorithm_globals.random.random((2 ** n, 2 ** n)) + \
            1j * algorithm_globals.random.random((2 ** n, 2 ** n))
    # print("basis", basis)         #(4,4) 随机的 basis？
    # print("basis1", basis.shape)   #(4,4) 随机的 basis？ 任意选择基底。

    basis = np.asarray(basis).conj().T @ np.asarray(basis)  # basis 的 共轭转置 与 basis 每一项 都相乘
    # print("basis", basis) # basis 的 共轭转置
    # print("basis2", basis.shape) #  (4, 4)

    [s_a, u_a] = np.linalg.eig(basis)  # 不是e指数！！。而是 Compute the eigenvalues and right eigenvectors of a square array.
    # print("[s_a, u_a]", [s_a, u_a])
    # print("s_a", s_a)
    # print(s_a.shape) # (4,)    eigenvalues
    # print(u_a.shape) # (4, 4)  eigenvectors

    idx = s_a.argsort()[::-1]  # eigenvalues 从大到小 排列 的index
    # print("s_a.argsort()", s_a.argsort())  # 如 [1 2 3 0] eigenvalues 从小到大 排列 的index
    # print("idx", idx)                      # 如 [0 3 2 1]eigenvalues 从大到小 排列 的index

    s_a = s_a[idx]
    # print("u_a", u_a)
    # array([[0.56138392 + 0.j,         -0.36746393 + 0.43128411j,  -0.44056741 + 0.39963147j,  0.08485788 + 0.05297682j],
    #        [0.45096947 + 0.10867222j, -0.30418847 - 0.09243137j,  0.63693007 + 0.j,           -0.52729117 + 0.00517816j],
    #        [0.47129903 + 0.04329199j,  0.62463951 + 0.j,          -0.2994188 - 0.3087979j,    -0.31316791 - 0.32054112j],
    #        [0.49105463 - 0.06733019j,  0.0615929 - 0.42886115j,   0.22283818 - 0.07648864j,    0.71491892 + 0.j]])

    u_a = u_a[:, idx]  # u_a： 就是 V: a random  unitary V,  BELONG TO SU(4) .  eigenvectors 内层按  idx 排列，外层不变 文章里所谓的 V
    # print("u_a", u_a)
    # [[0.56138392 + 0.j          0.08485788 + 0.05297682j    - 0.44056741 + 0.39963147j   - 0.36746393 + 0.43128411j]
    #  [0.45096947 + 0.10867222j  - 0.52729117 + 0.00517816j   0.63693007 + 0.j             - 0.30418847 - 0.09243137j]
    # [0.47129903 + 0.04329199j   - 0.31316791 - 0.32054112j  - 0.2994188 - 0.3087979j      0.62463951 + 0.j]
    # [0.49105463 - 0.06733019j   0.71491892 + 0.j              0.22283818 - 0.07648864j    0.0615929 - 0.42886115j]]

    # m_m： 就是 (V.+)(f)(V),      (ua.T)(d_m)(ua) :   expectation
    # d_m： 就是 f: parity  function  d_m 是 zz ， as  the  parity  function
    # u_a： 就是 V: a random unitary V,  BELONG TO SU(4) .

    # 换一个 d_m:
    # d_m = np.diag([1,1,1,1]) # 相当于没有
    # d_m = np.diag([1, 0, 0, 0])
    m_m = (np.asarray(u_a)).conj().T @ np.asarray(d_m) @ np.asarray(u_a)  # u_a共轭转置 * 对角矩阵 * u_a
    print("m_m", m_m)
    # m_m = np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    # m_m =  np.asarray(d_m)

    # print("ua.T  ua:") # 不要 d_m 的话？：
    # print(np.asarray(u_a).conj().T @ np.asarray(u_a))
    # m_m = np.asarray(u_a).conj().T @ np.asarray(u_a)

    # print("m_m", m_m)  (4,4)
    # [[0.1756782 + 0.j             - 0.43635024 + 0.20973891j   - 0.69224641 + 0.0971944j     0.37503739 + 0.32471578j]
    # [-0.43635024 - 0.20973891j    - 0.26198691 + 0.j           0.51802404 + 0.05794638j       0.38503943 + 0.52631631j]
    # [-0.69224641 - 0.0971944j      0.51802404 - 0.05794638j    - 0.47244213 + 0.j             - 0.02201489 + 0.12631408j]
    # [0.37503739 - 0.32471578j      0.38503943 - 0.52631631j     - 0.02201489 - 0.12631408j      0.55875083 + 0.j]]

    psi_plus = np.transpose(np.ones(2)) / np.sqrt(2)
    # print("psi_plus", psi_plus)   #  [0.70710678 0.70710678]

    psi_0 = [[1]]
    for k in range(n):
        psi_0 = np.kron(np.asarray(psi_0), np.asarray(psi_plus))
        # print("psi_0", psi_0)
        # psi_0 [[0.70710678 0.70710678]]
        # psi_0 [[0.5 0.5 0.5 0.5]]

    sample_total_a = []
    sample_total_b = []
    sample_total_void = []

    all_temp = []  ##
    ALL_X = []  ##
    if n == 2:
        for n_1 in range(count):  # 6    100
            for n_2 in range(count):  # 6    100
                x_1 = steps * n_1  # 每次转 1/100 圆 x_1 ，x_2 坐标  。 steps = 2 * np.pi / count  # 2 pi / 6  每次旋转 steps 弧度？
                x_2 = steps * n_2
                ALL_X.append([x_1, x_2])

                ###### feature map： phi（x）##################################################################################################################
                # # feature map： phi（x）
                phi = phi_function(x_1, x_2)  # 调用 phi_function 函数
                ################################################################################################################################################

                # print("x_1 * np.kron(z_m, j_m) ", x_1 * np.kron(z_m, j_m) )
                # [[0.31415927  0.            0.           0.]
                # [0.           0.31415927    0.           0.]
                # [0.           0.         - 0.31415927    0.]
                # [0.           0.           0. - 0.31415927]]

                # print("x_2 * np.kron(j_m, z_m)", x_2 * np.kron(j_m, z_m))
                # [[2.70176968  0.          0.          0.]
                # [0.        - 2.70176968  0.           0.]
                # [0.        0.          2.70176968     0.]
                # [0.          0.          0. - 2.70176968]]

                # print("(np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m)", (np.pi - x_1) * (np.pi - x_2) * np.kron(z_m, z_m))
                # [[1.24357015  0.          0.           0.]
                # [0. - 1.24357015          0.           0.]
                # [0.     0.         - 1.24357015        0.]
                # [0.          0.          0.      1.24357015]]

                # print("phi.shape", phi)  # phi.shape (4, 4)
                # phi
                # [[4.2594991   0.       0.      0.]
                # [0. - 3.63118057       0.      0.]
                # [0.    0.     1.14404026       0.]
                # [0.    0.      0.   - 1.77235879]]

                # U_phi(x) feature map 的 i 及 矩阵实现
                u_u = scipy.linalg.expm(1j * phi)  # pylint: disable=no-member  计算 e^A  的值，其中A为一个任意（应该是任意？）维数的方阵。
                # print("u_u.shape", u_u.shape)  # u_u.shape (4, 4)
                # u_u[
                # [-0.90268536 - 0.43030122j,  0. - 0.j,        0. + 0.j,    0. + 0.j]
                # [0. + 0.j,  - 0.90268536 + 0.43030122j,     - 0. + 0.j,    0. + 0.j]
                # [0. + 0.j,   0. + 0.j,   - 0.90268536 + 0.43030122j,     - 0. + 0.j]
                # [0. + 0.j,   0. + 0.j,    0. + 0.j,     - 0.90268536 - 0.43030122j]]

                ######### 重复次数 真正的feature map #####################################################################################################################
                # 重复2 次 真正的feature map： ZZ feature map 实现。 U_phi(x) H^2 U_phi(x) H^2 ｜0> ^n
                if repeat == 2:
                    psi = np.asarray(u_u) @ h_2 @ np.asarray(u_u) @ np.transpose(psi_0)  # ZZ map
                # print("np.asarray(u_u) @ h_2", np.asarray(u_u) @ h_2 )
                # print("np.asarray(u_u) @ h_2 @ np.asarray(u_u)", np.asarray(u_u) @ h_2 @ np.asarray(u_u))
                # # feature： 只重复1 次
                elif repeat == 1:
                    psi = np.asarray(u_u) @ np.transpose(psi_0)
                # # feature： 重复3 次
                elif repeat == 3:
                    psi = np.asarray(u_u) @ h_2 @ np.asarray(u_u) @ h_2 @ np.asarray(u_u) @ np.transpose(psi_0)
                # #重复4次
                elif repeat == 4:
                    psi = np.asarray(u_u) @ h_2 @ np.asarray(u_u) @ h_2 @ np.asarray(u_u) @ h_2 @ np.asarray(
                        u_u) @ np.transpose(psi_0)
                # # 重复5次
                elif repeat == 5:
                    psi = np.asarray(u_u) @ h_2 @ np.asarray(u_u) @ h_2 @ np.asarray(u_u) @ h_2 @ np.asarray(
                        u_u) @ h_2 @ np.asarray(u_u) @ np.transpose(psi_0)
                # # 重复6次
                elif repeat == 6:
                    psi = np.asarray(u_u) @ h_2 @ np.asarray(u_u) @ h_2 @ np.asarray(u_u) @ h_2 @ np.asarray(
                        u_u) @ h_2 @ np.asarray(
                        u_u) @ h_2 @ np.asarray(u_u) @ np.transpose(psi_0)
                ########################################################################################################################################################################

                # psi = np.asarray(u_u) @ h_2  @ np.transpose(psi_0)  # Z map 这个不但要调 gap 还要调 measure吧？？？
                # print("psi", psi)
                # print("psi.shape", psi.shape) # psi.shape (4, 1)
                # psi[
                # [8.14840863e-01 + 0.38842661j]
                # [-5.55111512e-17 + 0.j]
                # [1.38777878e-17 + 0.j]
                # [-1.85159137e-01 + 0.38842661j]]
                ###### 不是测量，是 expection。因为f是ZZ  ##################################################################################################################################################################
                #  m_m 是一种operator/hibert space ： real of 《psi｜ m_m ｜psi》  temp 是 期望。 为什么取 real？
                temp = np.real(
                    psi.conj().T @ m_m @ psi).item()  # 为什么取 real值 ？因为只有   operator是 hermintion 时，特征值才是real，导致期望是real。当然，这里operator应该是real
                # 但为什么有负数？期望应该可以是 负数。 只能是有些特征值可能为负数。
                # 去掉f之后，temp 都是1了。 也没什么分布了。说明：以 此时的 m_m 进行测量，得到的都是 m_m for sure
                print("temp", temp)

                # temp2 = np.real(psi.conj().T @ psi).item()  # 探究
                # print("temp", temp) #  一个float值
                # print("temp2", temp2) #  一个float值    等于1，毕竟是单位state
                all_temp.append(temp)
                # temp - 0.5391722272037293
                ########################################################################################################################################################################

                if temp > gap:
                    sample_total[n_1][n_2] = +1
                # elif temp < -gap:
                elif temp < -gap:
                    sample_total[n_1][n_2] = -1
                else:
                    sample_total[n_1][n_2] = 0

                # print("sample_total", sample_total)   # 这里是 n =2 ， sample_total 从3维直接降为2 维了
        # print("all_temp", all_temp)
        # print("sample_total all", sample_total)
        #
        # print("gap is", gap)
        ########### 画出 分配坐标前 的所有点的分布##################################
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5, 5))
        ax2 = fig.add_subplot(1, 1, 1)
        ax2.hist(all_temp, bins=1000, edgecolor='black')
        plt.show()

        ######### 画采样前的 所有的点 ####################################
        ALL_X = np.array(ALL_X)
        sample_total_ravel = np.array(sample_total).ravel()
        print("ALL_X",ALL_X)
        print("sample_total_ravel", sample_total_ravel)

        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.scatter(ALL_X[:, 0], ALL_X[:, 1], c=sample_total_ravel, cmap=plt.cm.coolwarm, )  # .cmap_d['OrRd']
        plt.show()

        ################## 采样 ###########################

        # Now sample randomly from sample_Total a number of times training_size+testing_size
        t_r = 0
        # sample_total [[1, 0, -1, 0, -1, -1], [0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 1, 0], [0, -1, -1, 0, -1, -1], [1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 1, 0]]
        while t_r < (training_size + test_size):
            draw1 = algorithm_globals.random.choice(count)
            draw2 = algorithm_globals.random.choice(count)
            # print("draw1", draw1) #如 3
            # print("draw2", draw2) #如 4
            # print(sample_total[draw1][draw2])
            if sample_total[draw1][draw2] == +1:
                sample_a[t_r] = [2 * np.pi * draw1 / count, 2 * np.pi * draw2 / count]  # 又生成一次坐标。与前面呼应
                t_r += 1
        # print("sample_a", sample_a)  # 第一类的 train +test

        t_r = 0
        while t_r < (training_size + test_size):
            draw1 = algorithm_globals.random.choice(count)
            draw2 = algorithm_globals.random.choice(count)
            if sample_total[draw1][draw2] == -1:
                sample_b[t_r] = [2 * np.pi * draw1 / count, 2 * np.pi * draw2 / count]  # 又生成一次坐标。与前面呼应
                t_r += 1
        # print("sample_b", sample_b)   # 第2 类的 train +test

        sample_train = [sample_a, sample_b]  # 总的data = 第一类 a + 第二类 b

        for lindex in range(training_size + test_size):
            label_train[lindex] = 0  # 【0，0，0，0，0，0，0】 # 第一类
        for lindex in range(training_size + test_size):
            label_train[training_size + test_size + lindex] = 1
        label_train = label_train.astype(int)  # 【0，0，0，0，0，0，0 ，1，1，1，1，1，1，1】

        # 所有data 暂时都叫 train
        sample_train = np.reshape(sample_train, (2 * (training_size + test_size), n))

        # trian 数量 根据 train size 定
        training_input = {
            key: (sample_train[label_train == k, :])[:training_size]
            for k, key in enumerate(class_labels)  # class_labels = [r"A", r"B"]   k:0,1 ; key:A ,B
        }
        # print("training_input", training_input)
        #  test 数量 根据 总量 - train size
        test_input = {
            key: (sample_train[label_train == k, :])[training_size: (training_size + test_size)]
            for k, key in enumerate(class_labels)
        }
        # print("test_input", test_input)

        if plot_data:
            try:
                import matplotlib.pyplot as plt
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    libname="Matplotlib",
                    name="ad_hoc_data",
                    pip_install="pip install matplotlib",
                ) from ex

            # plt.show()
            fig = plt.figure(figsize=(5, 5))
            for k in range(0, 2):
                plt.scatter(
                    sample_train[label_train == k, 0][:training_size],
                    sample_train[label_train == k, 1][:training_size],
                )
            plt.title("Ad-hoc Data")
            plt.show()

    training_feature_array, training_label_array = features_and_labels_transform(
        training_input, class_labels, one_hot
    )
    test_feature_array, test_label_array = features_and_labels_transform(
        test_input, class_labels, one_hot
    )
    print("training_feature_array", training_feature_array)
    print("training_label_array", training_label_array)
    if include_sample_total:
        return (
            training_feature_array,
            training_label_array,
            test_feature_array,
            test_label_array,
            sample_total,
        )
    else:
        return (
            training_feature_array,
            training_label_array,
            test_feature_array,
            test_label_array,
        )


# ##############产生数据集##################################
# repeat = 1
# X_train, y_train, X_test, y_test, sample_total = ad_hoc_data(
#     training_size = 25,
#     test_size = 10,
#     n=2,
#     gap=0.3,
#     repeat=repeat,
#     plot_data=True,
#     one_hot=False,
#     include_sample_total=True,
# )



########################### 存入 TEST_phi1_repeat1_' ########################
# y_train = [1 if i == 0 else -1 for i in y_train]
# y_test = [1 if i == 0 else -1 for i in y_test]
#
# dict_data_train = {"feature1": X_train.T[0], "feature2": X_train.T[1], "y_train": y_train,}
# dict_data_test = {"feature1": X_test.T[0],   "feature2": X_test.T[1],  "y_test": y_test,}
#
# import pandas as pd
# df_train = pd.DataFrame(dict_data_train,)
# df_test = pd.DataFrame(dict_data_test,)
#
# print(df_train)
# print(df_test)
#
# name ='TEST_phi1_repeat1_'
# df_train.to_csv("./quantum_circuit_designed_dataset/" + name + "train.csv")
# df_test.to_csv("./quantum_circuit_designed_dataset/" + name + "test.csv")



'''
dict_data_train = {"feature1": X_train.T[0], "feature2": X_train.T[1], "y_train": y_train,}
dict_data_test = {"feature1": X_test.T[0],   "feature2": X_test.T[1],  "y_test": y_test,}

import pandas as pd
df_train = pd.DataFrame(dict_data_train,)
df_test = pd.DataFrame(dict_data_test,)

print(df_train)
print(df_test)

# import os
# os.chdir('./dataset')
name ='phi1_repeat1_'
df_train.to_csv("./dataset/" + name + "train.csv")
df_test.to_csv("./dataset/" + name + "test.csv")
'''