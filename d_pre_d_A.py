
"""
Sample code automatically generated on 2022-04-17 12:25:45

by www.matrixcalculus.org

from input

d/dA b'*H'*A'*M*A*H*b = M*A*H*b*(H*b)'+M'*A*H*b*(H*b)'

where

A is a matrix
H is a matrix
M is a matrix
b is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(A, H, M, b):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(H, np.ndarray)
    dim = H.shape
    assert len(dim) == 2
    H_rows = dim[0]
    H_cols = dim[1]
    assert isinstance(M, np.ndarray)
    dim = M.shape
    assert len(dim) == 2
    M_rows = dim[0]
    M_cols = dim[1]
    assert isinstance(b, np.ndarray)
    dim = b.shape
    assert len(dim) == 1
    b_rows = dim[0]
    assert H_cols == b_rows
    assert H_rows == A_cols
    assert A_rows == M_rows == M_cols

    t_0 = (H).dot(b)
    t_1 = (A).dot(t_0)
    t_2 = (M).dot(t_1)
    functionValue = (b).dot((H.T).dot((A.T).dot(t_2)))
    gradient = (np.outer(t_2, t_0) + np.outer((M.T).dot(t_1), t_0))

    return functionValue, gradient

def checkGradient(A, H, M, b):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(4, 4)
    f1, _ = fAndG(A + t * delta, H, M, b)
    f2, _ = fAndG(A - t * delta, H, M, b)
    f, g = fAndG(A, H, M, b)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData():
    A = np.random.randn(4, 4)
    H = np.random.randn(4, 4)
    M = np.random.randn(4, 4)
    b = np.random.randn(4)

    return A, H, M, b

if __name__ == '__main__':
    A, H, M, b = generateRandomData()
    functionValue, gradient = fAndG(A, H, M, b)
    print('functionValue = ', functionValue.shape)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(A, H, M, b)



def d_pre_d_A():  # A, H, M, b
    t_0 = (H).dot(b)
    t_1 = (A).dot(t_0)
    t_2 = (M).dot(t_1)
    gradient = (np.outer(t_2, t_0) + np.outer((M.T).dot(t_1), t_0))
    return gradient