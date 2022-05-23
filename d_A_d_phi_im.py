"""
Sample code automatically generated on 2022-04-17 12:48:47

by www.matrixcalculus.org

from input

d/dphi phi*-(1/6)*phi*phi*phi = -(1/6*(phi*phi*phi)'\otimes eye+1/6*(phi*phi)'\otimes phi+1/6*phi'\otimes (phi*phi)+1/6*eye\otimes (phi*phi*phi))

where

phi is a matrix

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(phi):
    assert isinstance(phi, np.ndarray)
    dim = phi.shape
    assert len(dim) == 2
    phi_rows = dim[0]
    phi_cols = dim[1]
    assert phi_rows == phi_cols

    t_0 = (1 / 6)
    T_1 = (phi).dot(phi)
    T_2 = (T_1).dot(phi)
    functionValue = -(t_0 * (T_2).dot(phi))
    gradient = -((((t_0 * np.einsum('ik, jl', np.eye(phi_rows, phi_rows), T_2.T)) + (t_0 * np.einsum('ik, jl', phi, T_1.T))) + (t_0 * np.einsum('ik, jl', T_1, phi.T))) + (t_0 * np.einsum('ik, jl', T_2, np.eye(phi_rows, phi_cols))))

    return functionValue, gradient

def checkGradient(phi):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3, 3)
    f1, _ = fAndG(phi + t * delta)
    f2, _ = fAndG(phi - t * delta)
    f, g = fAndG(phi)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData():
    phi = np.random.randn(3, 3)

    return phi

if __name__ == '__main__':
    phi = generateRandomData()
    functionValue, gradient = fAndG(phi)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(phi)


def d_A_d_phi_im(phi):
    t_0 = (1 / 6)
    T_1 = (phi).dot(phi)
    T_2 = (T_1).dot(phi)
    gradient = -((((t_0 * np.einsum('ik, jl', np.eye(4, 4), T_2.T)) + (t_0 * np.einsum('ik, jl', phi, T_1.T))) + (t_0 * np.einsum('ik, jl', T_1, phi.T))) + (t_0 * np.einsum('ik, jl', T_2, np.eye(4, 4))))
    print(gradient.shape)
    return gradient

phi = np.random.randn(4, 4)
a  = d_A_d_phi_im(phi)
print(a)
print(a.shape)