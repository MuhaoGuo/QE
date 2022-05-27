"""
Sample code automatically generated on 2022-04-17 12:53:39

by www.matrixcalculus.org

from input

d/dphi I-(1/2)*phi*phi = -(1/2*phi'\otimes eye+1/2*eye\otimes phi)

where

I is a matrix
phi is a matrix

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(I, phi):
    assert isinstance(I, np.ndarray)
    dim = I.shape
    assert len(dim) == 2
    I_rows = dim[0]
    I_cols = dim[1]
    assert isinstance(phi, np.ndarray)
    dim = phi.shape
    assert len(dim) == 2
    phi_rows = dim[0]
    phi_cols = dim[1]
    assert I_rows == phi_rows == phi_cols
    assert phi_cols == phi_rows == I_cols
    assert I_cols == phi_rows == phi_cols == I_rows

    t_0 = (1 / 2)
    functionValue = (I - (t_0 * (phi).dot(phi)))
    gradient = -((t_0 * np.einsum('ik, jl', np.eye(I_rows, phi_rows), phi.T)) + (t_0 * np.einsum('ik, jl', phi, np.eye(phi_cols, phi_cols))))

    return functionValue, gradient

def checkGradient(I, phi):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3, 3)
    f1, _ = fAndG(I, phi + t * delta)
    f2, _ = fAndG(I, phi - t * delta)
    f, g = fAndG(I, phi)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData():
    I = np.random.randn(3, 3)
    phi = np.random.randn(3, 3)

    return I, phi

if __name__ == '__main__':
    I, phi = generateRandomData()
    functionValue, gradient = fAndG(I, phi)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(I, phi)


def d_A_d_phi_real(phi):
    t_0 = (1 / 2)
    gradient = -((t_0 * np.einsum('ik, jl', np.eye(4, 4), phi.T)) + (t_0 * np.einsum('ik, jl', phi, np.eye(4, 4))))
    print(gradient.shape)
    return gradient

phi = np.random.randn(4, 4)
d_A_d_phi_real(phi)


