"""
Numerically investigate MPS tangent space dimension.

Reference:
    J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
    Unifying time evolution and optimization with matrix product states
    Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
"""

import numpy as np
import sys
sys.path.append('../pytenet/')
from mps import MPS


def main():

    # physical dimension
    d = 3
    # fictitious bond dimensions (should be bounded by d^i and d^(L-i))
    D = [1, 2, 5, 7, 3, 1]
    # number of lattice sites
    L = len(D) - 1
    print('L:', L)

    psi = MPS(d, D, fill='random')

    T = []
    for i in range(L):
        s = psi.A[i].shape
        print('s:', s)
        for j in range(s[0]):
            for a in range(s[1]):
                for b in range(s[2]):
                    # derivative in direction (i, j, a, b)
                    B = np.zeros_like(psi.A[i])
                    B[j, a, b] = 1
                    # backup original A[i] tensor
                    Ai = psi.A[i]
                    psi.A[i] = B
                    T.append(psi.as_vector())
                    # restore A[i]
                    psi.A[i] = Ai

    T = np.array(T)
    num_entries = np.sum([Ai.size for Ai in psi.A])
    print('T.shape: ', T.shape)
    print('expected:', (num_entries, d**L))
    print('rank of T:', np.linalg.matrix_rank(T))
    # number of degrees of freedom based on sandwiching "X" matrices between bonds,
    # -2 for omitting the leading and trailing entry 1 in D
    print('expected: ', num_entries - ((np.array(D)**2).sum() - 2))

    # realization of random X matrices
    X = [np.identity(1, dtype=complex)]
    for i in range(L - 1):
        X.append(np.random.normal(size=(D[i+1], D[i+1])) + 1j*np.random.normal(size=(D[i+1], D[i+1])))
    X.append(np.identity(1, dtype=complex))
    N = []
    for i in range(L):
        B = np.tensordot(X[i], psi.A[i], axes=(1, 1)).transpose((1, 0, 2)) - \
            np.tensordot(psi.A[i], X[i+1], axes=(2, 0))
        # backup original A[i] tensor
        Ai = psi.A[i]
        psi.A[i] = B
        N.append(psi.as_vector())
        # restore A[i]
        psi.A[i] = Ai
    N = np.array(N)
    print('N.shape:', N.shape)
    # N should be contained in range of T
    print('rank of [T, N]:', np.linalg.matrix_rank(np.concatenate((T, N), axis=0)),
          '(should agree with rank of T)')
    # should be numerically zero by construction
    z = np.dot(N.transpose(), np.ones(L))
    print('|z|:', np.linalg.norm(z), '(should be numerically zero)')


if __name__ == '__main__':
    main()
