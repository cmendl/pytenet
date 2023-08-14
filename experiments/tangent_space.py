"""
Numerically investigate MPS tangent space dimension and projector.

Reference:
    J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
    Unifying time evolution and optimization with matrix product states
    Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
"""

import numpy as np
import scipy
import copy
import pytenet as ptn


def tangent_space_projector(psi: ptn.MPS):
    """
    Construct tangent space projector as matrix based on MPS formalism.
    """
    # physical local site dimension
    d = len(psi.qd)

    # number of lattice sites
    L = psi.nsites

    psi_c = copy.deepcopy(psi)

    # construct P_L operators
    psi_c.orthonormalize(mode='left')
    PL = []
    x = np.array([[[1.]]])
    for i in range(L):
        x = ptn.merge_mps_tensor_pair(x, psi_c.A[i])
        assert x.ndim == 3 and x.shape[1] == 1
        xmat = x.reshape((x.shape[0], x.shape[2]))
        # check orthonormalization
        assert np.allclose(xmat.conj().T @ xmat, np.identity(xmat.shape[1]))
        PL.append(xmat @ xmat.conj().T)

    # construct P_R operators
    psi_c.orthonormalize(mode='right')
    PR = []
    x = np.array([[[1.]]])
    for i in reversed(range(L)):
        x = ptn.merge_mps_tensor_pair(psi_c.A[i], x)
        assert x.ndim == 3 and x.shape[2] == 1
        xmat = x.reshape(x.shape[:2])
        # check orthonormalization
        assert np.allclose(xmat.conj().T @ xmat, np.identity(xmat.shape[1]))
        PR.append(xmat @ xmat.conj().T)
    PR = list(reversed(PR))

    # construct projector
    P = 0
    P += np.kron(np.identity(d), PR[1])
    for i in range(1, L-1):
        P += np.kron(np.kron(PL[i-1], np.identity(d)), PR[i+1])
    P += np.kron(PL[L-2], np.identity(d))
    for i in range(0, L-1):
        P -= np.kron(PL[i], PR[i+1])

    return P


def main():

    # physical dimension
    d = 3
    # fictitious bond dimensions (should be bounded by d^i and d^(L-i))
    D = [1, 2, 5, 7, 3, 1]

    # number of lattice sites
    L = len(D) - 1
    print('L:', L)

    psi = ptn.MPS(np.zeros(d, dtype=int), [np.zeros(Di, dtype=int) for Di in D], fill='random')

    # construct MPS derivatives with respect to entries of the A tensors
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
    rank = num_entries - ((np.array(D)**2).sum() - 2)
    print('expected: ', rank)

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
    z = N.transpose() @ np.ones(L)
    print('|z|:', np.linalg.norm(z), '(should be numerically zero)')

    # reference tangent space projector based on T
    # rank-revealing QR decomposition
    Q, R, _ = scipy.linalg.qr(T.T, mode='economic', pivoting=True)
    P_ref = Q[:, :rank] @ Q[:, :rank].conj().T

    # construct tangent space projector based on MPS formalism
    P = tangent_space_projector(psi)
    # compare
    print('|P - P_ref|:', np.linalg.norm(P - P_ref), '(should be numerically zero)')

    # apply projector to psi (psi should remain unaffected)
    x = psi.as_vector()
    print('|P psi - psi|:', np.linalg.norm(P @ x - x), '(should be numerically zero)')

    # define another state
    # fictitious bond dimensions (should be bounded by d^i and d^(L-i))
    D = [1, 4, 7, 5, 3, 1]
    chi = ptn.MPS(np.zeros(d, dtype=int), [np.zeros(Di, dtype=int) for Di in D], fill='random')

    # tangent space projector corresponding to the sum of two states
    Psum = tangent_space_projector(psi + chi)
    # apply projector to psi (should remain unaffected)
    x = psi.as_vector()
    print('|Psum psi - psi|:', np.linalg.norm(Psum @ x - x), '(should be numerically zero)')
    # apply projector to chi (should remain unaffected)
    x = chi.as_vector()
    print('|Psum chi - chi|:', np.linalg.norm(Psum @ x - x), '(should be numerically zero)')
    # apply projector to psi + chi (should remain unaffected)
    x = psi.as_vector() + chi.as_vector()
    print('|Psum (psi + chi) - (psi + chi)|:', np.linalg.norm(Psum @ x - x), '(should be numerically zero)')


if __name__ == '__main__':
    main()
