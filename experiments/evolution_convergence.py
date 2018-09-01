"""
Perform (mixed real and imaginary) time evolution using the TDVP algorithm and
numerically investigate rate of convergence (should be quadratic).

Reference:
    J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
    Unifying time evolution and optimization with matrix product states
    Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
"""

from __future__ import print_function
import numpy as np
from scipy.linalg import expm
import copy
import matplotlib.pyplot as plt
import sys
sys.path.append('../pytenet/')
from mps import MPS
import hamiltonian
import evolution


def main():

    # number of lattice sites
    L = 10

    # construct matrix product operator representation of Heisenberg Hamiltonian
    J  =  1.0
    DH =  1.2
    h  = -0.2
    mpoH = hamiltonian.heisenberg_XXZ_MPO(L, J, DH, h)

    # initial wavefunction as MPS with random entries
    Dmax = 20
    D = np.minimum(np.minimum(2**np.arange(L + 1), 2**(L - np.arange(L + 1))), Dmax)
    print('D:', D)
    np.random.seed(42)
    psi = MPS(2, D, fill='random')
    # effectively clamp virtual bond dimension
    for i in range(L):
        psi.A[i][:, 3:, :] = 0
        psi.A[i][:, :, 3:] = 0
    psi.orthonormalize(mode='right')
    psi.orthonormalize(mode='left')

    tmax = 0.1 + 0.5j

    # reference calculation
    psi_ref = np.dot(expm(-tmax*mpoH.as_matrix()), psi.as_vector())

    # number of time steps
    numsteps = 2**(np.arange(5))
    err = np.zeros(len(numsteps))

    for i, n in enumerate(numsteps):
        print('n:', n)

        # time step
        dt = tmax / n

        psi_t = copy.deepcopy(psi)
        evolution.integrate_local_singlesite(mpoH, psi_t, dt, n, numiter_lanczos=10)

        err[i] = np.linalg.norm(psi_t.as_vector() - psi_ref)

    dtinv = numsteps / abs(tmax)
    plt.loglog(dtinv, err, '.-')
    # show quadratic scaling for comparison
    plt.loglog(dtinv, 1.75e-4/dtinv**2, '--')
    plt.xlabel('1/dt')
    plt.ylabel('err')
    plt.title('TDVP time evolution rate of convergence for\nHeisenberg XXZ model (J={:g}, D={:g}, h={:g})'.format(J, DH, h))
    plt.savefig('evolution_convergence.pdf')
    plt.show()

    # exact singular values (Schmidt coefficients) at center bond
    sigma = np.linalg.svd(np.reshape(psi_ref, (2**(L//2), 2**(L//2))),
                          full_matrices=0, compute_uv=False)
    plt.semilogy(np.arange(len(sigma)) + 1, sigma, '.')
    plt.xlabel('i')
    plt.ylabel('$\sigma_i$')
    plt.title('Schmidt coefficients of time-evolved state at center bond')
    plt.savefig('evolution_convergence_sigma.pdf')
    plt.show()


if __name__ == '__main__':
    main()
