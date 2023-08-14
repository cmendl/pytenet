"""
Perform real time evolution using the TDVP algorithm and
numerically investigate rate of convergence (should be quadratic).

Reference:
    J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
    Unifying time evolution and optimization with matrix product states
    Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
"""

import numpy as np
from scipy.linalg import expm
import copy
import matplotlib.pyplot as plt
import pytenet as ptn


def entropy(lmbda: np.ndarray):
    """
    Compute the Neumann entropy given the eigenvalues of the density matrix.
    """
    lmbda = lmbda[lmbda > 0]
    return -np.dot(lmbda, np.log2(lmbda))


def schmidt_coefficients(d: int, L: int, psi: np.ndarray):
    """
    Compute the Schmidt coefficients (singular values)
    of a wavefunction for symmetric left-right partitioning.
    """
    return np.linalg.svd(psi.reshape((d**(L//2), d**(L//2))), full_matrices=0, compute_uv=False)


def main():

    # physical local Hilbert space dimension
    d = 2

    # number of lattice sites
    L = 10

    # construct matrix product operator representation of Heisenberg Hamiltonian
    J  =  1.0
    DH =  1.2
    h  = -0.2
    mpoH = ptn.heisenberg_xxz_mpo(L, J, DH, h)
    mpoH.zero_qnumbers()

    # initial wavefunction as MPS with random entries
    Dmax = 20
    D = np.minimum(np.minimum(d**np.arange(L + 1), d**(L - np.arange(L + 1))), Dmax)
    print('D:', D)
    np.random.seed(42)
    psi = ptn.MPS(mpoH.qd, [np.zeros(Di, dtype=int) for Di in D], fill='random')
    # effectively clamp virtual bond dimension
    for i in range(L):
        psi.A[i][:, 3:, :] = 0
        psi.A[i][:, :, 3:] = 0
    psi.orthonormalize(mode='right')
    psi.orthonormalize(mode='left')

    # initial average energy (should be conserved)
    e_avr_0 = ptn.operator_average(psi, mpoH).real
    print('e_avr_0:', e_avr_0)

    # exact singular values (Schmidt coefficients) initial state
    sigma_0 = schmidt_coefficients(d, L, psi.as_vector())
    plt.semilogy(np.arange(len(sigma_0)) + 1, sigma_0, '.')
    plt.xlabel('i')
    plt.ylabel('$\sigma_i$')
    plt.title('Schmidt coefficients of initial state')
    plt.savefig('evolution_schmidt_0.pdf')
    plt.show()
    print('entropy of initial state:', entropy((sigma_0 / np.linalg.norm(sigma_0))**2))

    # purely real time evolution
    t = 0.5j

    # reference calculation
    psi_ref = expm(-t*mpoH.as_matrix()) @ psi.as_vector()

    # exact Schmidt coefficients (singular values) of time-evolved state
    sigma_t = schmidt_coefficients(d, L, psi_ref)
    plt.semilogy(np.arange(len(sigma_t)) + 1, sigma_t, '.')
    plt.xlabel('i')
    plt.ylabel('$\sigma_i$')
    plt.title(f'Schmidt coefficients of time-evolved state (t = {t.imag:g})\n(based on exact time evolution)')
    plt.savefig('evolution_schmidt_t.pdf')
    plt.show()
    print('entropy of time-evolved state:', entropy((sigma_t / np.linalg.norm(sigma_t))**2))

    # number of time steps
    numsteps = 2**(np.arange(5))
    err = np.zeros(len(numsteps))

    for i, n in enumerate(numsteps):
        print('n:', n)

        # time step
        dt = t / n

        psi_t = copy.deepcopy(psi)
        ptn.integrate_local_singlesite(mpoH, psi_t, dt, n, numiter_lanczos=10)

        err[i] = np.linalg.norm(psi_t.as_vector() - psi_ref)

        # expecting numerically exact energy conservation
        # (for real time evolution)
        e_avr_t = ptn.operator_average(psi_t, mpoH).real
        print('e_avr_t:', e_avr_t)
        print('abs(e_avr_t - e_avr_0):', abs(e_avr_t - e_avr_0))

    dtinv = numsteps / abs(t)
    plt.loglog(dtinv, err, '.-')
    # show quadratic scaling for comparison
    plt.loglog(dtinv, 1.75e-4/dtinv**2, '--')
    plt.xlabel('1/dt')
    plt.ylabel(r'$\Vert\psi[A](t) - \psi_\mathrm{ref}(t)\Vert$')
    plt.title(f'TDVP time evolution rate of convergence (t = {t.imag:g}) for\nHeisenberg XXZ model (J={J:g}, D={DH:g}, h={h:g})')
    plt.savefig('evolution_convergence.pdf')
    plt.show()


if __name__ == '__main__':
    main()
