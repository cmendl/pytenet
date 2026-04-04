"""
Perform real time evolution using the TDVP algorithm and
numerically investigate convergence rate (should be quadratic).

Reference:
    J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
    Unifying time evolution and optimization with matrix product states
    Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
"""

import copy
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pytenet as ptn


def entropy(eigvals: np.ndarray):
    """
    Compute the Neumann entropy given the eigenvalues of the density matrix.
    """
    eigvals = eigvals[eigvals > 0]
    return sum(-eigvals * np.log2(eigvals))


def schmidt_coefficients(d: int, nsites: int, psi: np.ndarray):
    """
    Compute the Schmidt coefficients (singular values)
    of a wavefunction for symmetric left-right partitioning.
    """
    return np.linalg.svd(psi.reshape((d**(nsites//2), d**(nsites//2))),
                         full_matrices=0, compute_uv=False)


def main():

    # physical local Hilbert space dimension
    d = 2

    # number of lattice sites
    nsites = 10

    # construct matrix product operator representation of Heisenberg Hamiltonian
    J  =  1.0
    D  =  1.2
    h  = -0.2
    h_mpo = ptn.heisenberg_xxz_1d_mpo(nsites, J, D, h)
    h_mpo.zero_qnumbers()

    # initial wavefunction as MPS with random entries
    b_max = 20
    b = np.minimum(np.minimum(d**np.arange(nsites + 1), d**(nsites - np.arange(nsites + 1))), b_max)
    print("b:", b)
    rng = np.random.default_rng(42)
    psi = ptn.MPS(h_mpo.qsite, [np.zeros(bi, dtype=int) for bi in b], fill="random", rng=rng)
    # effectively clamp virtual bond dimension
    for i in range(nsites):
        psi.a[i][3:, :, :] = 0
        psi.a[i][:, :, 3:] = 0
    psi.orthonormalize(mode="right")
    psi.orthonormalize(mode="left")

    # initial average energy (should be conserved)
    e_avr_0 = ptn.mpo_average(psi, h_mpo).real
    print("e_avr_0:", e_avr_0)

    # exact singular values (Schmidt coefficients) of initial state
    sigma_0 = schmidt_coefficients(d, nsites, psi.to_vector())
    plt.semilogy(np.arange(len(sigma_0)) + 1, sigma_0, ".")
    plt.xlabel("i")
    plt.ylabel(r"$\sigma_i$")
    plt.title("Schmidt coefficients of initial state")
    plt.savefig("time_evolution_schmidt_0.pdf")
    plt.show()
    print("entropy of initial state:", entropy((sigma_0 / np.linalg.norm(sigma_0))**2))

    # purely real time evolution
    t = 0.5j

    # reference calculation
    psi_ref = expm(-t*h_mpo.to_matrix()) @ psi.to_vector()

    # exact Schmidt coefficients (singular values) of time-evolved state
    sigma_t = schmidt_coefficients(d, nsites, psi_ref)
    plt.semilogy(np.arange(len(sigma_t)) + 1, sigma_t, ".")
    plt.xlabel("i")
    plt.ylabel(r"$\sigma_i$")
    plt.title(f"Schmidt coefficients of time-evolved state (t = {t.imag:g})\n"
              f"(based on exact time evolution)")
    plt.savefig("time_evolution_schmidt_t.pdf")
    plt.show()
    print("entropy of time-evolved state:", entropy((sigma_t / np.linalg.norm(sigma_t))**2))

    # number of time steps
    numsteps = 2**(np.arange(5))
    err = np.zeros(len(numsteps))

    for i, n in enumerate(numsteps):
        print("n:", n)

        # time step
        dt = t / n

        psi_t = copy.deepcopy(psi)
        ptn.tdvp_singlesite(h_mpo, psi_t, dt, n, numiter_lanczos=10)

        err[i] = np.linalg.norm(psi_t.to_vector() - psi_ref)

        # expecting numerically exact energy conservation
        # (for real time evolution)
        e_avr_t = ptn.mpo_average(psi_t, h_mpo).real
        print("e_avr_t:", e_avr_t)
        print("abs(e_avr_t - e_avr_0):", abs(e_avr_t - e_avr_0))

    dtinv = numsteps / abs(t)
    plt.loglog(dtinv, err, ".-")
    # show quadratic scaling for comparison
    plt.loglog(dtinv, 1e-4/dtinv**2, "--")
    plt.xlabel(r"$1/\mathrm{dt}$")
    plt.ylabel(r"$\Vert\psi[A](t) - \psi_\mathrm{ref}(t)\Vert$")
    plt.title(f"TDVP time evolution rate of convergence (t = {t.imag:g}) for\n"
              f"the Heisenberg XXZ model (J={J:g}, D={D:g}, h={h:g})")
    plt.savefig("time_evolution_convergence.pdf")
    plt.show()


if __name__ == "__main__":
    main()
