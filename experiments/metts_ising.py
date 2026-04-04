"""
Numerically investigate METTS sampling for computing dynamical
correlation functions at finite temperature.

Reference:
    E. M. Stoudenmire, S. R. White
    Minimally entangled typical thermal state algorithms
    New J. Phys. 12, 055026 (2010) (arXiv:1002.1305)
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pytenet as ptn


def random_bloch_basis(rng: np.random.Generator):
    """
    Generate a uniformly random orthonormal Bloch basis.
    """
    theta = np.arccos(2*rng.uniform()-1)
    phi   = 2*np.pi*rng.uniform()
    return np.array([[               np.cos(theta/2),               -np.sin(theta/2)],
                     [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*phi)*np.cos(theta/2)]])


def collapse_random_cps(nsites: int, psi: np.ndarray, rng: np.random.Generator):
    """"
    Sequentially collapse the wavefunction `psi` onto a classical product state (CPS)
    using a random local Bloch basis for each site.
    """
    cps = np.array([1.], dtype=complex)
    for _ in range(nsites):
        u = random_bloch_basis(rng)
        # project wavefunction onto normalized `u` states at leading site
        chi = u.conj().T @ np.reshape(psi, (2, -1))
        p = (np.linalg.norm(chi[0]), np.linalg.norm(chi[1]))
        # randomly choose one of the two states
        if rng.uniform() < p[0]**2:
            cps = np.kron(cps, u[:, 0])
            psi = chi[0] / p[0]
        else:
            cps = np.kron(cps, u[:, 1])
            psi = chi[1] / p[1]
    assert len(psi) == 1
    return cps


def site_operator(nsites: int, i: int, op: np.ndarray):
    """
    Construct operator acting non-trivially on a single site.
    """
    return np.kron(np.identity(2**i), np.kron(op, np.identity(2**(nsites-i-1))))


def main():

    # number of lattice sites
    nsites = 7

    # construct matrix representation of Ising Hamiltonian
    J =  1.0
    h =  0.8
    g = -0.375
    hamiltonian = ptn.ising_1d_mpo(nsites, J, h, g).to_matrix()

    # inverse temperature
    beta = 1.2

    # exp(-beta H/2)
    rho_beta = scipy.linalg.expm(-0.5*beta*hamiltonian)

    # partition function
    z = np.linalg.norm(rho_beta, "fro")**2

    rng = np.random.default_rng(857)

    # local operators
    op_a = site_operator(nsites, 2, ptn.crandn((2, 2), rng))
    op_b = site_operator(nsites, 4, ptn.crandn((2, 2), rng))

    tlist = np.linspace(0., 5., 41)

    # approximate response function at largest time point using METTS sampling

    # time evolution operator
    u_tmax = scipy.linalg.expm(-0.5j*tlist[-1]*hamiltonian)

    # initial classical product state
    cps = np.array([1.], dtype=complex)
    for i in range(nsites):
        cps = np.kron(cps, random_bloch_basis(rng)[:, 0])
    nsamples = 10000
    nlist = []
    chi_n = []
    s = 0
    for n in range(1, nsamples+1):
        phi = rho_beta @ cps
        phi /= np.linalg.norm(phi)
        # A(-t/2)|phi> and B(t/2)^{dagger}|phi>
        s_a = u_tmax          @ op_a          @ u_tmax.conj().T @ phi
        s_b = u_tmax.conj().T @ op_b.conj().T @ u_tmax          @ phi
        s += np.vdot(s_b, s_a)
        if (n & (n - 1)) == 0:
            # if n is a power of 2...
            nlist.append(n)
            chi_n.append(s / n)
        # next classical product state
        cps = collapse_random_cps(nsites, phi, rng)

    # numerically exact response function (reference calculation)
    chi_ref = np.zeros(len(tlist), dtype=complex)
    for i, t in enumerate(tlist):
        u = scipy.linalg.expm(-0.5j*t*hamiltonian)
        t_a = u          @ op_a     @ rho_beta @ u.conj().T
        t_b = u.conj().T @ rho_beta @ op_b     @ u
        chi_ref[i] = np.trace(t_b @ t_a) / z

    print(f"reference <B(t) A(0)>_beta at t = {tlist[-1]}: {chi_ref[-1]}")
    print(f"METTS approximation using {nsamples} samples: {chi_n[-1]}")

    # visualize response function
    plt.plot(tlist, chi_ref.real, tlist, chi_ref.imag)
    plt.xlabel("t")
    plt.ylabel(r"$\langle B(t) A(0) \rangle_{\beta}$")
    plt.legend(("real", "imag"))
    plt.title("exact response function")
    plt.show()

    # visualize METTS error
    plt.loglog(nlist, abs(chi_n - chi_ref[-1])/abs(chi_ref[-1]))
    plt.xlabel("num samples")
    plt.ylabel("rel err")
    plt.title(f"METTS approximation error of response function at t = {tlist[-1]}")
    plt.show()


if __name__ == "__main__":
    main()
