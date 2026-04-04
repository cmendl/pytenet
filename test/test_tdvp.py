import copy
import numpy as np
from scipy.linalg import expm
import pytenet as ptn


def test_tdvp_approximation():

    rng = np.random.default_rng()

    # number of lattice sites
    nsites = 10

    # time step can have both real and imaginary parts;
    # for real-time evolution use purely imaginary dt!
    dt = 0.02 - 0.05j
    # number of steps
    numsteps = 12

    # construct matrix product operator representation of Heisenberg Hamiltonian
    J =  4.0/3
    D =  5.0/13
    h = -2.0/7
    h_mpo = ptn.heisenberg_xxz_1d_mpo(nsites, J, D, h)

    # fix total spin quantum number of wavefunction (trailing virtual bond)
    spin_tot = 2

    # enumerate all possible virtual bond quantum numbers (including multiplicities);
    # will be implicitly reduced by orthonormalization steps below
    qbonds = [np.array([0])]
    for i in range(nsites - 1):
        qbonds.append(np.sort(np.array([q + h_mpo.qsite for q in qbonds[-1]]).reshape(-1)))
    qbonds.append(np.array([2*spin_tot]))

    # initial wavefunction as MPS with random entries
    psi = ptn.MPS(h_mpo.qsite, qbonds, fill="random", rng=rng)
    psi.orthonormalize(mode="left")
    psi.orthonormalize(mode="right")
    # effectively clamp virtual bond dimension of initial state
    b_init = 8
    for i in range(nsites):
        psi.a[i][b_init:, :, :] = 0
        psi.a[i][:, :, b_init:] = 0
    # orthonormalize again
    psi.orthonormalize(mode="left")

    assert psi.qbonds[-1][0] == 2*spin_tot, \
        "trailing bond quantum number must not change during orthonormalization"

    # total spin operator as MPO
    sz_graph = ptn.OpGraph.from_opchains(
        [ptn.OpChain([1], [0, 0], 1.0, istart) for istart in range(nsites)], nsites, 0)
    sz_tot = ptn.MPO.from_opgraph(h_mpo.qsite, sz_graph,
                                  { 0: np.identity(2), 1: np.diag([0.5, -0.5]) })

    # explicitly compute average spin
    spin_avr = ptn.mpo_average(psi, sz_tot)
    assert abs(spin_avr - spin_tot) < 1e-14, \
        "average spin must be equal to prescribed value"

    # reference time evolution
    psi_ref = expm(-dt*numsteps * h_mpo.to_matrix()) @ psi.to_vector()

    # run TDVP time evolution
    psi1 = copy.deepcopy(psi)
    psi2 = copy.deepcopy(psi)
    ptn.tdvp_singlesite(h_mpo, psi1, dt, numsteps, numiter_lanczos=5)
    ptn.tdvp_twosite(h_mpo, psi2, dt, numsteps, numiter_lanczos=10)

    # compare time-evolved wavefunctions
    assert np.allclose(psi1.to_vector(), psi_ref, atol=2e-5), \
        "time-evolved wavefunction obtained by single-site TDVP time evolution must match reference"
    assert np.allclose(psi2.to_vector(), psi_ref, atol=1e-10), \
        "time-evolved wavefunction obtained by two-site TDVP time evolution must match reference"


def test_tdvp_symmetry():

    rng = np.random.default_rng()

    # number of lattice sites
    nsites = 10

    # real-time evolution
    dt = 0.5j

    # construct matrix product operator representation of Heisenberg Hamiltonian
    J =  4.0/3
    D =  5.0/13
    h = -2.0/7
    h_mpo = ptn.heisenberg_xxz_1d_mpo(nsites, J, D, h)
    h_mpo.zero_qnumbers()

    # quantum numbers not used here; set them to zero
    qbonds = [np.array([0])]
    for _ in range(nsites - 1):
        qbonds.append(np.zeros(5, dtype=int))
    qbonds.append(np.array([0]))

    # initial wavefunction as MPS with random entries
    psi = ptn.MPS(h_mpo.qsite, qbonds, fill="random", rng=rng)
    psi.orthonormalize(mode="left")

    psi_ref = psi.to_vector()

    # evolve forward and then backward in time;
    # should arrive at initial state since integration method is symmetric
    psi1 = copy.deepcopy(psi)
    ptn.tdvp_singlesite(h_mpo, psi1,  dt, 1, numiter_lanczos=10)
    ptn.tdvp_singlesite(h_mpo, psi1, -dt, 1, numiter_lanczos=10)
    psi2 = copy.deepcopy(psi)
    ptn.tdvp_twosite(h_mpo, psi2,  dt, 1, numiter_lanczos=10, tol_split=1e-10)
    ptn.tdvp_twosite(h_mpo, psi2, -dt, 1, numiter_lanczos=10, tol_split=1e-10)

    # compare
    assert np.allclose(psi1.to_vector(), psi_ref, atol=1e-10)
    # larger deviation for two-site TDVP presumably due to varying bond dimensions
    assert np.allclose(psi2.to_vector(), psi_ref, atol=1e-6)
