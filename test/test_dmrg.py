import numpy as np
import pytenet as ptn


def test_dmrg_singlesite():

    rng = np.random.default_rng()

    # number of lattice sites
    nsites = 10

    # number of left and right sweeps
    numsweeps = 4

    # construct matrix product operator representation of Heisenberg Hamiltonian
    J =  4/5
    D =  8/3
    h = -2/7
    hamiltonian = ptn.heisenberg_xxz_mpo(nsites, J, D, h)

    # initial wavefunction as MPS with random entries
    qnum_sector = 0
    psi = ptn.MPS.construct_random(nsites, hamiltonian.qsite, qnum_sector,
                                   max_vdim=32, dtype="real", rng=rng)

    en_min = ptn.dmrg_singlesite(hamiltonian, psi, numsweeps)
    # value after last iteration
    e0 = en_min[-1]

    # reference spectrum and wavefunctions
    en_ref, v_ref = np.linalg.eigh(hamiltonian.to_matrix())

    # compare ground state energy
    assert abs(e0 - en_ref[0]) < 1e-13, \
        "ground state energy obtained by single-site optimization must match reference"

    # MPS must indeed be an eigenvector
    psi_vec = psi.to_vector()
    assert np.allclose(hamiltonian.to_matrix() @ psi_vec, e0*psi_vec, atol=1e-7)

    # compare ground state wavefunction
    # multiply by phase factor to match (real-valued) reference wavefunction
    i = np.argmax(abs(psi_vec))
    z = psi_vec[i]
    psi_vec *= z.conj() / abs(z)
    if v_ref[i, 0] < 0:
        psi_vec = -psi_vec
    assert np.allclose(psi_vec, v_ref[:, 0], atol=1e-7), \
        "ground state wavefunction obtained by single-site optimization must match reference"


def test_dmrg_twosite():

    rng = np.random.default_rng()

    # number of lattice sites
    nsites = 10

    # number of left and right sweeps
    numsweeps = 2

    # construct matrix product operator representation of Heisenberg Hamiltonian
    J = -4/3
    D =  2/8
    h =  5/7
    hamiltonian = ptn.heisenberg_xxz_mpo(nsites, J, D, h)

    # initial wavefunction as MPS with random entries;
    # initial maximum bond dimension needs to be increased by DMRG algorithm
    qnum_sector = 4  # quantum number sector of ground state
    psi = ptn.MPS.construct_random(nsites, hamiltonian.qsite, qnum_sector, max_vdim=18, dtype="real", rng=rng)

    en_min = ptn.dmrg_twosite(hamiltonian, psi, numsweeps)
    # value after last iteration
    e0 = en_min[-1]

    # reference spectrum and wavefunctions
    en_ref, v_ref = np.linalg.eigh(hamiltonian.to_matrix())

    # compare ground state energy
    assert abs(e0 - en_ref[0]) < 1e-13, \
        "ground state energy obtained by two-site optimization must match reference"

    # MPS must indeed be an eigenvector
    psi_vec = psi.to_vector()
    assert np.allclose(hamiltonian.to_matrix() @ psi_vec, e0*psi_vec)

    # compare ground state wavefunction
    # multiply by sign factor to match (real-valued) reference wavefunction
    i = np.argmax(abs(psi_vec))
    if v_ref[i, 0] * psi_vec[i] < 0:
        psi_vec = -psi_vec
    assert np.allclose(psi_vec, v_ref[:, 0]), \
        "ground state wavefunction obtained by two-site optimization must match reference"
