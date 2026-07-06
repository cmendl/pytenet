import numpy as np
from test_molecular import construct_molecular_hamiltonian
import pytenet as ptn


def test_spin_molecular_hamiltonian_mpo():

    rng = np.random.default_rng()

    # number of spin-endowed lattice sites
    for nsites in range(2, 6):

        # Hamiltonian parameters
        tkin = ptn.crandn(2 * (nsites,), rng)
        vint = ptn.crandn(4 * (nsites,), rng)

        # reference Hamiltonian
        h_ref = construct_spin_molecular_hamiltonian(tkin, vint)

        for opt in (True, False):
            h_mpo = ptn.spin_molecular_hamiltonian_mpo(tkin, vint, opt)

            # compare matrix representations
            assert np.allclose(h_mpo.to_matrix(), h_ref.todense()), \
                "matrix representation of MPO and reference Hamiltonian must match"


def construct_spin_molecular_hamiltonian(tkin, vint):
    """
    Construct a molecular Hamiltonian for a spin orbital basis as sparse matrix.
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)

    nsites = tkin.shape[0]
    assert tkin.shape == 2 * (nsites,)
    assert vint.shape == 4 * (nsites,)

    # enlarge the single- and two-particle electron overlap integral tensors
    # from an orbital basis without spin to a spin orbital basis

    # single-particle integrals
    tkin_spin = np.kron(tkin, np.identity(2))

    # two-particle integrals
    tmp = np.zeros((2*nsites, nsites, 2*nsites, nsites), dtype=vint.dtype)
    for i in range(nsites):
        for j in range(nsites):
            tmp[:, i, :, j] = np.kron(vint[:, i, :, j], np.identity(2))
    vint_spin = np.zeros((2*nsites, 2*nsites, 2*nsites, 2*nsites), dtype=vint.dtype)
    for i in range(2*nsites):
        for j in range(2*nsites):
            vint_spin[i, :, j, :] = np.kron(tmp[i, :, j, :], np.identity(2))

    return construct_molecular_hamiltonian(tkin_spin, vint_spin)
