import numpy as np
from scipy.sparse.linalg import norm
from test_diagonal_molecular import construct_diagonal_molecular_hamiltonian
from test_spin_molecular import construct_spin_molecular_hamiltonian
import pytenet as ptn


def test_diagonal_spin_molecular_hamiltonian_mpo():

    rng = np.random.default_rng()

    # number of fermionic modes
    for nsites in range(2, 6):

        # Hamiltonian parameters
        tkin = ptn.crandn(2 * (nsites,), rng)
        # keep diagonal entries, corresponding to n_{i,\uparrow} n_{i,\downarrow} terms
        vint = np.triu(ptn.crandn(2 * (nsites,), rng))

        # reference Hamiltonian
        h_ref = construct_diagonal_spin_molecular_hamiltonian(tkin, vint)

        # alternative construction based on generic spin molecular Hamiltonian
        vint_full = np.zeros(4 * (nsites,), dtype=vint.dtype)
        for i in range(nsites):
            vint_full[i, i, i, i] = vint[i, i]
            for j in range(i + 1, nsites):  # i < j
                vint_full[i, j, i, j] = 2 * vint[i, j]
        h_alt = construct_spin_molecular_hamiltonian(tkin, vint_full)
        assert norm(h_alt - h_ref) < 1e-13, \
            "matrix representation of diagonal spin molecular Hamiltonian "\
            "must match generic construction"

        for opt in (True, False):
            h_mpo = ptn.diagonal_spin_molecular_hamiltonian_mpo(tkin, vint, opt)

            # theoretically predicted virtual bond dimensions
            b_theo = []
            for i in range(nsites + 1):
                n = min(i, nsites - i)
                # identity chains
                b1 = 2 if 1 <= i <= nsites - 1 else 1
                # a^{\dagger}_{i,\sigma}, a_{i,\sigma} and n_{i,\uparrow} + n_{i,\downarrow} chains,
                # reaching from one boundary to the center
                b2 = 5 * n
                b_theo.append(b1 + b2)
            assert h_mpo.bond_dims == b_theo

            # compare matrix representations
            assert np.allclose(h_mpo.to_matrix(), h_ref.todense()), \
                "matrix representation of MPO and reference Hamiltonian must match"


def construct_diagonal_spin_molecular_hamiltonian(tkin, vint):
    """
    Construct a molecular Hamiltonian for a spin orbital basis
    and a diagonal interaction term as a sparse matrix.
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)

    nsites = tkin.shape[0]
    assert tkin.shape == 2 * (nsites,)
    assert vint.shape == 2 * (nsites,)

    # enlarge the single- and two-particle electron overlap integral tensors
    # from an orbital basis without spin to a spin orbital basis
    # single-particle integrals
    tkin_spin = np.kron(tkin, np.identity(2))
    # two-particle integrals
    vint_spin = np.triu(np.kron(vint, np.ones((2, 2))), k=1)

    return construct_diagonal_molecular_hamiltonian(tkin_spin, vint_spin)
