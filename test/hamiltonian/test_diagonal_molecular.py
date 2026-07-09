import numpy as np
from scipy.sparse.linalg import norm
from fermi_operators import construct_fermi_operators
from test_molecular import construct_molecular_hamiltonian
import pytenet as ptn


def test_diagonal_molecular_hamiltonian_mpo():

    rng = np.random.default_rng()

    # number of fermionic modes
    for nsites in range(2, 11):

        # Hamiltonian parameters
        tkin = ptn.crandn(2 * (nsites,), rng)
        vint = np.triu(ptn.crandn(2 * (nsites,), rng), k=1)

        # reference Hamiltonian
        h_ref = construct_diagonal_molecular_hamiltonian(tkin, vint)

        # alternative construction based on generic molecular Hamiltonian
        vint_full = np.zeros(4 * (nsites,), dtype=vint.dtype)
        for i in range(nsites):
            for j in range(i + 1, nsites):  # i < j
                vint_full[i, j, i, j] = 2 * vint[i, j]
        h_alt = construct_molecular_hamiltonian(tkin, vint_full)
        assert norm(h_alt - h_ref) < 1e-13, \
            "matrix representation of diagonal molecular Hamiltonian "\
            "must match generic construction"

        for opt in (True, False):
            h_mpo = ptn.diagonal_molecular_hamiltonian_mpo(tkin, vint, opt)

            # theoretically predicted virtual bond dimensions
            b_theo = []
            for i in range(nsites + 1):
                n = min(i, nsites - i)
                # identity chains
                if opt:
                    b1 = 2 if 1 < i < nsites - 1 else 1
                else:
                    # slightly sub-optimal
                    b1 = 2 if 1 <= i <= nsites - 1 else 1
                # a^{\dagger}_i, a_i and n_i chains, reaching from one boundary to the center
                b2 = 3 * n
                b_theo.append(b1 + b2)
            assert h_mpo.bond_dims == b_theo

            # compare matrix representations
            assert np.allclose(h_mpo.to_matrix(), h_ref.todense()), \
                "matrix representation of MPO and reference Hamiltonian must match"


def construct_diagonal_molecular_hamiltonian(tkin, vint):
    """
    Construct the molecular Hamiltonian with a diagonal interaction term as a sparse matrix.
    """
    nmodes = tkin.shape[0]
    assert tkin.shape == (nmodes, nmodes)
    assert vint.shape == (nmodes, nmodes)

    clist, alist, nlist = construct_fermi_operators(nmodes)

    # kinetic hopping and interaction terms
    hamiltonian = \
        sum(tkin[i, j] * (clist[i] @ alist[j])
            for i in range(nmodes)
            for j in range(nmodes)) + \
        sum(vint[i, j] * (nlist[i] @ nlist[j])
            for i in range(nmodes)
            for j in range(i + 1, nmodes))
    hamiltonian.eliminate_zeros()
    return hamiltonian
