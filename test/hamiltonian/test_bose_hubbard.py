import numpy as np
from scipy import sparse
import pytenet as ptn


def test_bose_hubbard_1d_mpo():

    # Hamiltonian parameters
    t  = 0.7
    u  = 3.2
    mu = 1.3

    # number of lattice sites
    for nsites in range(2, 7):
        # physical dimension per site (maximal occupancy is d - 1)
        for d in range(2, 5):

            # construct the MPO
            h_mpo = ptn.bose_hubbard_1d_mpo(nsites, d, t, u, mu)
            # matrix representation, for comparison with reference
            h_mat = h_mpo.to_matrix()

            # reference Hamiltonian
            h_ref = construct_bose_hubbard_1d_hamiltonian(nsites, d, t, u, mu)

            # compare
            assert np.allclose(h_mat, h_ref.todense()), \
                "matrix representation of MPO and reference Hamiltonian must match"


def construct_bose_hubbard_1d_hamiltonian(nsites: int, d: int, t: float, u: float, mu: float):
    """
    Construct the Bose-Hubbard Hamiltonian
    with nearest-neighbor hopping on a one-dimensional lattice as a sparse matrix.
    """
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    # kinetic hopping terms, interaction terms and external field
    tkin = -t * (sparse.kron(b_dag, b_ann) + sparse.kron(b_ann, b_dag))
    hint = 0.5 * u * (numop @ (numop - np.identity(d))) - mu * numop
    hamiltonian = \
        sum(sparse.kron(sparse.identity(d**j),
            sparse.kron(tkin,
                        sparse.identity(d**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(d**j),
            sparse.kron(hint,
                        sparse.identity(d**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian
