import numpy as np
from scipy import sparse
import pytenet as ptn


def test_heisenberg_xxz_1d_mpo():

    # Hamiltonian parameters
    J = 14.0/25
    D = 13.0/8
    h =  2.0/7
    # number of lattice sites
    nsites = 7

    # construct MPO
    h_mpo = ptn.heisenberg_xxz_1d_mpo(nsites, J, D, h)
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()

    # reference Hamiltonian
    h_ref = construct_heisenberg_xxz_1d_hamiltonian(nsites, J, D, h)

    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference Hamiltonian must match"


def test_heisenberg_xxz_spin1_1d_mpo():

    # Hamiltonian parameters
    J =  1.2
    D = -0.9
    h =  1.0/7
    # number of lattice sites
    nsites = 6

    # construct MPO
    h_mpo = ptn.heisenberg_xxz_spin1_1d_mpo(nsites, J, D, h)
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()

    # reference Hamiltonian
    h_ref = construct_heisenberg_xxz_spin1_1d_hamiltonian(nsites, J, D, h)

    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference Hamiltonian must match"


def construct_heisenberg_xxz_1d_hamiltonian(nsites: int, J: float, D: float, h: float):
    """
    Construct the XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a one-dimensional lattice as sparse matrix.
    """
    # spin operators
    sup = np.array([[0.,  1.], [0.,  0. ]])
    sdn = np.array([[0.,  0.], [1.,  0. ]])
    sz  = np.array([[0.5, 0.], [0., -0.5]])
    # interaction terms and external field
    hint = J * 0.5 * (sparse.kron(sup, sdn) + sparse.kron(sdn, sup)) + D * sparse.kron(sz, sz)
    hamiltonian = \
        sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(hint,
                        sparse.identity(2**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(-h * sz,
                        sparse.identity(2**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_heisenberg_xxz_spin1_1d_hamiltonian(nsites: int, J: float, D: float, h: float):
    """
    Construct the spin-1 XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a one-dimensional lattice as sparse matrix.
    """
    # spin operators
    sq2 = np.sqrt(2.)
    sup = np.array([[0.,  sq2, 0.], [0.,  0.,  sq2], [0.,  0.,  0.]])
    sdn = np.array([[0.,  0.,  0.], [sq2, 0.,  0. ], [0.,  sq2, 0.]])
    sz  = np.array([[1.,  0.,  0.], [0.,  0.,  0. ], [0.,  0., -1.]])
    # interaction terms
    hint = J * 0.5 * (sparse.kron(sup, sdn) + sparse.kron(sdn, sup)) + D * sparse.kron(sz, sz)
    hamiltonian = \
        sum(sparse.kron(sparse.identity(3**j),
            sparse.kron(hint,
                        sparse.identity(3**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(3**j),
            sparse.kron(-h * sz,
                        sparse.identity(3**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian
