import numpy as np
from scipy import sparse
import pytenet as ptn


def test_ising_1d_mpo():

    # Hamiltonian parameters
    J =  5.0/11
    h = -2.0/7
    g = 13.0/8
    # number of lattice sites
    nsites = 7

    # construct MPO
    h_mpo = ptn.ising_1d_mpo(nsites, J, h, g)
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()

    # reference Hamiltonian
    h_ref = construct_ising_1d_hamiltonian(nsites, J, h, g)

    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference Hamiltonian must match"


def construct_ising_1d_hamiltonian(nsites: int, J: float, h: float, g: float):
    """
    Construct the Ising Hamiltonian `sum J Z Z + h Z + g X`
    on a one-dimensional lattice as sparse matrix.
    """
    # Pauli-X and Z matrices
    sigma_x = sparse.csr_matrix([[0., 1.], [1.,  0.]])
    sigma_z = sparse.csr_matrix([[1., 0.], [0., -1.]])
    # interaction terms and external field
    hint = sparse.kron(sigma_z, sigma_z)
    hamiltonian = \
        sum(J * sparse.kron(sparse.identity(2**j),
                sparse.kron(hint,
                            sparse.identity(2**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(h*sigma_z + g*sigma_x,
                        sparse.identity(2**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian
