import numpy as np
from scipy import sparse
from fermi_operators import construct_fermi_operators
import pytenet as ptn


def test_fermi_hubbard_1d_mpo():

    # number of lattice sites
    nsites = 5
    # Hamiltonian parameters
    t  = 1.2
    u  = 2.7
    mu = 0.3

    # construct MPO
    h_mpo = ptn.fermi_hubbard_1d_mpo(nsites, t, u, mu)
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()

    # reference Hamiltonian
    h_ref = construct_fermi_hubbard_1d_hamiltonian(nsites, t, u, mu)

    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference Hamiltonian must match"


def construct_fermi_hubbard_1d_hamiltonian(nsites: int, t: float, u: float, mu: float):
    """
    Construct the Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping on a one-dimensional lattice as sparse matrix.
    """
    clist, alist, nlist = construct_fermi_operators(2*nsites)
    # kinetic hopping terms and
    # interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn)
    hamiltonian = sum(-t * (clist[j] @ alist[j+2] + clist[j+2] @ alist[j])
                      for j in range(2*nsites - 2)) \
                + sum((u * (nlist[j]   - 0.5*sparse.identity(4**nsites)) \
                         @ (nlist[j+1] - 0.5*sparse.identity(4**nsites)) \
                       - mu * (nlist[j] + nlist[j+1]))
                      for j in range(0, 2*nsites, 2))
    hamiltonian.eliminate_zeros()
    return hamiltonian
