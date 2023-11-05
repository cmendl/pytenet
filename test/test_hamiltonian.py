import unittest
import numpy as np
from scipy import sparse
import pytenet as ptn


class TestHamiltonian(unittest.TestCase):

    def test_ising(self):

        # Hamiltonian parameters
        J =  5.0/11
        h = -2.0/7
        g = 13.0/8
        # number of lattice sites
        L = 7

        # construct MPO
        mpoH = ptn.ising_mpo(L, J, h, g)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_ising_hamiltonian(L, J, h, g)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_heisenberg_xxz(self):

        # Hamiltonian parameters
        J = 14.0/25
        D = 13.0/8
        h =  2.0/7
        # number of lattice sites
        L = 7

        # construct MPO
        mpoH = ptn.heisenberg_xxz_mpo(L, J, D, h)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_heisenberg_xxz_hamiltonian(L, J, D, h)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_heisenberg_xxz_spin1(self):

        # Hamiltonian parameters
        J =  1.2
        D = -0.9
        h =  1.0/7
        # number of lattice sites
        L = 6

        # construct MPO
        mpoH = ptn.heisenberg_xxz_spin1_mpo(L, J, D, h)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_heisenberg_xxz_spin1_hamiltonian(L, J, D, h)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_bose_hubbard(self):

        # physical dimension per site (maximal occupancy is d - 1)
        d = 4
        # number of lattice sites
        L = 5
        # Hamiltonian parameters
        t  = 0.7
        U  = 3.2
        mu = 1.3

        # construct MPO
        mpoH = ptn.bose_hubbard_mpo(d, L, t, U, mu)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_bose_hubbard_hamiltonian(d, L, t, U, mu)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_fermi_hubbard(self):

        # number of lattice sites
        L = 5
        # Hamiltonian parameters
        t  = 1.2
        U  = 2.7
        mu = 0.3

        # construct MPO
        mpoH = ptn.fermi_hubbard_mpo(L, t, U, mu)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_fermi_hubbard_hamiltonian(L, t, U, mu)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


def construct_ising_hamiltonian(L: int, J: float, h: float, g: float):
    """
    Construct Ising Hamiltonian `sum J sz sz + h sz + g sx`
    on a 1D lattice as sparse matrix.
    """
    # Pauli-X and Z matrices
    sigma_x = sparse.csr_matrix([[0., 1.], [1.,  0.]])
    sigma_z = sparse.csr_matrix([[1., 0.], [0., -1.]])
    H = sparse.csr_matrix((2**L, 2**L), dtype=float)
    # interaction terms
    hint = sparse.kron(sigma_z, sigma_z)
    for j in range(L - 1):
        H += J * sparse.kron(sparse.identity(2**j),
                 sparse.kron(hint,
                             sparse.identity(2**(L-j-2))))
    # external field
    for j in range(L):
        H += sparse.kron(sparse.identity(2**j),
             sparse.kron(h*sigma_z + g*sigma_x,
                         sparse.identity(2**(L-j-1))))
    return H


def construct_heisenberg_xxz_hamiltonian(L: int, J: float, D: float, h: float):
    """
    Construct XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a 1D lattice as sparse matrix.
    """
    # spin operators
    Sup = np.array([[0.,  1.], [0.,  0. ]])
    Sdn = np.array([[0.,  0.], [1.,  0. ]])
    Sz  = np.array([[0.5, 0.], [0., -0.5]])
    H = sparse.csr_matrix((2**L, 2**L), dtype=float)
    # interaction terms
    hint = J * 0.5 * (sparse.kron(Sup, Sdn) + sparse.kron(Sdn, Sup)) + D * sparse.kron(Sz, Sz)
    for j in range(L - 1):
        H += sparse.kron(sparse.identity(2**j),
             sparse.kron(hint,
                         sparse.identity(2**(L-j-2))))
    # external field
    for j in range(L):
        H -= sparse.kron(sparse.identity(2**j),
             sparse.kron(h*Sz,
                         sparse.identity(2**(L-j-1))))
    return H


def construct_heisenberg_xxz_spin1_hamiltonian(L: int, J: float, D: float, h: float):
    """
    Construct spin-1 XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a 1D lattice as sparse matrix.
    """
    # spin operators
    sq2 = np.sqrt(2.)
    Sup = np.array([[0.,  sq2, 0.], [0.,  0.,  sq2], [0.,  0.,  0.]])
    Sdn = np.array([[0.,  0.,  0.], [sq2, 0.,  0. ], [0.,  sq2, 0.]])
    Sz  = np.array([[1.,  0.,  0.], [0.,  0.,  0. ], [0.,  0., -1.]])
    H = sparse.csr_matrix((3**L, 3**L), dtype=float)
    # interaction terms
    hint = J * 0.5 * (sparse.kron(Sup, Sdn) + sparse.kron(Sdn, Sup)) + D * sparse.kron(Sz, Sz)
    for j in range(L - 1):
        H += sparse.kron(sparse.identity(3**j),
             sparse.kron(hint,
                         sparse.identity(3**(L-j-2))))
    # external field
    for j in range(L):
        H -= sparse.kron(sparse.identity(3**j),
             sparse.kron(h*Sz,
                         sparse.identity(3**(L-j-1))))
    return H


def construct_bose_hubbard_hamiltonian(d: int, L: int, t: float, U: float, mu: float):
    """
    Construct Bose-Hubbard Hamiltonian
    with nearest-neighbor hopping on a 1D lattice as sparse matrix.
    """
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    H = sparse.csr_matrix((d**L, d**L), dtype=float)
    # interaction terms
    hint = -t * (sparse.kron(b_dag, b_ann) + sparse.kron(b_ann, b_dag))
    for j in range(L - 1):
        H += sparse.kron(sparse.identity(d**j),
             sparse.kron(hint,
                         sparse.identity(d**(L-j-2))))
    # external field
    for j in range(L):
        H += sparse.kron(sparse.identity(d**j),
             sparse.kron(0.5*U*(numop @ (numop - np.identity(d))) - mu*numop,
                         sparse.identity(d**(L-j-1))))
    return H


def generate_fermi_operators(L: int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `L` sites (or modes).
    """
    I = sparse.identity(2)
    Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    for i in range(L):
        c = sparse.identity(1)
        for j in range(L):
            if j < i:
                c = sparse.kron(c, I)
            elif j == i:
                c = sparse.kron(c, U)
            else:
                c = sparse.kron(c, Z)
        clist.append(c)
    # corresponding annihilation operators
    alist = [c.conj().T for c in clist]
    return (clist, alist)


def construct_fermi_hubbard_hamiltonian(L: int, t: float, U: float, mu: float):
    """
    Construct Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping on a 1D lattice as sparse matrix.
    """
    clist, alist = generate_fermi_operators(2*L)
    nlist = [c @ a for c, a in zip(clist, alist)]
    H = sparse.csr_matrix((4**L, 4**L), dtype=float)
    # kinetic hopping terms
    for j in range(2*L - 2):
        H -= t * (clist[j] @ alist[j+2] + clist[j+2] @ alist[j])
    # interaction U (n_up-1/2) (n_dn-1/2) and number operator - mu (n_up + n_dn)
    for j in range(0, 2*L, 2):
        H += (U * (nlist[j] - 0.5*sparse.identity(4**L)) @ (nlist[j+1] - 0.5*sparse.identity(4**L))
              - mu * (nlist[j] + nlist[j+1]))
    H.eliminate_zeros()
    return H


if __name__ == '__main__':
    unittest.main()
