import unittest
import numpy as np
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
        mpoH = ptn.ising_MPO(L, J, h, g)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()
        # reference Hamiltonian
        Href = np.load('test_hamiltonian_ising.npz')['H']
        # compare
        self.assertAlmostEqual(np.linalg.norm(H - Href), 0., delta=1e-12,
            msg='matrix representation of MPO and reference Hamiltonian must match')

    def test_heisenberg_XXZ(self):
        # Hamiltonian parameters
        J = 14.0/25
        D = 13.0/8
        h =  2.0/7
        # number of lattice sites
        L = 7
        # construct MPO
        mpoH = ptn.heisenberg_XXZ_MPO(L, J, D, h)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()
        # reference Hamiltonian
        Href = np.load('test_hamiltonian_heisenberg_XXZ.npz')['H']
        # compare
        self.assertAlmostEqual(np.linalg.norm(H - Href), 0., delta=1e-12,
            msg='matrix representation of MPO and reference Hamiltonian must match')

    def test_heisenberg_XXZ_spin1(self):
        # Hamiltonian parameters
        J =  1.2
        D = -0.9
        h =  1.0/7
        # number of lattice sites
        L = 6
        # construct MPO
        mpoH = ptn.heisenberg_XXZ_spin1_MPO(L, J, D, h)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()
        # reference Hamiltonian
        Href = np.load('test_hamiltonian_heisenberg_XXZ_spin1.npz')['H']
        # compare
        self.assertAlmostEqual(np.linalg.norm(H - Href), 0., delta=1e-12,
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
        mpoH = ptn.bose_hubbard_MPO(d, L, t, U, mu)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()
        # reference Hamiltonian
        Href = np.load('test_hamiltonian_bose_hubbard.npz')['H']
        # compare
        self.assertAlmostEqual(np.linalg.norm(H - Href), 0., delta=1e-12,
            msg='matrix representation of MPO and reference Hamiltonian must match')

    def test_fermi_hubbard(self):
        # number of lattice sites
        L = 5
        # Hamiltonian parameters
        t  = 1.2
        U  = 2.7
        mu = 0.3
        # construct MPO
        mpoH = ptn.fermi_hubbard_MPO(L, t, U, mu)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()
        # reference Hamiltonian
        Href = np.load('test_hamiltonian_fermi_hubbard.npz')['H']
        # compare
        self.assertAlmostEqual(np.linalg.norm(H - Href), 0., delta=1e-12,
            msg='matrix representation of MPO and reference Hamiltonian must match')


if __name__ == '__main__':
    unittest.main()
