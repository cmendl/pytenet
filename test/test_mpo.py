import unittest
import numpy as np
import sys
sys.path.append('../pytenet/')
from mpo import MPO
from opchain import OpChain


class TestMPO(unittest.TestCase):

    def test_random_init(self):

        # create random matrix product operator
        d = 3
        D = [1, 5, 4, 7, 1]
        mpo0 = MPO(d, D=D, fill='random')

        self.assertEqual(mpo0.bond_dims, D, 'virtual bond dimensions must agree')


    def test_orthonormalization(self):

        # create random matrix product operator
        d = 3
        D = [1, 10, 3, 4, 7, 1]
        mpo0 = MPO(d, D=D, fill='random')

        self.assertEqual(mpo0.bond_dims, D, msg='virtual bond dimensions')

        # density matrix on full Hilbert space
        rho = mpo0.as_matrix()

        # performing left-orthonormalization...
        cL = mpo0.orthonormalize(mode='left')

        self.assertLessEqual(mpo0.bond_dims[1], d**2,
            msg='virtual bond dimension can only increase by factor of "d^2" per site')

        rhoL = mpo0.as_matrix()
        # density matrix should now be normalized
        self.assertAlmostEqual(np.linalg.norm(rhoL, 'fro'), 1., delta=1e-12, msg='density matrix normalization')

        # density matrices before and after left-normalization must match
        # (up to normalization factor)
        self.assertAlmostEqual(np.linalg.norm(cL*rhoL - rho), 0., delta=1e-10,
                               msg='density matrices before and after left-normalization must match')

        # check left-orthonormalization
        for i in range(mpo0.nsites):
            s = mpo0.A[i].shape
            assert s[0] == d and s[1] == d
            Q = mpo0.A[i].reshape((s[0]*s[1]*s[2], s[3]))
            QH_Q = np.dot(Q.conj().T, Q)
            self.assertAlmostEqual(np.linalg.norm(QH_Q - np.identity(s[3])), 0., delta=1e-12,
                                   msg='left-orthonormalization')

        # performing right-orthonormalization...
        cR = mpo0.orthonormalize(mode='right')

        self.assertLessEqual(mpo0.bond_dims[-2], d**2,
            msg='virtual bond dimension can only increase by factor of "d^2" per site')

        self.assertAlmostEqual(abs(cR), 1., delta=1e-12,
            msg='normalization factor must have magnitude 1 due to previous left-orthonormalization')

        rhoR = mpo0.as_matrix()
        # density matrices must match
        self.assertAlmostEqual(np.linalg.norm(rhoL - cR*rhoR), 0., delta=1e-10,
                               msg='density matrices after left- and right-orthonormalization must match')

        # check right-orthonormalization
        for i in range(mpo0.nsites):
            s = mpo0.A[i].shape
            assert s[0] == d and s[1] == d
            Q = mpo0.A[i].transpose((0, 1, 3, 2)).reshape((s[0]*s[1]*s[3], s[2]))
            QH_Q = np.dot(Q.conj().T, Q)
            self.assertAlmostEqual(np.linalg.norm(QH_Q - np.identity(s[2])), 0., delta=1e-12,
                                   msg='right-orthonormalization')


    def test_from_opchains(self):

        # dimensions
        d = 4
        L = 5

        # fictitious operator chains
        opchains = []
        n = np.random.randint(20)
        for _ in range(n):
            istart = np.random.randint(L)
            length = np.random.randint(L - istart + 1)
            oplist = [np.random.normal(size=(d, d)) + 1j*np.random.normal(size=(d, d)) for _ in range(length)]
            opchains.append(OpChain(istart, oplist))

        # construct MPO representation corresponding to operator chains
        mpo0 = MPO(d, L=L, opchains=opchains)

        # construct full Hamiltonian from operator chains, as reference
        Href = sum([opc.as_matrix(d, L) for opc in opchains])

        # compare
        self.assertAlmostEqual(np.linalg.norm(mpo0.as_matrix() - Href), 0., delta=1e-10,
            msg='full merging of MPO must be equal to matrix representation of operator chains')


if __name__ == '__main__':
    unittest.main()
