import unittest
import numpy as np
import sys
sys.path.append('../pytenet/')
from mpo import MPO
from opchain import OpChain
from qnumber import is_qsparse, qnumber_outer_sum


class TestMPO(unittest.TestCase):

    def test_orthonormalization(self):

        # create random matrix product operator
        d = 4
        D = [1, 10, 13, 14, 7, 1]
        mpo0 = MPO(np.random.randint(-2, 3, size=d), qD=[np.random.randint(-2, 3, size=Di) for Di in D], fill='random')

        self.assertEqual(mpo0.bond_dims, D, msg='virtual bond dimensions')

        # density matrix on full Hilbert space
        rho = mpo0.as_matrix()

        # performing left-orthonormalization...
        cL = mpo0.orthonormalize(mode='left')

        self.assertLessEqual(mpo0.bond_dims[1], d**2,
            msg='virtual bond dimension can only increase by factor of "d^2" per site')

        for i in range(mpo0.nsites):
            self.assertTrue(is_qsparse(mpo0.A[i], [mpo0.qd, -mpo0.qd, mpo0.qD[i], -mpo0.qD[i+1]]),
                            msg='sparsity pattern of MPO tensors must match quantum numbers')

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

        for i in range(mpo0.nsites):
            self.assertTrue(is_qsparse(mpo0.A[i], [mpo0.qd, -mpo0.qd, mpo0.qD[i], -mpo0.qD[i+1]]),
                            msg='sparsity pattern of MPO tensors must match quantum numbers')

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

        # physical quantum numbers
        qd = np.random.randint(-2, 3, size=d)

        # fictitious operator chains
        opchains = []
        n = np.random.randint(20)
        for _ in range(n):
            istart = np.random.randint(L)
            length = np.random.randint(1, L - istart + 1)
            oplist = [randn_complex((d, d)) for _ in range(length)]
            qD = np.random.randint(-2, 3, size=length-1)
            # enforce sparsity structure dictated by quantum numbers
            qDpad = np.pad(qD, 1, mode='constant')
            for i in range(length):
                mask = qnumber_outer_sum([qd + qDpad[i], -(qd + qDpad[i+1])])
                oplist[i] = np.where(mask == 0, oplist[i], 0)
            opchains.append(OpChain(oplist, qD, istart))

        # construct MPO representation corresponding to operator chains
        mpo0 = MPO(qd, L=L, opchains=opchains)

        for i in range(mpo0.nsites):
            self.assertTrue(is_qsparse(mpo0.A[i], [mpo0.qd, -mpo0.qd, mpo0.qD[i], -mpo0.qD[i+1]]),
                            msg='sparsity pattern of MPO tensors must match quantum numbers')

        # construct full Hamiltonian from operator chains, as reference
        Href = sum([opc.as_matrix(d, L) for opc in opchains])

        # compare
        self.assertAlmostEqual(np.linalg.norm(mpo0.as_matrix() - Href), 0., delta=1e-10,
            msg='full merging of MPO must be equal to matrix representation of operator chains')


def randn_complex(size):
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)


if __name__ == '__main__':
    unittest.main()
