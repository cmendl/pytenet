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
