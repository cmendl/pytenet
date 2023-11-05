import unittest
import numpy as np
import pytenet as ptn


class TestOpChain(unittest.TestCase):

    def test_as_matrix(self):

        # physical quantum numbers
        qd = np.array([0, -1, 1])

        opids = [ 6,  3,  5,  4]
        qnums = [ 0, -1,  0,  2,  0]
        coeff = 0.7
        chain = ptn.OpChain(opids, qnums, coeff, 1)
        self.assertEqual(chain.length, len(opids))

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in range(3, 7) }
        # enforce sparsity pattern of local operators in the chain according to quantum numbers.
        for i, opid in enumerate(opids):
            mask = ptn.qnumber_outer_sum([qd, -qd, [qnums[i]], [-qnums[i+1]]])[:, :, 0, 0]
            opmap[opid] = np.where(mask == 0, opmap[opid], 0)

        # reference matrix representation
        mat_ref = coeff * np.identity(1)
        for opid in opids:
            mat_ref = np.kron(mat_ref, opmap[opid])

        # compare
        self.assertTrue(np.allclose(chain.as_matrix(opmap), mat_ref))


if __name__ == '__main__':
    unittest.main()
