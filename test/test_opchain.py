import unittest
import numpy as np
import pytenet as ptn


class TestOpChain(unittest.TestCase):

    def test_as_matrix(self):

        rng = np.random.default_rng()

        # dimensions
        d = 3
        L = 6

        # create random operator chain
        oplist = [ptn.crandn((d, d), rng) for _ in range(2)]
        opchain = ptn.OpChain(oplist=oplist, qD=[0], istart=3)

        # matrix representation on full Hilbert space
        A = opchain.as_matrix(d, L)

        # check dimensions
        self.assertEqual(A.shape, (d**L, d**L), msg='matrix dimensions')

        # pad identities on the left and check if matrix representation still matches
        opchain.pad_identities_left(d)
        A1 = opchain.as_matrix(d, L)
        self.assertEqual(np.linalg.norm(A1 - A), 0,
            msg='matrix representation after padding identities on the left must remain unchanged')

        # pad identities on the right and check if matrix representation still matches
        opchain.pad_identities_right(d, L)
        self.assertEqual(opchain.length, L, msg='operator chain length')
        A2 = opchain.as_matrix(d, L)
        self.assertEqual(np.linalg.norm(A2 - A), 0,
            msg='matrix representation after padding identities on the right must remain unchanged')


if __name__ == '__main__':
    unittest.main()
