import unittest
import numpy as np
import sys
sys.path.append('../pytenet/')
from bond_ops import qr
from qnumber import qnumber_outer_sum, is_qsparse


class TestBondOps(unittest.TestCase):

    def test_qr(self):

        A = randn_complex((23, 15))

        # fictitious quantum numbers
        q0 = np.random.randint(-2, 3, size=A.shape[0])
        q1 = np.random.randint(-2, 3, size=A.shape[1])

        # enforce block sparsity structure dictated by quantum numbers
        mask = qnumber_outer_sum([q0, -q1])
        A = np.where(mask == 0, A, 0)

        # perform QR decomposition, taking quantum numbers into account
        Q, R, qinterm = qr(A, q0, q1)

        self.assertAlmostEqual(np.linalg.norm(np.dot(Q, R) - A), 0.,
                               delta=1e-14, msg='Q.R must match A matrix')

        self.assertAlmostEqual(np.linalg.norm(np.dot(Q.conj().T, Q) - np.identity(Q.shape[1])), 0.,
                               delta=1e-14, msg='columns of Q matrix must be orthonormalized')

        self.assertTrue(is_qsparse(Q, [q0, -qinterm]),
                        msg='sparsity pattern of Q matrix must match quantum numbers')
        self.assertTrue(is_qsparse(R, [qinterm, -q1]),
                        msg='sparsity pattern of R matrix must match quantum numbers')


def randn_complex(size):
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)


if __name__ == '__main__':
    unittest.main()
