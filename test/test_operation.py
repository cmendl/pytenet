import unittest
import numpy as np
import sys
sys.path.append('../pytenet/')
from mps import MPS
from mpo import MPO
from operation import operator_average


class TestOperation(unittest.TestCase):

    def test_operator_average(self):

        # physical dimension
        d = 3

        # physical quantum numbers
        qd = np.random.randint(-1, 2, size=d)

        # create random matrix product state
        D = [1, 7, 26, 19, 25, 8, 1]
        psi = MPS(qd, [np.random.randint(-1, 2, size=Di) for Di in D], fill='random')
        # rescale to achieve norm of order 1
        for i in range(psi.nsites):
            psi.A[i] *= 5

        # create random matrix product operator
        D = [1, 5, 16, 14, 17, 4, 1]
        # set bond quantum numbers to zero since otherwise,
        # sparsity pattern often leads to <psi | op | psi> = 0
        op = MPO(qd, qD=[np.zeros(Di) for Di in D], fill='random')

        # calculate average (expectation value) <psi | op | psi>
        avr = operator_average(psi, op)

        # reference value based on full Fock space representation
        x = psi.as_vector()
        avr_ref = np.vdot(x, np.dot(op.as_matrix(), x))

        # relative error
        err = abs(avr - avr_ref) / max(abs(avr_ref), 1e-12)
        self.assertAlmostEqual(err, 0., delta=1e-12, msg='operator average must match reference value')


if __name__ == '__main__':
    unittest.main()
