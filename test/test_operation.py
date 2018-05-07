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

        # create random matrix product state
        D = [1, 7, 6, 4, 5, 8, 1]
        psi = MPS(d, D, fill='random')

        # create random matrix product operator
        D = [1, 5, 6, 4, 7, 4, 1]
        op = MPO(d, D=D, fill='random')

        # calculate average (expectation value) <psi | op | psi>
        avr = operator_average(psi, op)

        # reference value based on full Fock space representation
        x = psi.as_vector()
        avr_ref = np.vdot(x, np.dot(op.as_matrix(), x))

        # relative error
        err = abs(avr - avr_ref) / abs(avr_ref)
        self.assertAlmostEqual(err, 0., delta=1e-12, msg='operator average must match reference value')


if __name__ == '__main__':
    unittest.main()
