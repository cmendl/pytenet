import unittest
import numpy as np
import pytenet as ptn


class TestOperation(unittest.TestCase):

    def test_vdot(self):

        rng = np.random.default_rng()

        # physical dimension
        d = 4

        # create random matrix product states
        psi = ptn.MPS(np.zeros(d, dtype=int), [np.zeros(Di, dtype=int) for Di in [1, 3, 9, 13, 4, 1]], fill='random', rng=rng)
        chi = ptn.MPS(np.zeros(d, dtype=int), [np.zeros(Di, dtype=int) for Di in [1, 4, 7, 8,  2, 1]], fill='random', rng=rng)

        # calculate dot product <chi | psi>
        s = ptn.vdot(chi, psi)

        # reference value
        s_ref = np.vdot(chi.as_vector(), psi.as_vector())

        # relative error
        err = abs(s - s_ref) / max(abs(s_ref), 1e-12)
        self.assertAlmostEqual(err, 0., delta=1e-12, msg='dot product must match reference value')


    def test_norm(self):

        rng = np.random.default_rng()

        # physical dimension
        d = 3

        # create random matrix product state
        psi = ptn.MPS(np.zeros(d, dtype=int), [np.zeros(Di, dtype=int) for Di in [1, 3, 5, 8, 7, 2, 1]], fill='random', rng=rng)

        # calculate the norm of psi using the MPS representation
        n = ptn.norm(psi)

        # reference value
        n_ref = np.linalg.norm(psi.as_vector())

        # relative error
        err = abs(n - n_ref) / max(abs(n_ref), 1e-12)
        self.assertAlmostEqual(err, 0., delta=1e-12, msg='wavefunction norm must match reference value')


    def test_operator_average(self):

        rng = np.random.default_rng()

        # physical dimension
        d = 3

        # physical quantum numbers
        qd = rng.integers(-1, 2, size=d)

        # create random matrix product state
        D = [1, 7, 26, 19, 25, 8, 1]
        psi = ptn.MPS(qd, [rng.integers(-1, 2, size=Di) for Di in D], fill='random', rng=rng)
        # rescale to achieve norm of order 1
        for i in range(psi.nsites):
            psi.A[i] *= 5

        # create random matrix product operator
        D = [1, 5, 16, 14, 17, 4, 1]
        # set bond quantum numbers to zero since otherwise,
        # sparsity pattern often leads to <psi | op | psi> = 0
        op = ptn.MPO(qd, [np.zeros(Di, dtype=int) for Di in D], fill='random', rng=rng)

        # calculate average (expectation value) <psi | op | psi>
        avr = ptn.operator_average(psi, op)

        # reference value based on full Fock space representation
        x = psi.as_vector()
        avr_ref = np.vdot(x, op.as_matrix() @ x)

        # relative error
        err = abs(avr - avr_ref) / max(abs(avr_ref), 1e-12)
        self.assertAlmostEqual(err, 0., delta=1e-12, msg='operator average must match reference value')

        # represent a density matrix as random matrix product operator
        # (Hermition property not relevant here)
        D = [1, 3, 7, 6, 11, 5, 1]
        # set bond quantum numbers to zero since otherwise,
        # sparsity pattern often leads to <psi | op | psi> = 0
        rho = ptn.MPO(qd, [np.zeros(Di, dtype=int) for Di in D], fill='random', rng=rng)

        # calculate average (expectation value) tr[op rho]
        avr = ptn.operator_density_average(rho, op)

        # reference value based on full Fock space representation
        avr_ref = np.trace(op.as_matrix() @ rho.as_matrix())

        # relative error
        err = abs(avr - avr_ref) / max(abs(avr_ref), 1e-12)
        self.assertAlmostEqual(err, 0., delta=1e-12, msg='operator average must match reference value')


if __name__ == '__main__':
    unittest.main()
