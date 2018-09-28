import unittest
import numpy as np
from scipy.linalg import expm
import pytenet as ptn


class TestEvolution(unittest.TestCase):

    def test_single_site(self):

        # number of lattice sites
        L = 10

        # time step can have both real and imaginary parts;
        # for real-time evolution use purely imaginary dt!
        dt = 0.02 - 0.05j
        # number of steps
        numsteps = 12

        # construct matrix product operator representation of Heisenberg Hamiltonian
        J =  4.0/3
        D =  5.0/13
        h = -2.0/7
        mpoH = ptn.heisenberg_XXZ_MPO(L, J, D, h)

        # fix total spin quantum number of wavefunction (trailing virtual bond)
        spin_tot = 2

        # enumerate all possible virtual bond quantum numbers (including multiplicities);
        # will be implicitly reduced by orthonormalization steps below
        qD = [np.array([0])]
        for i in range(L-1):
            qD.append(np.sort(np.array([q + mpoH.qd for q in qD[-1]]).reshape(-1)))
        qD.append(np.array([2*spin_tot]))

        # initial wavefunction as MPS with random entries
        psi = ptn.MPS(mpoH.qd, qD, fill='random')
        psi.orthonormalize(mode='left')
        psi.orthonormalize(mode='right')
        # effectively clamp virtual bond dimension of initial state
        Dinit = 8
        for i in range(L):
            psi.A[i][:, Dinit:, :] = 0
            psi.A[i][:, :, Dinit:] = 0
        # orthonormalize again
        psi.orthonormalize(mode='left')

        self.assertEqual(psi.qD[-1][0], 2*spin_tot,
            msg='trailing bond quantum number must not change during orthonormalization')

        # total spin operator as MPO
        Sztot = ptn.local_opchains_to_MPO(mpoH.qd, L, [ptn.OpChain([np.diag([0.5, -0.5])], [])])

        # explicity compute average spin
        spin_avr = ptn.operator_average(psi, Sztot)
        self.assertAlmostEqual(abs(spin_avr - spin_tot), 0, delta=1e-14,
            msg='average spin must be equal to prescribed value')

        # reference time evolution
        psi_ref = np.dot(expm(-dt*numsteps * mpoH.as_matrix()), psi.as_vector())

        # run TDVP time evolution
        ptn.integrate_local_singlesite(mpoH, psi, dt, numsteps, numiter_lanczos=5)

        # compare time-evolved wavefunctions
        self.assertAlmostEqual(np.linalg.norm(psi.as_vector() - psi_ref), 0, delta=2e-5,
            msg='time-evolved wavefunction obtained by single-site MPS time evolution must match reference')


if __name__ == '__main__':
    unittest.main()
