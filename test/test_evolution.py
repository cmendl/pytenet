import unittest
import numpy as np
from scipy.linalg import expm
import sys
sys.path.append('../pytenet/')
from mps import MPS
import hamiltonian
from evolution import integrate_local_singlesite


class TestEvolution(unittest.TestCase):

    def test_single_site(self):

        # number of lattice sites
        L = 10

        # physical dimension per lattice site
        d = 2

        # time step can have both real and imaginary parts;
        # for real-time evolution use purely imaginary dt!
        dt = 0.02 - 0.05j
        # number of steps
        numsteps = 12

        # construct matrix product operator representation of Heisenberg Hamiltonian
        J =  4.0/3
        D =  5.0/13
        h = -2.0/7
        mpoH = hamiltonian.heisenberg_XXZ_MPO(L, J, D, h)

        # maximum virtual bond dimension
        Dmax = 20

        # initial wavefunction as MPS with random entries
        D = np.minimum(np.minimum(d**np.arange(L + 1), d**(L - np.arange(L + 1))), Dmax)
        psi = MPS(d, D, fill='random')
        # effectively clamp virtual bond dimension of initial state
        Dinit = 3
        for i in range(L):
            psi.A[i][:, Dinit:, :] = 0
            psi.A[i][:, :, Dinit:] = 0
        psi.orthonormalize(mode='right')
        psi.orthonormalize(mode='left')

        # represent psi as vector for reference calculation
        psi_ref = np.dot(expm(-dt*numsteps * mpoH.as_matrix()), psi.as_vector())

        # run time evolution
        integrate_local_singlesite(mpoH, psi, dt, numsteps)

        # compare time-evolved wavefunctions
        self.assertAlmostEqual(np.linalg.norm(psi.as_vector() - psi_ref), 0, delta=2e-5,
            msg='time-evolved wavefunction obtained by single-site MPS time evolution must match reference')


if __name__ == '__main__':
    unittest.main()
