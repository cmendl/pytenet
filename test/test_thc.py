import unittest
import numpy as np
from scipy import sparse
import pytenet as ptn
from test_hamiltonian import construct_spin_molecular_hamiltonian


class TestTensorHypercontraction(unittest.TestCase):

    def test_thc_spin_molecular_hamiltonian(self):

        rng = np.random.default_rng()

        # number of spin-endowed lattice sites
        L = 4
        # THC rank
        N = 7

        # kinetic coefficients can be complex in general
        tkin = 0.4 * ptn.crandn((L, L), rng)
        tkin = 0.5 * (tkin + tkin.conj().T)
        thc_kernel = rng.normal(size=(N, N))
        thc_kernel = 0.5 * (thc_kernel + thc_kernel.T)
        thc_transform = 0.4 * rng.normal(size=(L, N))

        H = ptn.THCSpinMolecularHamiltonian(tkin, thc_kernel, thc_transform)

        # reference Hamiltonian
        vint_thc = np.einsum(
            thc_kernel, (4, 5),
            thc_transform, (0, 4),
            thc_transform, (1, 4),
            thc_transform, (2, 5),
            thc_transform, (3, 5),
            (0, 1, 2, 3))
        # switch to different ordering convention for interaction term
        Href = construct_spin_molecular_hamiltonian(tkin + 0.5*np.trace(vint_thc, axis1=1, axis2=2),
                                                    vint_thc.transpose((0, 2, 1, 3)))

        # compare matrix representations
        self.assertTrue(np.allclose(H.as_matrix(sparse_format=False), Href.todense()),
            msg='matrix representation of THC and reference Hamiltonian must match')
        self.assertAlmostEqual(sparse.linalg.norm(H.as_matrix(sparse_format=True) - Href), 0., delta=1e-12,
            msg='matrix representation of THC and reference Hamiltonian must match')


if __name__ == '__main__':
    unittest.main()
