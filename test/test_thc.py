import unittest
import numpy as np
from scipy import sparse
import pytenet as ptn
from pytenet.hamiltonian import _encode_quantum_number_pair
from test_hamiltonian import construct_spin_molecular_hamiltonian


class TestTensorHypercontraction(unittest.TestCase):

    def test_thc_spin_molecular_hamiltonian(self):

        rng = np.random.default_rng()

        # number of spin-endowed lattice sites
        nsites = 4
        # THC rank
        thc_rank = 7

        H = _generate_random_thc_hamiltonian(nsites, thc_rank, rng)

        # reference Hamiltonian
        vint_thc = np.einsum(
            H.thc_kernel, (4, 5),
            H.thc_transform, (0, 4),
            H.thc_transform, (1, 4),
            H.thc_transform, (2, 5),
            H.thc_transform, (3, 5),
            (0, 1, 2, 3))
        # switch to different ordering convention for interaction term
        Href = construct_spin_molecular_hamiltonian(H.tkin + 0.5*np.trace(vint_thc, axis1=1, axis2=2),
                                                    vint_thc.transpose((0, 2, 1, 3)))

        # compare matrix representations
        self.assertTrue(np.allclose(H.as_matrix(sparse_format=False), Href.todense()),
            msg='matrix representation of THC and reference Hamiltonian must match')
        self.assertAlmostEqual(sparse.linalg.norm(H.as_matrix(sparse_format=True) - Href), 0., delta=1e-12,
            msg='matrix representation of THC and reference Hamiltonian must match')


    def test_apply_thc_spin_molecular_hamiltonian(self):

        rng = np.random.default_rng()

        # number of spin-endowed lattice sites
        nsites = 5
        # THC rank
        thc_rank = 8

        H = _generate_random_thc_hamiltonian(nsites, thc_rank, rng)

        # create a random matrix product state
        # physical particle number and spin quantum numbers (encoded as single integer)
        qN = [0,  1,  1,  2]
        qS = [0, -1,  1,  0]
        qd = [_encode_quantum_number_pair(q[0], q[1]) for q in zip(qN, qS)]
        D = [1, 15, 22, 33, 13, 1]
        # ensure that the MPS does not represent a zero vector
        while True:
            qD = [[_encode_quantum_number_pair(rng.integers(-1, 2), rng.integers(-1, 2))
                   for _ in range(Di)]
                   for Di in D]
            psi = ptn.MPS(qd, qD, fill='random', rng=rng)
            if ptn.norm(psi) > 0:
                break
        # rescale to achieve norm of order 1
        for i in range(psi.nsites):
            psi.A[i] *= 5

        Hpsi_ref = H.as_matrix(sparse_format=True) @ psi.as_vector()

        for tol in (0, 1e-3):
            Hpsi = ptn.apply_thc_spin_molecular_hamiltonian(H, psi, tol)

            # compare H |psi> vectors
            ctol = (1e-13 if tol == 0 else 0.005)
            self.assertTrue(np.allclose(Hpsi.as_vector(), Hpsi_ref, atol=ctol, rtol=ctol))


def _generate_random_thc_hamiltonian(nsites: int, thc_rank: int, rng: np.random.Generator):
    """
    Generate a spin molecular Hamiltonian using the tensor hypercontraction
    representation with random coefficients.
    """
    # kinetic coefficients can be complex in general
    tkin = 0.4 * ptn.crandn((nsites, nsites), rng)
    tkin = 0.5 * (tkin + tkin.conj().T)

    thc_kernel = rng.normal(size=(thc_rank, thc_rank))
    thc_kernel = 0.5 * (thc_kernel + thc_kernel.T)
    thc_transform = 0.4 * rng.normal(size=(nsites, thc_rank))

    return ptn.THCSpinMolecularHamiltonian(tkin, thc_kernel, thc_transform)


if __name__ == '__main__':
    unittest.main()
