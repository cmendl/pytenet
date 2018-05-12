import unittest
import numpy as np
from scipy.linalg import eigh_tridiagonal
import sys
sys.path.append('../pytenet/')
from krylov import lanczos_iteration


class TestKrylov(unittest.TestCase):

    def test_lanczos_iteration(self):

        n = 256
        numiter = 24

        # random Hermitian matrix
        A = (np.random.normal(size=(n, n)) + 1j*np.random.normal(size=(n, n))) / np.sqrt(n)
        A = 0.5 * (A + A.conj().T)

        # reference eigenvalues
        w_ref = np.linalg.eigvalsh(A)

        # random complex starting vector
        vstart = np.random.normal(size=n) + 1j*np.random.normal(size=n)

        # simply use A as linear transformation
        alpha, beta, V = lanczos_iteration(lambda v: np.dot(A, v), vstart, numiter)

        # check orthogonality of V
        self.assertAlmostEqual(np.linalg.norm(np.dot(V.T.conj(), V) - np.identity(numiter)), 0., delta=1e-12,
                               msg='matrix of Lanczos vectors must be orthonormalized')

        # diagonalize Hessenberg matrix
        w_hess, u_hess = eigh_tridiagonal(alpha, beta)

        # compare lowest eigenvalue
        self.assertAlmostEqual(w_hess[0], w_ref[0], delta=0.005,
                               msg='lowest Lanczos eigenvalue should approximately match reference')

        # compute Ritz eigenvectors
        u_ritz = np.dot(V, u_hess)

        # check orthogonalization of Ritz matrix
        self.assertAlmostEqual(np.linalg.norm(np.dot(u_ritz.conj().T, u_ritz) - np.identity(numiter)), 0., delta=1e-12,
                               msg='matrix of Ritz eigenvectors must be orthonormalized')

        # check U^H A U = diag(w)
        self.assertAlmostEqual(np.linalg.norm(np.dot(u_ritz.conj().T, np.dot(A, u_ritz)) - np.diag(w_hess)), 0., delta=1e-12,
                               msg='Ritz eigenvectors must diagonalize A within Krylov subspace')


if __name__ == '__main__':
    unittest.main()
