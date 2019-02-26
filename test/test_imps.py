import unittest
import numpy as np
import pytenet as ptn


class TestiMPS(unittest.TestCase):

    def test_left_orthonormalize_uniform(self):

        d = 3
        D = 7

        A = randn_complex((d, D, D))

        # perform left-orthonormalization
        Aleft, L, nrm = ptn.left_orthonormalize_uniform(A)
        A /= nrm

        err_repr = np.linalg.norm(np.tensordot(L, A, axes=(1, 1)).transpose((1, 0, 2)) - np.tensordot(Aleft, L, axes=1))
        self.assertAlmostEqual(err_repr, 0., delta=1e-12, msg='Aleft must be equal to L A L^{-1}')

        # check left-orthonormalization
        AA = np.einsum(Aleft, (0, 1, 2), Aleft.conj(), (0, 1, 3), (2, 3))
        self.assertAlmostEqual(np.linalg.norm(AA - np.identity(len(AA))), 0., delta=1e-12, msg='left-orthonormalization')


    def test_right_orthonormalize_uniform(self):

        d = 4
        D = 10

        A = randn_complex((d, D, D))

        # perform right-orthonormalization
        Aright, R, nrm = ptn.right_orthonormalize_uniform(A)
        A /= nrm

        err_repr = np.linalg.norm(np.tensordot(R, Aright, axes=(1, 1)).transpose((1, 0, 2)) - np.tensordot(A, R, axes=1))
        self.assertAlmostEqual(err_repr, 0., delta=1e-12, msg='Aright must be equal to R^{-1} A R')

        # check right-orthonormalization
        AA = np.einsum(Aright, (0, 1, 2), Aright.conj(), (0, 3, 2), (1, 3))
        self.assertAlmostEqual(np.linalg.norm(AA - np.identity(len(AA))), 0., delta=1e-12, msg='right-orthonormalization')


def randn_complex(size):
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)


if __name__ == '__main__':
    unittest.main()
