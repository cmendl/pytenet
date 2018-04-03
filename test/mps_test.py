import unittest
import numpy as np
from mps import MPS


class TestMPS(unittest.TestCase):

    def test_orthonormalization(self):

        # create random matrix product state
        d = 4
        D = [1, 5, 3, 7, 1]
        mps0 = MPS(d, D, fill='random')

        self.assertEqual(mps0.bond_dims, D)

        # wavefunction on full Hilbert space
        psi = mps0.merge_full()

        print('Performing left-orthonormalization...')
        cL = mps0.orthonormalize(mode='left')
        print('Normalization factor:', cL)
        print('New virtual bond dimensions:', mps0.bond_dims)

        psiL = mps0.merge_full()
        # wavefunction should now be normalized
        self.assertAlmostEqual(np.linalg.norm(psiL), 1., delta=1e-12)

        # wavefunctions before and after normalization must match
        # (up to normalization factor)
        self.assertAlmostEqual(np.linalg.norm(cL*psiL - psi), 0., delta=1e-10)

        # check left-orthonormalization
        for i in range(mps0.nsites):
            s = mps0.A[i].shape
            assert s[0] == d
            Q = mps0.A[i].reshape((s[0]*s[1], s[2]))
            QH_Q = np.dot(Q.conj().T, Q)
            self.assertAlmostEqual(np.linalg.norm(QH_Q - np.identity(s[2])), 0., delta=1e-12)

        print('')

        print('Performing right-orthonormalization...')
        cR = mps0.orthonormalize(mode='right')
        print('Normalization factor:', cR)
        print('New virtual bond dimensions:', mps0.bond_dims)
        # factor must have magnitude 1 due to previous left-orthonormalization
        self.assertAlmostEqual(abs(cR), 1., delta=1e-12)

        psiR = mps0.merge_full()
        # wavefunctions must match
        self.assertAlmostEqual(np.linalg.norm(psiL - cR*psiR), 0., delta=1e-10)

        # check right-orthonormalization
        for i in range(mps0.nsites):
            s = mps0.A[i].shape
            assert s[0] == d
            Q = mps0.A[i].transpose((0, 2, 1)).reshape((s[0]*s[2], s[1]))
            QH_Q = np.dot(Q.conj().T, Q)
            self.assertAlmostEqual(np.linalg.norm(QH_Q - np.identity(s[1])), 0., delta=1e-12)


if __name__ == '__main__':
    unittest.main()
