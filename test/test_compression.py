import unittest

from numpy import allclose
from numpy.random import default_rng

from pytenet.compression import rounding
from pytenet.mps import MPS

class TestRoundingCompression(unittest.TestCase):

    def test_no_truncation(self):
        # Create a random MPS
        rng = default_rng(seed=123446234512)

        phys_dim = 3
        virt_dims = [1, 3, 5, 7, 3, 1]
        qd = rng.integers(-2, 3, size=phys_dim)
        qD = [rng.integers(-2, 3, size=Di) for Di in virt_dims]
        mps = MPS(qd, qD, fill='random', rng=rng)

        compr_mps = rounding(mps, tol=-1)

        self.assertTrue(allclose(mps.as_vector(), compr_mps.as_vector()))

    def test_truncation(self):
        # Create a random MPS
        rng = default_rng(seed=123446234512)

        phys_dim = 3
        virt_dims = [1, 3, 5, 7, 3, 1]
        qd = rng.integers(-2, 3, size=phys_dim)
        qD = [rng.integers(-2, 3, size=Di) for Di in virt_dims]
        mps = MPS(qd, qD, fill='random', rng=rng)

        compr_mps = rounding(mps, tol=0.15)

        self.assertTrue(allclose(mps.as_vector(), compr_mps.as_vector(), rtol=0.15))

if __name__ == '__main__':
    unittest.main()
