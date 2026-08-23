"""
Generic utility functions.
"""

import time
import autoray as ar

__all__ = ["argsort", "crandn"]


def argsort(items):
    """
    Compute the indices that sort `items`.
    The sorting is stable.
    """
    return sorted(range(len(items)), key=items.__getitem__)


def crandn(size = None, rng = None, scale: float | complex = 1.0):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if size is None:
        size = 1
    if rng is None:
        rng = ar.do("random.default_rng", int(time.time()))
    return ar.do("random.array", shape=size, dist="normal", dtype=complex, scale=scale, rng=rng)
