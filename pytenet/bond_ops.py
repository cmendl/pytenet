"""
Functions concerning virtual bonds.
"""

import numpy as np
from .block_sparse_util import block_sparse_svd

__all__ = ["retained_bond_indices", "split_block_sparse_matrix_svd"]


def von_neumann_entropy(sigma):
    """
    Compute the von Neumann entropy of the singular values `sigma`.
    """
    nrm = np.linalg.norm(sigma)
    if nrm == 0:
        return 0
    sq = (sigma / nrm)**2
    sq = sq[sq > 0]
    return sum(-sq * np.log(sq))


def retained_bond_indices(s, tol):
    """
    Indices of retained singular values based on the specified tolerance.
    """
    w = np.linalg.norm(s)
    if w == 0:
        return np.array([], dtype=int)

    # normalized squares
    s = (s / w)**2

    # accumulate values from smallest to largest
    sort_idx = np.argsort(s)
    s[sort_idx] = np.cumsum(s[sort_idx])

    return np.where(s > tol)[0]


def split_block_sparse_matrix_svd(a, q0, q1, tol):
    """
    Split a matrix by singular value decomposition,
    taking block sparsity structure dictated by quantum numbers into account,
    and truncate small singular values based on the specified tolerance.
    """
    u, s, v, q = block_sparse_svd(a, q0, q1)
    # truncate small singular values
    idx = retained_bond_indices(s, tol)
    u = u[:, idx]
    v = v[idx, :]
    s = s[idx]
    q = q[idx]
    return (u, s, v, q)
