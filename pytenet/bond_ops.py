import numpy as np


def retained_bond_indices(s, tol):
    """
    Indices of retained singular values based on given tolerance.
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


def split_matrix_svd(A, tol):
    """
    Split a matrix by singular value decomposition,
    and truncate small singular values based on tolerance.
    """
    u, s, v = np.linalg.svd(A, full_matrices=False)
    idx = retained_bond_indices(s, tol)
    return (u[:, idx], s[idx], v[idx, :])
