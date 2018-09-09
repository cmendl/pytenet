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


def split_matrix_svd(A, q0, q1, tol):
    """
    Split a matrix by singular value decomposition,
    taking block sparsity structure dictated by quantum numbers into account,
    and truncate small singular values based on tolerance.
    """
    assert A.ndim == 2
    assert len(q0) == A.shape[0]
    assert len(q1) == A.shape[1]

    # find common quantum numbers
    qis = np.intersect1d(q0, q1)

    if len(qis) == 0:
        assert np.linalg.norm(A) == 0
        # special case: no common quantum numbers;
        # use dummy intermediate dimension 1
        u = np.zeros((A.shape[0], 1), dtype=A.dtype)
        v = np.zeros((1, A.shape[1]), dtype=A.dtype)
        s = np.zeros(1)
        # single column of 'u' should have norm 1
        u[0, 0] = 1
        # ensure non-zero entry in 'u' formally matches quantum numbers
        q = q0[:1]
        # 'v' must remain zero matrix to satisfy quantum number constraints
        return (u, s, v, q)

    # require NumPy arrays for indexing
    q0 = np.array(q0)
    q1 = np.array(q1)

    # sort quantum numbers and arrange entries in A accordingly;
    # using mergesort to avoid permutations of identical quantum numbers
    idx0 = np.argsort(q0, kind='mergesort')
    idx1 = np.argsort(q1, kind='mergesort')
    if np.any(idx0 - np.arange(len(idx0))):
        # if not sorted yet...
        q0 = q0[idx0]
        A = A[idx0, :]
    if np.any(idx1 - np.arange(len(idx1))):
        # if not sorted yet...
        q1 = q1[idx1]
        A = A[:, idx1]

    # maximum intermediate dimension
    max_interm_dim = min(A.shape)

    # keep track of intermediate dimension
    D = 0

    u = np.zeros((A.shape[0], max_interm_dim), dtype=A.dtype)
    v = np.zeros((max_interm_dim, A.shape[1]), dtype=A.dtype)
    s = np.zeros(max_interm_dim)

    # corresponding intermediate quantum numbers
    q = np.zeros(max_interm_dim, dtype=q0.dtype)

    # for each shared quantum number...
    for qn in qis:
        # indices of current quantum number
        iqn = np.where(q0 == qn)[0]; i0 = iqn[0]; i1 = iqn[-1] + 1
        iqn = np.where(q1 == qn)[0]; j0 = iqn[0]; j1 = iqn[-1] + 1

        # perform SVD decomposition of current block
        usub, ssub, vsub = np.linalg.svd(A[i0:i1, j0:j1], full_matrices=False)

        # update intermediate dimension
        Dprev = D
        D += len(ssub)

        u[i0:i1, Dprev:D] = usub
        v[Dprev:D, j0:j1] = vsub
        s[Dprev:D] = ssub
        q[Dprev:D] = qn

    assert D <= max_interm_dim

    # use actual intermediate dimensions
    u = u[:, :D]
    v = v[:D, :]
    s = s[:D]
    q = q[:D]

    # truncate small singular values
    idx = retained_bond_indices(s, tol)
    u = u[:, idx]
    v = v[idx, :]
    s = s[idx]
    q = q[idx]

    # undo sorting of quantum numbers
    if np.any(idx0 - np.arange(len(idx0))):
        u = u[np.argsort(idx0), :]
    if np.any(idx1 - np.arange(len(idx1))):
        v = v[:, np.argsort(idx1)]

    return (u, s, v, q)


def qr(A, q0, q1):
    """
    Compute the block-wise QR decompositions of a matrix, taking block sparsity
    structure dictated by quantum numbers into account (that is, A[i, j] can
    only be non-zero if q0[i] == q1[j]).

    The resulting R matrix is not necessarily upper triangular due to
    reordering of entries.
    """
    assert A.ndim == 2
    assert len(q0) == A.shape[0]
    assert len(q1) == A.shape[1]

    # find common quantum numbers
    qis = np.intersect1d(q0, q1)

    if len(qis) == 0:
        assert np.linalg.norm(A) == 0
        # special case: no common quantum numbers;
        # use dummy intermediate dimension 1 with all entries in 'R' set to zero
        Q = np.zeros((A.shape[0], 1), dtype=A.dtype)
        R = np.zeros((1, A.shape[1]), dtype=A.dtype)
        # single column of 'Q' should have norm 1
        Q[0, 0] = 1
        # ensure non-zero entry in 'Q' formally matches quantum numbers
        qinterm = q0[:1]
        return (Q, R, qinterm)

    # require NumPy arrays for indexing
    q0 = np.array(q0)
    q1 = np.array(q1)

    # sort quantum numbers and arrange entries in A accordingly;
    # using mergesort to avoid permutations of identical quantum numbers
    idx0 = np.argsort(q0, kind='mergesort')
    idx1 = np.argsort(q1, kind='mergesort')
    if np.any(idx0 - np.arange(len(idx0))):
        # if not sorted yet...
        q0 = q0[idx0]
        A = A[idx0, :]
    if np.any(idx1 - np.arange(len(idx1))):
        # if not sorted yet...
        q1 = q1[idx1]
        A = A[:, idx1]

    # maximum intermediate dimension
    max_interm_dim = min(A.shape)

    # keep track of intermediate dimension
    D = 0

    Q = np.zeros((A.shape[0], max_interm_dim), dtype=A.dtype)
    R = np.zeros((max_interm_dim, A.shape[1]), dtype=A.dtype)

    # corresponding intermediate quantum numbers
    qinterm = np.zeros(max_interm_dim, dtype=q0.dtype)

    # for each shared quantum number...
    for qn in qis:
        # indices of current quantum number
        iqn = np.where(q0 == qn)[0]; i0 = iqn[0]; i1 = iqn[-1] + 1
        iqn = np.where(q1 == qn)[0]; j0 = iqn[0]; j1 = iqn[-1] + 1

        # perform QR decomposition of current block
        Qsub, Rsub = np.linalg.qr(A[i0:i1, j0:j1], mode='reduced')

        # update intermediate dimension
        Dprev = D
        D += Qsub.shape[1]

        Q[i0:i1, Dprev:D] = Qsub
        R[Dprev:D, j0:j1] = Rsub
        qinterm[Dprev:D] = qn

    assert D <= max_interm_dim

    # use actual intermediate dimensions
    Q = Q[:, :D]
    R = R[:D, :]
    qinterm = qinterm[:D]

    # undo sorting of quantum numbers
    if np.any(idx0 - np.arange(len(idx0))):
        Q = Q[np.argsort(idx0), :]
    if np.any(idx1 - np.arange(len(idx1))):
        R = R[:, np.argsort(idx1)]

    return (Q, R, qinterm)
