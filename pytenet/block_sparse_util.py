"""
utility functions for handling block-sparse tensors with quantum number conservation.
"""

import numpy as np

__all__ = [
    "qnumber_outer_sum", "common_qnumbers", "qnumber_flatten", "is_qsparse",
    "enforce_qsparsity", "sort_by_qnumbers", "slice_with_qnumber",
    "block_sparse_qr", "block_sparse_eigh", "block_sparse_svd"]


def qnumber_outer_sum(qnums):
    """
    Compute the sum of all combinations of quantum numbers in `qnums`,
    and return the result as a tensor.

    Example:
        >>> qnumber_outer_sum([[0, 2, -1], [0, 1]])
        [[ 0,  1],
         [ 2,  3],
         [-1,  0]]
    """
    if len(qnums) == 0:
        return np.array(0)

    t = qnums[0]
    for i in range(1, len(qnums)):
        t = np.add.outer(t, qnums[i])
    return t


def common_qnumbers(qnums0, qnums1):
    """
    Find common quantum numbers between two lists of quantum numbers.
    """
    return np.intersect1d(qnums0, qnums1)


def qnumber_flatten(qnums):
    """
    Combine quantum numbers into a single vector.
    """
    return qnumber_outer_sum(qnums).reshape(-1)


def is_qsparse(a, qnums):
    """
    Test whether sparsity structure of `a` matches quantum numbers, i.e., if the
    quantum numbers corresponding to non-zero entries in `a` sum to zero.
    """
    mask = qnumber_outer_sum(qnums)
    return not np.any(np.where(mask == 0, 0, a))


def enforce_qsparsity(a, qnums):
    """
    Enforce sparsity pattern on `a` based on quantum numbers.
    """
    mask = qnumber_outer_sum(qnums)
    it = np.nditer(a, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        if mask[it.multi_index] != 0:
            x[...] = 0


def sort_by_qnumbers(a, q0, q1):
    """
    Sorts a matrix according to quantum numbers.

    Returns:
        a: sorted matrix
        (idx0, q0): indices that would sort the qnums, sorted quantum numbers for the rows
        (idx1, q1): indices that would sort the qnums, sorted quantum numbers for the columns
    """
    # require NumPy arrays for indexing
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)

    # sort quantum numbers and arrange entries in 'a' accordingly;
    # using mergesort to avoid permutations of identical quantum numbers
    idx0 = np.argsort(q0, kind="mergesort")
    idx1 = np.argsort(q1, kind="mergesort")
    if np.any(idx0 - np.arange(len(idx0))):
        # if not sorted yet...
        q0 = q0[idx0]
        a = a[idx0, :]
    if np.any(idx1 - np.arange(len(idx1))):
        # if not sorted yet...
        q1 = q1[idx1]
        a = a[:, idx1]
    return a, (idx0, q0), (idx1, q1)


def slice_with_qnumber(qn, qnums):
    """
    Assuming the quantum numbers are sorted, find the first and last indices
    at which a given quantum number appears.
    """
    iqn = np.where(qnums == qn)[0]
    start = iqn[0]
    end = iqn[-1] + 1
    return slice(start, end)


def block_sparse_qr(a, q0, q1):
    """
    Compute the block-wise QR decompositions of a matrix, taking block sparsity
    structure dictated by quantum numbers into account (that is, `a[i, j]` can
    only be non-zero if `q0[i] == q1[j]`).

    The resulting 'r' matrix is not necessarily upper triangular due to
    the reordering of entries.
    """
    a = np.asarray(a)
    assert a.ndim == 2
    assert len(q0) == a.shape[0]
    assert len(q1) == a.shape[1]
    assert is_qsparse(a, [q0, -q1])

    # find common quantum numbers
    qis = common_qnumbers(q0, q1)

    if len(qis) == 0:
        assert np.linalg.norm(a) == 0
        # special case: no common quantum numbers;
        # use dummy intermediate dimension 1 with all entries in 'r' set to zero
        q = np.zeros((a.shape[0], 1), dtype=a.dtype)
        r = np.zeros((1, a.shape[1]), dtype=a.dtype)
        # single column of 'q' should have norm 1
        q[0, 0] = 1
        # ensure non-zero entry in 'q' formally matches quantum numbers
        qinterm = q0[:1]
        return (q, r, qinterm)

    a, (idx0, q0), (idx1, q1) = sort_by_qnumbers(a, q0, q1)

    # maximum intermediate dimension
    max_interm_dim = min(a.shape)

    # keep track of intermediate dimension
    dim_bond = 0

    q = np.zeros((a.shape[0], max_interm_dim), dtype=a.dtype)
    r = np.zeros((max_interm_dim, a.shape[1]), dtype=a.dtype)

    # corresponding intermediate quantum numbers
    qinterm = np.zeros(max_interm_dim, dtype=q0.dtype)

    # for each shared quantum number...
    for qnum in qis:
        # indices of current quantum number
        row_slice = slice_with_qnumber(qnum, q0)
        col_slice = slice_with_qnumber(qnum, q1)

        # perform QR decomposition of current block
        qsub, rsub = np.linalg.qr(a[row_slice, col_slice], mode='reduced')

        # update intermediate dimension
        dim_prev = dim_bond
        dim_bond += qsub.shape[1]

        q[row_slice, dim_prev:dim_bond] = qsub
        r[dim_prev:dim_bond, col_slice] = rsub
        qinterm[dim_prev:dim_bond] = qnum

    assert dim_bond <= max_interm_dim

    # use actual intermediate dimensions
    q = q[:, :dim_bond]
    r = r[:dim_bond, :]
    qinterm = qinterm[:dim_bond]

    # undo sorting of quantum numbers
    if np.any(idx0 - np.arange(len(idx0))):
        q = q[np.argsort(idx0), :]
    if np.any(idx1 - np.arange(len(idx1))):
        r = r[:, np.argsort(idx1)]

    return (q, r, qinterm)


def block_sparse_eigh(a, q0):
    r"""
    Compute the block-wise diagonalization of a Hermitian matrix `a`,
    taking the block sparsity structure dictated by quantum numbers into account
    (that is, `a[i, j]` can only be non-zero if `q0[i] == q0[j]`).

    Finds `u` and eigvals such that
        $a = u @ diag(eigvals) @ u^\dagger$
    """
    assert a.ndim == 2
    assert a.shape[0] == a.shape[1]
    assert len(q0) == a.shape[0]
    assert is_qsparse(a, [q0, -q0])

    # find common quantum numbers
    qis = set(q0)

    a, (idx0, q0), (_, q1) = sort_by_qnumbers(a, q0, q0)

    # maximum intermediate dimension
    max_interm_dim = a.shape[0]

    # keep track of intermediate dimension
    dim_bond = 0

    # allocate memory for unitary `u` and diagonal eval matrices
    u = np.zeros((a.shape[0], max_interm_dim), dtype=a.dtype)
    evals = np.zeros(max_interm_dim)  # `evals` vector corresponds to the diagonal matrix
    q = np.zeros(max_interm_dim, dtype=q0.dtype)

    # for each shared quantum number...
    for qnum in qis:
        # indices of current quantum numbers
        row_slice = slice_with_qnumber(qnum, q0)
        col_slice = slice_with_qnumber(qnum, q1)

        # perform diagonalization of current block
        eval_sub, u_sub = np.linalg.eigh(a[row_slice, col_slice])

        # update intermediate dimension
        dim_prev = dim_bond
        dim_bond += len(eval_sub)

        u[row_slice, dim_prev:dim_bond] = u_sub
        evals[dim_prev:dim_bond] = eval_sub
        q[dim_prev:dim_bond] = qnum

    assert dim_bond <= max_interm_dim

    # use actual intermediate dimensions
    u = u[:, :dim_bond]
    evals = evals[:dim_bond]
    q = q[:dim_bond]

    # undo sorting of quantum numbers
    if np.any(idx0 - np.arange(len(idx0))):
        u = u[np.argsort(idx0), :]

    return (u, evals, q)


def block_sparse_svd(a, q0, q1):
    """
    Compute the block-wise SVD of a block-sparse matrix.
    """
    a = np.asarray(a)
    assert a.ndim == 2
    assert len(q0) == a.shape[0]
    assert len(q1) == a.shape[1]
    assert is_qsparse(a, [q0, -q1])

    # find common quantum numbers
    qis = common_qnumbers(q0, q1)

    if len(qis) == 0:
        assert np.linalg.norm(a) == 0
        # special case: no common quantum numbers;
        # use dummy intermediate dimension 1
        u = np.zeros((a.shape[0], 1), dtype=a.dtype)
        v = np.zeros((1, a.shape[1]), dtype=a.dtype)
        s = np.zeros(1)
        # single column of 'u' should have norm 1
        if a.shape[0] > 0:
            u[0, 0] = 1
        # ensure non-zero entry in 'u' formally matches quantum numbers
        q = q0[:1]
        # 'v' must remain zero matrix to satisfy quantum number constraints
        return (u, s, v, q)

    a, (idx0, q0), (idx1, q1) = sort_by_qnumbers(a, q0, q1)

    # maximum intermediate dimension
    max_interm_dim = min(a.shape)

    # keep track of intermediate dimension
    dim_bond = 0

    # allocate memory for U and V matrices, singular values and
    # corresponding intermediate quantum numbers
    u = np.zeros((a.shape[0], max_interm_dim), dtype=a.dtype)
    v = np.zeros((max_interm_dim, a.shape[1]), dtype=a.dtype)
    s = np.zeros(max_interm_dim)
    q = np.zeros(max_interm_dim, dtype=q0.dtype)

    # for each shared quantum number...
    for qnum in qis:
        # indices of current quantum number
        row_slice = slice_with_qnumber(qnum, q0)
        col_slice = slice_with_qnumber(qnum, q1)

        # perform SVD decomposition of current block
        usub, ssub, vsub = np.linalg.svd(a[row_slice, col_slice], full_matrices=False)

        # update intermediate dimension
        dim_prev = dim_bond
        dim_bond += len(ssub)

        u[row_slice, dim_prev:dim_bond] = usub
        v[dim_prev:dim_bond, col_slice] = vsub
        s[dim_prev:dim_bond] = ssub
        q[dim_prev:dim_bond] = qnum

    assert dim_bond <= max_interm_dim

    # use actual intermediate dimensions
    u = u[:, :dim_bond]
    v = v[:dim_bond, :]
    s = s[:dim_bond]
    q = q[:dim_bond]

    # undo sorting of quantum numbers
    if np.any(idx0 - np.arange(len(idx0))):
        u = u[np.argsort(idx0), :]
    if np.any(idx1 - np.arange(len(idx1))):
        v = v[:, np.argsort(idx1)]

    return (u, s, v, q)
