"""
utility functions for handling block-sparse tensors with quantum number conservation.
"""

from autoray import numpy as np
from .util import argsort

__all__ = [
    "qnumber_outer_sum", "common_qnumbers", "qnumber_flatten", "is_qsparse",
    "enforce_qsparsity", "neg_qnumbers", "random_qnumbers", "sort_by_qnumbers",
    "slice_with_qnumber", "block_sparse_qr", "block_sparse_eigh", "block_sparse_svd"]


def qnumber_outer_sum(qnums, shift: int = 0):
    """
    Compute the sum of all combinations of quantum numbers in `qnums`,
    and return the result as nested lists.

    Example:
        >>> qnumber_outer_sum([[0, 2, -1], [0, 1]])
        [[ 0,  1],
         [ 2,  3],
         [-1,  0]]
    """
    if len(qnums) == 0:
        return shift
    if len(qnums) == 1:
        return [qn + shift for qn in qnums[0]]
    return [qnumber_outer_sum(qnums[1:], qn + shift) for qn in qnums[0]]


def _unique(items) -> list:
    """
    Extract the unique entries from a sequence of items.
    The output list is sorted.
    """
    output = []
    for x in items:
        if x not in output:
            output.append(x)
    return sorted(output)


def common_qnumbers(qnums0, qnums1) -> list[int]:
    """
    Find common quantum numbers between two lists of quantum numbers.
    """
    return [qn for qn in _unique(qnums0) if qn in qnums1]


def _flatten_nested_list(nested_list):
    """
    Flatten recursively nested lists.
    """
    for item in nested_list:
        if isinstance(item, list):
            yield from _flatten_nested_list(item)
        else:
            yield item


def qnumber_flatten(qnums) -> list[int]:
    """
    Combine quantum numbers into a single vector.
    """
    return list(_flatten_nested_list(qnumber_outer_sum(qnums)))


def is_qsparse(a, qnums) -> bool:
    """
    Test whether sparsity structure of `a` matches quantum numbers, i.e., if the
    quantum numbers corresponding to non-zero entries in `a` sum to zero.
    """
    mask = np.array(qnumber_outer_sum(qnums), dtype=int, like=a)
    return not np.any(np.where(mask == 0, 0, a))


def enforce_qsparsity(a, qnums):
    """
    Enforce a sparsity pattern on `a` based on quantum numbers.
    """
    mask = np.array(qnumber_outer_sum(qnums), dtype=int)
    a[np.where(mask != 0)] = 0


def neg_qnumbers(qnums) -> list[int]:
    """
    Negate a list of quantum numbers.
    """
    return [-qn for qn in qnums]


def random_qnumbers(low: int, high: int, size: int, rng) -> list[int]:
    """
    Generate a list of random quantum numbers from `low` (inclusive) to `high` (exclusive).
    """
    qnums = rng.integers(low, high, size)
    # convert to native integers
    return [int(qn) for qn in qnums]


def sort_by_qnumbers(a, q0, q1):
    """
    Sort a matrix according to quantum numbers.

    Returns:
        a: sorted matrix
        (idx0, q0): indices that would sort the qnums, sorted quantum numbers for the rows
        (idx1, q1): indices that would sort the qnums, sorted quantum numbers for the columns
    """
    # sort quantum numbers and arrange entries in `a` accordingly;
    # sorting has to be stable to avoid permutations of identical quantum numbers
    idx0 = argsort(q0)
    idx1 = argsort(q1)
    if any(i - idx for i, idx in enumerate(idx0)):
        # if not sorted yet...
        q0 = [q0[idx] for idx in idx0]
        a = a[idx0, :]
    if any(i - idx for i, idx in enumerate(idx1)):
        # if not sorted yet...
        q1 = [q1[idx] for idx in idx1]
        a = a[:, idx1]
    return a, (idx0, q0), (idx1, q1)


def slice_with_qnumber(qn, qnums):
    """
    Assuming the quantum numbers are sorted, find the first and last indices
    at which a given quantum number appears.
    """
    return slice(qnums.index(qn), len(qnums) - qnums[::-1].index(qn))


def block_sparse_qr(a, q0, q1):
    """
    Compute the block-wise QR decompositions of a matrix, taking block sparsity
    structure dictated by quantum numbers into account (that is, `a[i, j]` can
    only be non-zero if `q0[i] == q1[j]`).

    The resulting `r` matrix is not necessarily upper triangular due to
    the reordering of entries.
    """
    a = np.asarray(a)
    assert a.ndim == 2
    assert len(q0) == a.shape[0]
    assert len(q1) == a.shape[1]
    assert is_qsparse(a, [q0, neg_qnumbers(q1)])

    # find common quantum numbers
    qis = common_qnumbers(q0, q1)

    if len(qis) == 0:
        assert np.linalg.norm(a) == 0
        # special case: no common quantum numbers;
        # use dummy intermediate dimension 1 with all entries in `r` set to zero
        q = np.zeros((a.shape[0], 1), like=a)
        r = np.zeros((1, a.shape[1]), like=a)
        # single column of `q` should have norm 1
        q[0, 0] = 1
        # ensure non-zero entry in `q` formally matches quantum numbers
        qinterm = q0[:1]
        return q, r, qinterm

    a, (idx0, q0), (idx1, q1) = sort_by_qnumbers(a, q0, q1)

    # maximum intermediate dimension
    max_interm_dim = min(a.shape)

    # keep track of intermediate dimension
    dim_bond = 0

    q = np.zeros((a.shape[0], max_interm_dim), like=a)
    r = np.zeros((max_interm_dim, a.shape[1]), like=a)

    # corresponding intermediate quantum numbers
    qinterm = [0 for _ in range(max_interm_dim)]

    # for each shared quantum number...
    for qnum in qis:
        # indices of current quantum number
        row_slice = slice_with_qnumber(qnum, q0)
        col_slice = slice_with_qnumber(qnum, q1)

        # perform QR decomposition of current block
        qsub, rsub = np.linalg.qr(a[row_slice, col_slice], mode="reduced")

        # update intermediate dimension
        dim_prev = dim_bond
        dim_bond += qsub.shape[1]

        q[row_slice, dim_prev:dim_bond] = qsub
        r[dim_prev:dim_bond, col_slice] = rsub
        qinterm[dim_prev:dim_bond] = (dim_bond - dim_prev) * [qnum]

    assert dim_bond <= max_interm_dim

    # use actual intermediate dimensions
    q = q[:, :dim_bond]
    r = r[:dim_bond, :]
    qinterm = qinterm[:dim_bond]

    # undo sorting of quantum numbers
    if any(i - idx for i, idx in enumerate(idx0)):
        q = q[argsort(idx0), :]
    if any(i - idx for i, idx in enumerate(idx1)):
        r = r[:, argsort(idx1)]

    return q, r, qinterm


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
    assert is_qsparse(a, [q0, neg_qnumbers(q0)])

    # find common quantum numbers
    qis = _unique(q0)

    a, (idx0, q0), (_, q1) = sort_by_qnumbers(a, q0, q0)

    # maximum intermediate dimension
    max_interm_dim = a.shape[0]

    # keep track of intermediate dimension
    dim_bond = 0

    # allocate memory for unitary `u` and diagonal eval matrices
    u = np.zeros((a.shape[0], max_interm_dim), like=a)
    # `eigvals` vector corresponds to the diagonal matrix
    eigvals = np.zeros(max_interm_dim, dtype=float, like=a)
    q = max_interm_dim * [0]

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
        eigvals[dim_prev:dim_bond] = eval_sub
        q[dim_prev:dim_bond] = (dim_bond - dim_prev) * [qnum]

    assert dim_bond <= max_interm_dim

    # use actual intermediate dimensions
    u = u[:, :dim_bond]
    eigvals = eigvals[:dim_bond]
    q = q[:dim_bond]

    # undo sorting of quantum numbers
    if any(i - idx for i, idx in enumerate(idx0)):
        u = u[argsort(idx0), :]

    return u, eigvals, q


def block_sparse_svd(a, q0, q1):
    """
    Compute the block-wise SVD of a block-sparse matrix.
    """
    a = np.asarray(a)
    assert a.ndim == 2
    assert len(q0) == a.shape[0]
    assert len(q1) == a.shape[1]
    assert is_qsparse(a, [q0, neg_qnumbers(q1)])

    # find common quantum numbers
    qis = common_qnumbers(q0, q1)

    if len(qis) == 0:
        assert np.linalg.norm(a) == 0
        # special case: no common quantum numbers;
        # use dummy intermediate dimension 1
        u = np.zeros((a.shape[0], 1), like=a)
        v = np.zeros((1, a.shape[1]), like=a)
        s = np.zeros(1, dtype=float, like=a)
        # single column of `u` should have norm 1
        if a.shape[0] > 0:
            u[0, 0] = 1
        # ensure non-zero entry in `u` formally matches quantum numbers
        q = q0[:1]
        # `v` must remain zero matrix to satisfy quantum number constraints
        return u, s, v, q

    a, (idx0, q0), (idx1, q1) = sort_by_qnumbers(a, q0, q1)

    # maximum intermediate dimension
    max_interm_dim = min(a.shape)

    # keep track of intermediate dimension
    dim_bond = 0

    # allocate memory for U and V matrices, singular values and
    # corresponding intermediate quantum numbers
    u = np.zeros((a.shape[0], max_interm_dim), like=a)
    v = np.zeros((max_interm_dim, a.shape[1]), like=a)
    s = np.zeros(max_interm_dim, dtype=float, like=a)
    q = max_interm_dim * [0]

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
        q[dim_prev:dim_bond] = (dim_bond - dim_prev) * [qnum]

    assert dim_bond <= max_interm_dim

    # use actual intermediate dimensions
    u = u[:, :dim_bond]
    v = v[:dim_bond, :]
    s = s[:dim_bond]
    q = q[:dim_bond]

    # undo sorting of quantum numbers
    if any(i - idx for i, idx in enumerate(idx0)):
        u = u[argsort(idx0), :]
    if any(i - idx for i, idx in enumerate(idx1)):
        v = v[:, argsort(idx1)]

    return u, s, v, q
