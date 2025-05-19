import numpy as np

__all__ = ['qnumber_outer_sum', 'qnumber_flatten', 'is_qsparse', 'enforce_qsparsity']


def qnumber_outer_sum(qnums):
    """
    Compute the sum of all combinations of quantum numbers in `qnums`,
    and return result as tensor.

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
