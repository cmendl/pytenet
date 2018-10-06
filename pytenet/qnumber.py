import numpy as np

__all__ = ['qnumber_outer_sum', 'qnumber_flatten', 'is_qsparse']


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

    T = qnums[0]
    for i in range(1, len(qnums)):
        T = np.add.outer(T, qnums[i])
    return T


def qnumber_flatten(qnums):
    """
    Combine quantum numbers into a single vector.
    """
    return qnumber_outer_sum(qnums).reshape(-1)


def is_qsparse(A, qnums):
    """
    Test whether sparsity structure of `A` matches quantum numbers, i.e., if the
    quantum numbers corresponding to non-zero entries in `A` sum to zero.
    """
    mask = qnumber_outer_sum(qnums)
    return not np.any(np.where(mask == 0, 0, A))
