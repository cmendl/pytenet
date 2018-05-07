import numpy as np


def operator_average(psi, op):
    """Compute the expectation value <psi | op | psi>.

    Args:
        psi:    wavefunction represented as MPS
        op:     operator represented as MPO

    Returns:    <psi | op | psi> (complex number)
    """

    assert psi.nsites == op.nsites

    if psi.nsites == 0:
        return 0

    # initialize T by identity matrix
    T = np.identity(psi.A[-1].shape[2], dtype=type(psi.A[-1]))
    T = np.reshape(T, (psi.A[-1].shape[2], 1, psi.A[-1].shape[2]))

    for i in reversed(range(psi.nsites)):
        T = _contraction_operator_step_right(psi.A[i], op.A[i], T)

    # T should now be a 1x1x1 tensor
    assert T.shape == (1, 1, 1)

    return T[0, 0, 0]


def _contraction_operator_step_right(A, W, R):
    """Contraction step from right to left, with a matrix product operator sandwiched in between."""

    assert len(A.shape) == 3
    assert len(W.shape) == 4
    assert len(R.shape) == 3

    # multiply with A tensor
    T = np.tensordot(A, R, 1)

    # multiply with W tensor
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))

    # interchange levels 0 <-> 2 in T
    T = np.transpose(T, (2, 1, 0, 3))

    # multiply with conjugated A tensor
    Rnext = np.tensordot(T, A.conj(), axes=((2, 3), (0, 2)))

    return Rnext
