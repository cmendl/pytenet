import numpy as np
from .mps import MPS
from .mpo import MPO
from .qnumber import qnumber_flatten, is_qsparse

__all__ = ['vdot', 'norm', 'operator_average', 'operator_inner_product', 'operator_density_average',
           'apply_operator', 'compute_right_operator_blocks', 'apply_local_hamiltonian', 'apply_local_bond_contraction']


def vdot(chi: MPS, psi: MPS):
    """
    Compute the dot (scalar) product `<chi | psi>`, complex conjugating `chi`.

    Args:
        chi: wavefunction represented as MPS
        psi: wavefunction represented as MPS

    Returns:
        complex: `<chi | psi>`
    """
    assert psi.nsites == chi.nsites
    if psi.nsites == 0:
        return 0
    # initialize T by identity matrix
    T = np.identity(psi.A[-1].shape[2], dtype=psi.A[-1].dtype)
    for i in reversed(range(psi.nsites)):
        T = contraction_step_right(psi.A[i], chi.A[i], T)
    # T should now be a 1x1 tensor
    assert T.shape == (1, 1)
    return T[0, 0]


def norm(psi: MPS):
    """
    Compute the standard L2 norm of a matrix product state.
    """
    return np.sqrt(vdot(psi, psi).real)


def contraction_step_right(A: np.ndarray, B: np.ndarray, R: np.ndarray):
    r"""
    Contraction step from right to left, for example to compute the
    inner product of two matrix product states.

    To-be contracted tensor network::

          _____           ______
         /     \         /
      ---|1 B*2|---   ---|1
         \__0__/         |
            |            |
                         |    R
          __|__          |
         /  0  \         |
      ---|1 A 2|---   ---|0
         \_____/         \______
    """

    assert A.ndim == 3
    assert B.ndim == 3
    assert R.ndim == 2
    # multiply with A tensor
    T = np.tensordot(A, R, 1)
    # multiply with conjugated B tensor
    Rnext = np.tensordot(T, B.conj(), axes=((0, 2), (0, 2)))
    return Rnext


def contraction_step_left(A: np.ndarray, B: np.ndarray, L: np.ndarray):
    r"""
    Contraction step from left to right, for example to compute the
    inner product of two matrix product states.

    To-be contracted tensor network::

     ______           _____
           \         /     \
          1|---   ---|1 B*2|---
           |         \__0__/
           |            |
      L    |
           |          __|__
           |         /  0  \
          0|---   ---|1 A 2|---
     ______/         \_____/
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert L.ndim == 2
    # multiply with conjugated B tensor
    T = np.tensordot(L, B.conj(), axes=(1, 1))
    # multiply with A tensor
    Lnext = np.tensordot(A, T, axes=((0, 1), (1, 0)))
    return Lnext


def operator_average(psi: MPS, op: MPO):
    """
    Compute the expectation value `<psi | op | psi>`.

    Args:
        psi: wavefunction represented as MPS
        op:  operator represented as MPO

    Returns:
        complex: `<psi | op | psi>`
    """
    assert psi.nsites == op.nsites
    if psi.nsites == 0:
        return 0
    # initialize T by identity matrix
    T = np.identity(psi.A[-1].shape[2], dtype=psi.A[-1].dtype)
    T = T.reshape((psi.A[-1].shape[2], 1, psi.A[-1].shape[2]))
    for i in reversed(range(psi.nsites)):
        T = contraction_operator_step_right(psi.A[i], psi.A[i], op.A[i], T)
    # T should now be a 1x1x1 tensor
    assert T.shape == (1, 1, 1)
    return T[0, 0, 0]


def operator_inner_product(chi: MPS, op: MPO, psi: MPS):
    """
    Compute the inner product `<chi | op | psi>`.

    Args:
        chi: wavefunction represented as MPS
        op:  operator represented as MPO
        psi: wavefunction represented as MPS

    Returns:
        complex: `<chi | op | psi>`
    """
    assert chi.nsites == op.nsites
    assert psi.nsites == op.nsites
    if psi.nsites == 0:
        return 0
    # initialize T by identity matrix
    assert chi.A[-1].shape[2] == psi.A[-1].shape[2]
    T = np.identity(psi.A[-1].shape[2], dtype=psi.A[-1].dtype)
    T = T.reshape((psi.A[-1].shape[2], 1, psi.A[-1].shape[2]))
    for i in reversed(range(psi.nsites)):
        T = contraction_operator_step_right(psi.A[i], chi.A[i], op.A[i], T)
    # T should now be a 1x1x1 tensor
    assert T.shape == (1, 1, 1)
    return T[0, 0, 0]


def operator_density_average(rho: MPO, op: MPO):
    """
    Compute the expectation value `tr[op rho]`.

    Args:
        rho: density matrix represented as MPO
        op:  operator represented as MPO

    Returns:
        complex: `tr[op rho]`
    """
    assert rho.nsites == op.nsites
    if rho.nsites == 0:
        return 0
    # initialize T as 1x1 matrix
    T = np.identity(1, dtype=rho.A[-1].dtype)
    for i in reversed(range(rho.nsites)):
        T = contraction_operator_density_step_right(rho.A[i], op.A[i], T)
    # T should now be a 1x1 matrix
    assert T.shape == (1, 1)
    return T[0, 0]


def apply_operator(op: MPO, psi: MPS) -> MPS:
    """
    Apply an operator represented as MPO to a state in MPS form.
    """
    # quantum numbers on physical sites must match
    assert np.array_equal(psi.qd, op.qd)
    assert psi.nsites == op.nsites
    # bond quantum numbers
    qD = [qnumber_flatten((op.qD[i], psi.qD[i])) for i in range(psi.nsites + 1)]
    op_psi = MPS(psi.qd, qD, fill='postpone')
    for i in range(psi.nsites):
        A = np.tensordot(op.A[i], psi.A[i], axes=(1, 0))
        A = np.transpose(A, (0, 1, 3, 2, 4))
        # group virtual bonds
        s = A.shape
        A = np.reshape(A, (s[0], s[1]*s[2], s[3]*s[4]))
        op_psi.A[i] = A
        assert is_qsparse(op_psi.A[i], [op_psi.qd, op_psi.qD[i], -op_psi.qD[i+1]]), \
            'sparsity pattern of MPS tensor does not match quantum numbers'
    return op_psi


def contraction_operator_step_right(A: np.ndarray, B: np.ndarray, W: np.ndarray, R: np.ndarray):
    r"""
    Contraction step from right to left, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

          _____           ______
         /     \         /
      ---|1 B*2|---   ---|2
         \__0__/         |
            |            |
                         |
          __|__          |
         /  0  \         |
      ---|2 W 3|---   ---|1   R
         \__1__/         |
            |            |
                         |
          __|__          |
         /  0  \         |
      ---|1 A 2|---   ---|0
         \_____/         \______
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert W.ndim == 4
    assert R.ndim == 3
    # multiply with A tensor
    T = np.tensordot(A, R, 1)
    # multiply with W tensor
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))
    # interchange levels 0 <-> 2 in T
    T = T.transpose((2, 1, 0, 3))
    # multiply with conjugated B tensor
    Rnext = np.tensordot(T, B.conj(), axes=((2, 3), (0, 2)))
    return Rnext


def contraction_operator_step_left(A: np.ndarray, B: np.ndarray, W: np.ndarray, L: np.ndarray):
    r"""
    Contraction step from left to right, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

     ______           _____
           \         /     \
          2|---   ---|1 B*2|---
           |         \__0__/
           |            |
           |
           |          __|__
           |         /  0  \
      L   1|---   ---|2 W 3|---
           |         \__1__/
           |            |
           |
           |          __|__
           |         /  0  \
          0|---   ---|1 A 2|---
     ______/         \_____/
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert W.ndim == 4
    assert L.ndim == 3
    # multiply with conjugated B tensor
    T = np.tensordot(L, B.conj(), axes=(2, 1))
    # multiply with W tensor
    T = np.tensordot(W, T, axes=((0, 2), (2, 1)))
    # multiply with A tensor
    Lnext = np.tensordot(A, T, axes=((0, 1), (0, 2)))
    return Lnext


def contraction_operator_density_step_right(A: np.ndarray, W: np.ndarray, R: np.ndarray):
    r"""
    Contraction step between two matrix product operators
    (typically density matrix and Hamiltonian).

    To-be contracted tensor network (with = denoting a connected loop)::

            =
          __|__           ______
         /  0  \         /
      ---|2 W 3|---   ---|1
         \__1__/         |
            |            |
                         |    R
          __|__          |
         /  0  \         |
      ---|2 A 3|---   ---|0
         \__1__/         \______
            |
            =
    """
    assert A.ndim == 4
    assert W.ndim == 4
    assert R.ndim == 2
    # multiply with A tensor
    T = np.tensordot(A, R, 1)
    # multiply with W tensor
    T = np.tensordot(T, W, axes=((1, 0, 3), (0, 1, 3)))
    return T


def compute_right_operator_blocks(psi: MPS, op: MPO):
    """
    Compute all partial contractions from the right.
    """
    L = psi.nsites
    assert L == op.nsites
    BR = [None for _ in range(L)]
    # initialize rightmost dummy block
    BR[L-1] = np.array([[[1]]], dtype=complex)
    for i in reversed(range(L-1)):
        BR[i] = contraction_operator_step_right(psi.A[i+1], psi.A[i+1], op.A[i+1], BR[i+1])
    return BR


def apply_local_hamiltonian(L: np.ndarray, R: np.ndarray, W: np.ndarray, A: np.ndarray):
    r"""
    Apply a local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor)::

     ______                           ______
           \                         /
          2|---1                 2---|2
           |                         |
           |                         |
           |            0            |
           |          __|__          |
           |         /  0  \         |
      L   1|---   ---|2 W 3|---   ---|1   R
           |         \__1__/         |
           |            |            |
           |                         |
           |          __|__          |
           |         /  0  \         |
          0|---   ---|1 A 2|---   ---|0
     ______/         \_____/         \______
    """
    assert L.ndim == 3
    assert R.ndim == 3
    assert W.ndim == 4
    assert A.ndim == 3
    # multiply A with R tensor and store result in T
    T = np.tensordot(A, R, 1)
    # multiply T with W tensor
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))
    # multiply T with L tensor
    T = np.tensordot(T, L, axes=((2, 1), (0, 1)))
    # interchange levels 1 <-> 2 in T
    T = T.transpose((0, 2, 1))
    return T


def apply_local_bond_contraction(L, R, C):
    r"""
    Apply "zero-site" bond contraction.

    To-be contracted tensor network::

     ______                           ______
           \                         /
          2|---                   ---|2
           |                         |
           |                         |
           |                         |
           |                         |
           |                         |
      L   1|-----------   -----------|1   R
           |                         |
           |                         |
           |                         |
           |          _____          |
           |         /     \         |
          0|---   ---|0 C 1|---   ---|0
     ______/         \_____/         \______
    """
    assert L.ndim == 3
    assert R.ndim == 3
    assert C.ndim == 2
    # multiply C with R tensor and store result in T
    T = np.tensordot(C, R, 1)
    # multiply L with T tensor
    T = np.tensordot(L, T, axes=((0, 1), (0, 1)))
    return T
