"""
Higher-level tensor network operations on a chain topology.
"""

import numpy as np
from .mps import MPS
from .mpo import MPO
from .block_sparse_util import qnumber_flatten, is_qsparse

__all__ = ["mps_vdot", "mps_norm", "compute_left_state_blocks", "operator_average",
           "operator_inner_product", "operator_density_average",
           "apply_mpo", "compute_right_operator_blocks",
           "apply_local_hamiltonian", "apply_local_bond_contraction"]


def mps_vdot(chi: MPS, psi: MPS):
    """
    Compute the dot (scalar) product `<chi | psi>`, complex conjugating `chi`.

    Args:
        chi: wavefunction represented as MPS
        psi: wavefunction represented as MPS

    Returns:
        `<chi | psi>`
    """
    assert psi.nsites == chi.nsites
    if psi.nsites == 0:
        return 0
    # initialize `t` by the identity matrix
    t = np.identity(psi.a[-1].shape[2], dtype=psi.a[-1].dtype)
    for i in reversed(range(psi.nsites)):
        t = mps_contraction_step_right(psi.a[i], chi.a[i], t)
    # t should now be a 1 x 1 tensor
    assert t.shape == (1, 1)
    return t[0, 0]


def mps_norm(psi: MPS):
    """
    Compute the standard L2 norm of a matrix product state.
    """
    return np.sqrt(mps_vdot(psi, psi).real)


def mps_contraction_step_right(a: np.ndarray, b: np.ndarray, r: np.ndarray):
    r"""
    Contraction step from right to left, for example to compute the
    inner product of two matrix product states.

    To-be contracted tensor network::

       ╭───────╮       ╭─────────╮
       │       │       │         │
     ──1   b*  2──   ──1         │
       │       │       │         │
       ╰───0───╯       │         │
           │           │         │
                       │    r    │
           │           │         │
       ╭───0───╮       │         │
       │       │       │         │
     ──1   a   2──   ──0         │
       │       │       │         │
       ╰───────╯       ╰─────────╯
    """

    assert a.ndim == 3
    assert b.ndim == 3
    assert r.ndim == 2
    # multiply with `a` tensor
    t = np.tensordot(a, r, 1)
    # multiply with conjugated b tensor
    r_next = np.tensordot(t, b.conj(), axes=((0, 2), (0, 2)))
    return r_next


def mps_contraction_step_left(a: np.ndarray, b: np.ndarray, l: np.ndarray):
    r"""
    Contraction step from left to right, for example to compute the
    inner product of two matrix product states.

    To-be contracted tensor network::

     ╭─────────╮       ╭───────╮
     │         │       │       │
     │         1──   ──1   b*  2──
     │         │       │       │
     │         │       ╰───0───╯
     │         │           │
     │    l    │
     │         │           │
     │         │       ╭───0───╮
     │         │       │       │
     │         0──   ──1   a   2──
     │         │       │       │
     ╰─────────╯       ╰───────╯
    """
    assert a.ndim == 3
    assert b.ndim == 3
    assert l.ndim == 2
    # multiply with conjugated `b` tensor
    t = np.tensordot(l, b.conj(), axes=(1, 1))
    # multiply with `a` tensor
    l_next = np.tensordot(a, t, axes=((0, 1), (1, 0)))
    return l_next


def compute_left_state_blocks(chi: MPS, psi: MPS):
    """
    Compute all partial contractions from the left of the inner product `<chi | psi>`.
    """
    nsites = chi.nsites
    assert nsites == psi.nsites
    blocks = [None for _ in range(nsites + 1)]
    # initialize leftmost dummy block
    blocks[0] = np.identity(1, dtype=psi.a[0].dtype)
    # compute left environment blocks
    for i in range(1, nsites + 1):
        blocks[i] = mps_contraction_step_left(psi.a[i-1], chi.a[i-1], blocks[i-1])
    return blocks


def operator_average(psi: MPS, op: MPO):
    """
    Compute the expectation value `<psi | op | psi>`.

    Args:
        psi: wavefunction represented as MPS
        op:  operator represented as MPO

    Returns:
        `<psi | op | psi>`
    """
    assert psi.nsites == op.nsites
    if psi.nsites == 0:
        return 0
    # initialize `t` by identity matrix
    t = np.identity(psi.a[-1].shape[2], dtype=psi.a[-1].dtype)
    t = t.reshape((psi.a[-1].shape[2], 1, psi.a[-1].shape[2]))
    for i in reversed(range(psi.nsites)):
        t = contraction_operator_step_right(psi.a[i], psi.a[i], op.a[i], t)
    # t should now be a 1 x 1 x 1 tensor
    assert t.shape == (1, 1, 1)
    return t[0, 0, 0]


def operator_inner_product(chi: MPS, op: MPO, psi: MPS):
    """
    Compute the inner product `<chi | op | psi>`.

    Args:
        chi: wavefunction represented as MPS
        op:  operator represented as MPO
        psi: wavefunction represented as MPS

    Returns:
        `<chi | op | psi>`
    """
    assert chi.nsites == op.nsites
    assert psi.nsites == op.nsites
    if psi.nsites == 0:
        return 0
    # initialize `t` by identity matrix
    assert chi.a[-1].shape[2] == psi.a[-1].shape[2]
    t = np.identity(psi.a[-1].shape[2], dtype=psi.a[-1].dtype)
    t = t.reshape((psi.a[-1].shape[2], 1, psi.a[-1].shape[2]))
    for i in reversed(range(psi.nsites)):
        t = contraction_operator_step_right(psi.a[i], chi.a[i], op.a[i], t)
    # t should now be a 1 x 1 x 1 tensor
    assert t.shape == (1, 1, 1)
    return t[0, 0, 0]


def operator_density_average(rho: MPO, op: MPO):
    """
    Compute the expectation value `tr[op rho]`.

    Args:
        rho: density matrix represented as MPO
        op:  operator represented as MPO

    Returns:
        `tr[op rho]`
    """
    assert rho.nsites == op.nsites
    if rho.nsites == 0:
        return 0
    # initialize `t` as 1 x 1 matrix
    t = np.identity(1, dtype=rho.a[-1].dtype)
    for i in reversed(range(rho.nsites)):
        t = contraction_operator_density_step_right(rho.a[i], op.a[i], t)
    # t should now be a 1 x 1 matrix
    assert t.shape == (1, 1)
    return t[0, 0]


def apply_mpo(op: MPO, psi: MPS) -> MPS:
    """
    Apply an operator represented as MPO to a state in MPS form.
    """
    # quantum numbers on physical sites must match
    assert np.array_equal(psi.qsite, op.qsite)
    assert psi.nsites == op.nsites
    # bond quantum numbers
    qbonds = [qnumber_flatten((op.qbonds[i], psi.qbonds[i])) for i in range(psi.nsites + 1)]
    op_psi = MPS(psi.qsite, qbonds, fill="postpone")
    for i in range(psi.nsites):
        a = np.tensordot(op.a[i], psi.a[i], axes=(1, 0))
        a = a.transpose((0, 1, 3, 2, 4))
        # group virtual bonds
        s = a.shape
        a = a.reshape((s[0], s[1]*s[2], s[3]*s[4]))
        op_psi.a[i] = a
        assert is_qsparse(op_psi.a[i], [op_psi.qsite, op_psi.qbonds[i], -op_psi.qbonds[i+1]]), \
            "sparsity pattern of MPS tensor does not match quantum numbers"
    return op_psi


def contraction_operator_step_right(a: np.ndarray, b: np.ndarray, w: np.ndarray, r: np.ndarray):
    r"""
    Contraction step from right to left, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

       ╭───────╮       ╭─────────╮
       │       │       │         │
     ──1   b*  2──   ──2         │
       │       │       │         │
       ╰───0───╯       │         │
           │           │         │
                       │         │
           │           │         │
       ╭───0───╮       │         │
       │       │       │         │
     ──2   w   3──   ──1    r    │
       │       │       │         │
       ╰───1───╯       │         │
           │           │         │
                       │         │
           │           │         │
       ╭───0───╮       │         │
       │       │       │         │
     ──1   a   2──   ──0         │
       │       │       │         │
       ╰───────╯       ╰─────────╯
    """
    assert a.ndim == 3
    assert b.ndim == 3
    assert w.ndim == 4
    assert r.ndim == 3
    # multiply with `a` tensor
    t = np.tensordot(a, r, 1)
    # multiply with `w` tensor
    t = np.tensordot(w, t, axes=((1, 3), (0, 2)))
    # interchange levels 0 <-> 2 in t
    t = t.transpose((2, 1, 0, 3))
    # multiply with conjugated b tensor
    r_next = np.tensordot(t, b.conj(), axes=((2, 3), (0, 2)))
    return r_next


def contraction_operator_step_left(a: np.ndarray, b: np.ndarray, w: np.ndarray, l: np.ndarray):
    r"""
    Contraction step from left to right, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

     ╭─────────╮       ╭───────╮
     │         │       │       │
     │         2──   ──1   b*  2──
     │         │       │       │
     │         │       ╰───0───╯
     │         │           │
     │         │
     │         │           │
     │         │       ╭───0───╮
     │         │       │       │
     │    l    1──   ──2   w   3──
     │         │       │       │
     │         │       ╰───1───╯
     │         │           │
     │         │
     │         │           │
     │         │       ╭───0───╮
     │         │       │       │
     │         0──   ──1   a   2──
     │         │       │       │
     ╰─────────╯       ╰───────╯
    """
    assert a.ndim == 3
    assert b.ndim == 3
    assert w.ndim == 4
    assert l.ndim == 3
    # multiply with conjugated `b` tensor
    t = np.tensordot(l, b.conj(), axes=(2, 1))
    # multiply with `w` tensor
    t = np.tensordot(w, t, axes=((0, 2), (2, 1)))
    # multiply with `a` tensor
    l_next = np.tensordot(a, t, axes=((0, 1), (0, 2)))
    return l_next


def contraction_operator_density_step_right(a: np.ndarray, w: np.ndarray, r: np.ndarray):
    r"""
    Contraction step between two matrix product operators
    (typically density matrix and Hamiltonian).

    To-be contracted tensor network (with a connected loop over the physical axes)::

          ╭╮
           │
       ╭───0───╮       ╭─────────╮
       │       │       │         │
     ──2   w   3──   ──1         │
       │       │       │         │
       ╰───1───╯       │         │
           │           │         │
                       │    r    │
           │           │         │
       ╭───0───╮       │         │
       │       │       │         │
     ──2   a   3──   ──0         │
       │       │       │         │
       ╰───1───╯       ╰─────────╯
           │
          ╰╯
    """
    assert a.ndim == 4
    assert w.ndim == 4
    assert r.ndim == 2
    # multiply with `a` tensor
    t = np.tensordot(a, r, 1)
    # multiply with `w` tensor
    t = np.tensordot(t, w, axes=((1, 0, 3), (0, 1, 3)))
    return t


def compute_right_operator_blocks(psi: MPS, op: MPO):
    """
    Compute all partial contractions from the right.
    """
    nsites = psi.nsites
    assert nsites == op.nsites
    blocks = [None for _ in range(nsites)]
    # initialize rightmost dummy block
    blocks[nsites-1] = np.array([[[1]]])
    for i in reversed(range(nsites - 1)):
        blocks[i] = contraction_operator_step_right(psi.a[i+1], psi.a[i+1], op.a[i+1], blocks[i+1])
    return blocks


def apply_local_hamiltonian(l: np.ndarray, r: np.ndarray, w: np.ndarray, a: np.ndarray):
    r"""
    Apply a local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor)::

     ╭─────────╮                       ╭─────────╮
     │         │                       │         │
     │         2──   1           2   ──2         │
     │         │                       │         │
     │         │                       │         │
     │         │           0           │         │
     │         │                       │         │
     │         │           │           │         │
     │         │       ╭───0───╮       │         │
     │         │       │       │       │         │
     │    l    1──   ──2   w   3──   ──1    r    │
     │         │       │       │       │         │
     │         │       ╰───1───╯       │         │
     │         │           │           │         │
     │         │                       │         │
     │         │           │           │         │
     │         │       ╭───0───╮       │         │
     │         │       │       │       │         │
     │         0──   ──1   a   2──   ──0         │
     │         │       │       │       │         │
     ╰─────────╯       ╰───────╯       ╰─────────╯
    """
    assert l.ndim == 3
    assert r.ndim == 3
    assert w.ndim == 4
    assert a.ndim == 3
    # multiply `a` with `r` tensor and store result in `t`
    t = np.tensordot(a, r, 1)
    # multiply `t` with `w` tensor
    t = np.tensordot(w, t, axes=((1, 3), (0, 2)))
    # multiply `t` with `l` tensor
    t = np.tensordot(t, l, axes=((2, 1), (0, 1)))
    # interchange levels 1 <-> 2 in `t`
    t = t.transpose((0, 2, 1))
    return t


def apply_local_bond_contraction(l, r, c):
    r"""
    Apply "zero-site" bond contraction.

    To-be contracted tensor network::

     ╭─────────╮                       ╭─────────╮
     │         │                       │         │
     │         2──                   ──2         │
     │         │                       │         │
     │         │                       │         │
     │         │                       │         │
     │         │                       │         │
     │         │                       │         │
     │         │                       │         │
     │         │                       │         │
     │    l    1──────────   ──────────1    r    │
     │         │                       │         │
     │         │                       │         │
     │         │                       │         │
     │         │                       │         │
     │         │                       │         │
     │         │       ╭───────╮       │         │
     │         │       │       │       │         │
     │         0──   ──0   c   1──   ──0         │
     │         │       │       │       │         │
     ╰─────────╯       ╰───────╯       ╰─────────╯
    """
    assert l.ndim == 3
    assert r.ndim == 3
    assert c.ndim == 2
    # multiply `c` with `r` tensor and store result in `t`
    t = np.tensordot(c, r, 1)
    # multiply `l` with `t` tensor
    t = np.tensordot(l, t, axes=((0, 1), (0, 1)))
    return t
