"""
Higher-level tensor network operations on a chain topology.
"""

from autoray import numpy as np
from .mps import MPS
from .mpo import MPO
from .block_sparse_util import qnumber_flatten, is_qsparse, neg_qnumbers, block_sparse_qr
from .bond_ops import split_block_sparse_matrix_svd

__all__ = ["contraction_operator_step_right", "contraction_operator_step_left",
           "compute_right_operator_blocks",
           "mpo_average", "mpo_inner_product", "mpo_density_average",
           "apply_mpo", "apply_local_hamiltonian", "apply_local_bond_contraction"]


def contraction_operator_step_right(a: np.ndarray, b: np.ndarray, w: np.ndarray, r: np.ndarray):
    r"""
    Contraction step from right to left, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

       ╭───────╮       ╭─────────╮
       │       │       │         │
     ──0   b*  2──   ──2         │
       │       │       │         │
       ╰───1───╯       │         │
           │           │         │
                       │         │
           │           │         │
       ╭───1───╮       │         │
       │       │       │         │
     ──0   w   3──   ──1    r    │
       │       │       │         │
       ╰───2───╯       │         │
           │           │         │
                       │         │
           │           │         │
       ╭───1───╮       │         │
       │       │       │         │
     ──0   a   2──   ──0         │
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
    t = np.tensordot(w, t, axes=([2, 3], [1, 2]))
    # make original left virtual bond of `a` the leading dimension
    t = np.transpose(t, (2, 0, 1, 3))
    # multiply with conjugated `b` tensor
    r_next = np.tensordot(t, b.conj(), axes=([2, 3], [1, 2]))
    return r_next


def contraction_operator_step_left(a: np.ndarray, b: np.ndarray, w: np.ndarray, l: np.ndarray):
    r"""
    Contraction step from left to right, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

     ╭─────────╮       ╭───────╮
     │         │       │       │
     │         2──   ──0   b*  2──
     │         │       │       │
     │         │       ╰───1───╯
     │         │           │
     │         │
     │         │           │
     │         │       ╭───1───╮
     │         │       │       │
     │    l    1──   ──0   w   3──
     │         │       │       │
     │         │       ╰───2───╯
     │         │           │
     │         │
     │         │           │
     │         │       ╭───1───╮
     │         │       │       │
     │         0──   ──0   a   2──
     │         │       │       │
     ╰─────────╯       ╰───────╯
    """
    assert a.ndim == 3
    assert b.ndim == 3
    assert w.ndim == 4
    assert l.ndim == 3
    # multiply with conjugated `b` tensor
    t = np.tensordot(l, b.conj(), axes=([2], [0]))
    # multiply with `w` tensor
    t = np.tensordot(w, t, axes=([0, 1], [1, 2]))
    # multiply with `a` tensor
    l_next = np.tensordot(a, t, axes=([0, 1], [2, 0]))
    return l_next


def compute_right_operator_blocks(psi: MPS, op: MPO):
    """
    Compute all partial contractions from the right.
    """
    nsites = psi.nsites
    assert nsites == op.nsites
    blocks = [None for _ in range(nsites)]
    # initialize rightmost dummy block
    blocks[nsites-1] = np.array([[[1]]], dtype=op.a[0].dtype, like=op.a[0])
    for i in reversed(range(nsites - 1)):
        blocks[i] = contraction_operator_step_right(psi.a[i+1], psi.a[i+1], op.a[i+1], blocks[i+1])
    return blocks


def contraction_operator_density_step_right(a: np.ndarray, w: np.ndarray, r: np.ndarray):
    r"""
    Contraction step between two matrix product operators
    (typically a density matrix and a Hamiltonian).

    To-be contracted tensor network (with a connected loop over the physical axes)::

          ╭╮
           │
       ╭───1───╮       ╭─────────╮
       │       │       │         │
     ──0   w   3──   ──1         │
       │       │       │         │
       ╰───2───╯       │         │
           │           │         │
                       │    r    │
           │           │         │
       ╭───1───╮       │         │
       │       │       │         │
     ──0   a   3──   ──0         │
       │       │       │         │
       ╰───2───╯       ╰─────────╯
           │
          ╰╯
    """
    assert a.ndim == 4
    assert w.ndim == 4
    assert r.ndim == 2
    # multiply with `a` tensor
    t = np.tensordot(a, r, 1)
    # multiply with `w` tensor
    t = np.tensordot(t, w, axes=([2, 1, 3], [1, 2, 3]))
    return t


def mpo_average(psi: MPS, op: MPO):
    """
    Compute the expectation value `<psi | op | psi>`.

    Args:
        psi: wavefunction represented as MPS
        op:  operator represented as MPO

    Returns:
        `<psi | op | psi>`
    """
    return mpo_inner_product(psi, op, psi)


def mpo_inner_product(chi: MPS, op: MPO, psi: MPS):
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
    t = np.identity(psi.a[-1].shape[2], like=psi.a[-1]).reshape(
            (psi.a[-1].shape[2], 1, psi.a[-1].shape[2]))
    for i in reversed(range(psi.nsites)):
        t = contraction_operator_step_right(psi.a[i], chi.a[i], op.a[i], t)
    # `t` should now be a 1 x 1 x 1 tensor
    assert t.shape == (1, 1, 1)
    return t[0, 0, 0]


def mpo_density_average(rho: MPO, op: MPO):
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
    t = np.identity(1, like=rho.a[-1])
    for i in reversed(range(rho.nsites)):
        t = contraction_operator_density_step_right(rho.a[i], op.a[i], t)
    # `t` should now be a 1 x 1 matrix
    assert t.shape == (1, 1)
    return t[0, 0]


def apply_mpo(op: MPO, psi: MPS, mode="direct", tol=0) -> MPS:
    """
    Apply an operator represented as MPO to a state in MPS form.

    The Cholesky-based compression (CBC) method is described in
    "Efficient application of tensor network operators to tensor network states" (see below).

    Reference:
        Richard M. Milbradt, Shuo Sun, Christian B. Mendl, Johnnie Gray, Garnet K.-L. Chan
        Efficient application of tensor network operators to tensor network states
        arXiv:2601.19650
    """
    # quantum numbers on physical sites must match
    assert psi.qsite == op.qsite
    assert psi.nsites == op.nsites
    assert psi.nsites > 0
    # (initial) bond quantum numbers, will be updated for CBC mode
    qbonds_init = [qnumber_flatten((op.qbonds[i], psi.qbonds[i])) for i in range(psi.nsites + 1)]
    op_psi = MPS(op.qsite, qbonds_init, fill="postpone")
    if mode == "direct":
        for i in range(psi.nsites):
            a = np.einsum(op.a[i], (0, 2, 5, 3), psi.a[i], (1, 5, 4), (0, 1, 2, 3, 4))
            # group virtual bonds
            s = a.shape
            a = a.reshape((s[0]*s[1], s[2], s[3]*s[4]))
            op_psi.a[i] = a
            assert is_qsparse(op_psi.a[i], [op_psi.qbonds[i], op_psi.qsite,
                                            neg_qnumbers(op_psi.qbonds[i+1])]), \
                "sparsity pattern of MPS tensor does not match quantum numbers"
        return op_psi
    elif mode in ("cbc", "CBC"):
        if psi.nsites == 1:
            # special case
            a = np.einsum(op.a[0], (0, 2, 5, 3), psi.a[0], (1, 5, 4), (0, 1, 2, 3, 4))
            # group virtual bonds
            s = a.shape
            a = a.reshape((s[0]*s[1], s[2], s[3]*s[4]))
            op_psi.a[0] = a
            return op_psi
        assert psi.a[0].shape[0] == 1 and op.a[0].shape[0] == 1
        # dummy left block
        lblock = np.array([[[1]]], dtype=psi.a[0].dtype, like=psi.a[0])
        lblocks = [lblock]
        lqbonds = [qnumber_flatten((op.qbonds[0], psi.qbonds[0]))]
        for i in range(psi.nsites):
            # contract left block with MPS and MPO tensors
            t = np.tensordot(lblock, psi.a[i], axes=([2], [0]))
            t = np.einsum(t, (0, 4, 5, 3), op.a[i], (4, 1, 5, 2), (0, 1, 2, 3))
            s = t.shape
            if i < psi.nsites - 1:
                q0 = qnumber_flatten((lqbonds[i], op.qsite))
                q1 = qnumber_flatten((op.qbonds[i + 1], psi.qbonds[i + 1]))
                _, sigma, v, qbond = split_block_sparse_matrix_svd(
                    t.reshape((s[0]*s[1], s[2]*s[3])), q0, q1, tol)
                lblock = np.reshape(sigma[:, None] * v, (v.shape[0], s[2], s[3]))
                lblocks.append(lblock)
                lqbonds.append(qbond)
            else:
                # rightmost site
                assert t.shape[2] == 1 and t.shape[3] == 1
                q0 = qnumber_flatten((op.qsite, neg_qnumbers(op_psi.qbonds[i + 1])))
                q1 = neg_qnumbers(lqbonds[i])
                q, _, qinterm = block_sparse_qr(t.reshape((s[0], s[1])).T, q0, q1)
                op_psi.a[i] = np.reshape(q.T, (q.shape[1], q.shape[0], 1))
                op_psi.qbonds[i] = neg_qnumbers(qinterm)
        # dummy right block
        rblock = np.array([[[1]]], dtype=psi.a[-1].dtype, like=psi.a[-1])
        for i in reversed(range(psi.nsites - 1)):
            rblock = contraction_operator_step_right(
                    psi.a[i + 1], op_psi.a[i + 1], op.a[i + 1], rblock)
            b = apply_local_hamiltonian(
                    psi.a[i], op.a[i], np.transpose(lblocks[i], (2, 1, 0)), rblock)
            if i > 0:
                s = b.shape
                q0 = qnumber_flatten((op.qsite, neg_qnumbers(op_psi.qbonds[i + 1])))
                q, _, qinterm = block_sparse_qr(b.reshape((s[0], s[1]*s[2])).T,
                                                q0, neg_qnumbers(lqbonds[i]))
                op_psi.a[i] = np.reshape(q.T, (q.shape[1], b.shape[1], b.shape[2]))
                op_psi.qbonds[i] = neg_qnumbers(qinterm)
            else:
                op_psi.a[i] = b
        for i in range(op_psi.nsites):
            assert is_qsparse(op_psi.a[i], [op_psi.qbonds[i], op_psi.qsite,
                              neg_qnumbers(op_psi.qbonds[i+1])]), \
                "sparsity pattern of MPS tensor does not match quantum numbers"
        return op_psi
    else:
        raise ValueError(f'`mode` = {mode} invalid; must be "direct" or "CBC".')


def apply_local_hamiltonian(a: np.ndarray, w: np.ndarray, l: np.ndarray, r: np.ndarray):
    r"""
    Apply a local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor)::

           .................................
          '                                 '
     ╭────:────╮                       ╭────:────╮
     │    :    │                       │    :    │
     │    :    2──   0           2   ──2    :    │
     │    :    │                       │    :    │
     │    :    │                       │    :    │
     │    :    │           1           │    :    │
     │    :    │                       │    :    │
     │    :    │           │           │    :    │
     │    '....│.......╭───1───╮.......│....'    │
     │         │       │       │       │         │
     │    l    1──   ──0   w   3──   ──1    r    │
     │         │       │       │       │         │
     │         │       ╰───2───╯       │         │
     │         │           │           │         │
     │         │                       │         │
     │         │           │           │         │
     │         │       ╭───1───╮       │         │
     │         │       │       │       │         │
     │         0──   ──0   a   2──   ──0         │
     │         │       │       │       │         │
     ╰─────────╯       ╰───────╯       ╰─────────╯
    """
    assert a.ndim == 3
    assert w.ndim == 4
    assert l.ndim == 3
    assert r.ndim == 3
    # multiply `a` with `r` tensor and store result in `t`
    t = np.tensordot(a, r, 1)
    # multiply `t` with `w` tensor
    # multiply with `w` tensor
    t = np.tensordot(w, t, axes=([2, 3], [1, 2]))
    # multiply `t` with `l` tensor
    t = np.tensordot(l, t, axes=([0, 1], [2, 0]))
    return t


def apply_local_bond_contraction(c: np.ndarray, l: np.ndarray, r: np.ndarray):
    r"""
    Apply a "zero-site" bond contraction.

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
    assert c.ndim == 2
    assert l.ndim == 3
    assert r.ndim == 3
    # multiply `c` with `r` tensor and store result in `t`
    t = np.tensordot(c, r, 1)
    # multiply `l` with `t` tensor
    t = np.tensordot(l, t, axes=([0, 1], [0, 1]))
    return t
