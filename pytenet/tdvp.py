"""
TDVP time integration algorithms for MPS, based on
a Lanczos iteration for the local time evolution steps.

Reference:
  - J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete\n
    Unifying time evolution and optimization with matrix product states\n
    Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
"""

import numpy as np
from .mps import MPS, mps_merge_tensor_pair, mps_split_tensor_svd
from .mpo import MPO, mpo_merge_tensor_pair
from .operation import (
        contraction_operator_step_right,
        contraction_operator_step_left,
        compute_right_operator_blocks,
        apply_local_hamiltonian,
        apply_local_bond_contraction)
from .krylov import expm_krylov
from .block_sparse_util import qnumber_flatten, is_qsparse, block_sparse_qr

__all__ = ["tdvp_singlesite", "tdvp_twosite"]


def tdvp_singlesite(hamiltonian: MPO, psi: MPS, dt, numsteps: int, numiter_lanczos: int = 25):
    """
    Symmetric single-site TDVP integration.
    `psi` is overwritten in-place with the time-evolved state.

    Args:
        hamiltonian: Hamiltonian as MPO
        psi: initial state as MPS
        dt: time step; for real-time evolution, use purely imaginary dt
        numsteps: number of time steps
        numiter_lanczos: number of Lanczos iterations for each site-local step

    Returns:
        float: norm of initial psi

    Reference:
        J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
        Unifying time evolution and optimization with matrix product states
        Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
    """

    # number of lattice sites
    nsites = hamiltonian.nsites
    assert nsites == psi.nsites

    # right-normalize input matrix product state
    nrm = psi.orthonormalize(mode="right")

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    rblocks = compute_right_operator_blocks(psi, hamiltonian)
    lblocks = [None for _ in range(nsites)]
    lblocks[0] = np.array([[[1]]], dtype=rblocks[0].dtype)

    # consistency check
    for i, rb in enumerate(rblocks):
        assert is_qsparse(rb, [psi.qbonds[i+1], hamiltonian.qbonds[i+1], -psi.qbonds[i+1]]), \
            "sparsity pattern of operator blocks must match quantum numbers"

    for _ in range(numsteps):

        # sweep from left to right
        for i in range(nsites - 1):
            # evolve psi.a[i] forward in time by half a time step
            psi.a[i] = _local_hamiltonian_step(
                lblocks[i], rblocks[i], hamiltonian.a[i], psi.a[i], 0.5*dt, numiter_lanczos)
            # left-orthonormalize current psi.a[i]
            s = psi.a[i].shape
            (q, c, psi.qbonds[i+1]) = block_sparse_qr(
                psi.a[i].reshape((s[0]*s[1], s[2])),
                qnumber_flatten([psi.qsite, psi.qbonds[i]]), psi.qbonds[i+1])
            psi.a[i] = q.reshape((s[0], s[1], q.shape[1]))
            # update the left blocks
            lblocks[i+1] = contraction_operator_step_left(
                psi.a[i], psi.a[i], hamiltonian.a[i], lblocks[i])
            # evolve `c` backward in time by half a time step
            c = _local_bond_step(lblocks[i+1], rblocks[i], c, -0.5*dt, numiter_lanczos)
            # update psi.a[i+1] tensor: multiply with c from left
            psi.a[i+1] = np.einsum(psi.a[i+1], (0, 3, 2), c, (1, 3), (0, 1, 2), optimize=True)

        # evolve psi.a[nsites-1] forward in time by a full time step
        i = nsites - 1
        psi.a[i] = _local_hamiltonian_step(
            lblocks[i], rblocks[i], hamiltonian.a[i], psi.a[i], dt, numiter_lanczos)

        # sweep from right to left
        for i in reversed(range(1, nsites)):
            # right-orthonormalize current psi.a[i]
            # flip left and right virtual bond dimensions
            psi.a[i] = psi.a[i].transpose((0, 2, 1))
            # perform QR decomposition
            s = psi.a[i].shape
            (q, c, qbond) = block_sparse_qr(
                psi.a[i].reshape((s[0]*s[1], s[2])),
                qnumber_flatten([psi.qsite, -psi.qbonds[i+1]]), -psi.qbonds[i])
            psi.qbonds[i] = -qbond
            # replace psi.a[i] by reshaped `q` matrix and
            # undo flip of left and right virtual bond dimensions
            psi.a[i] = q.reshape((s[0], s[1], q.shape[1])).transpose((0, 2, 1))
            # update the right blocks
            rblocks[i-1] = contraction_operator_step_right(
                psi.a[i], psi.a[i], hamiltonian.a[i], rblocks[i])
            # evolve `c` backward in time by half a time step
            c = np.transpose(c)
            c = _local_bond_step(lblocks[i], rblocks[i-1], c, -0.5*dt, numiter_lanczos)
            # update psi.a[i-1] tensor: multiply with c from right
            psi.a[i-1] = np.einsum(psi.a[i-1], (0, 1, 3), c, (3, 2), (0, 1, 2), optimize=True)
            # evolve psi.a[i-1] forward in time by half a time step
            psi.a[i-1] = _local_hamiltonian_step(
                lblocks[i-1], rblocks[i-1], hamiltonian.a[i-1], psi.a[i-1], 0.5*dt, numiter_lanczos)

    # return norm of initial psi
    return nrm


def tdvp_twosite(hamiltonian: MPO, psi: MPS, dt, numsteps: int,
                 numiter_lanczos: int = 25, tol_split = 0):
    """
    Symmetric two-site TDVP integration.
    `psi` is overwritten in-place with the time-evolved state.

    Args:
        hamiltonian: Hamiltonian as MPO
        psi: initial state as MPS
        dt: time step; for real-time evolution, use purely imaginary dt
        numsteps: number of time steps
        numiter_lanczos: number of Lanczos iterations for each site-local step
        tol_split: tolerance for SVD-splitting of neighboring MPS tensors

    Returns:
        float: norm of initial psi

    Reference:
        J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
        Unifying time evolution and optimization with matrix product states
        Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
    """

    # number of lattice sites
    nsites = hamiltonian.nsites
    assert nsites == psi.nsites
    assert nsites >= 2

    # right-normalize input matrix product state
    nrm = psi.orthonormalize(mode="right")

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    rblocks = compute_right_operator_blocks(psi, hamiltonian)
    lblocks = [None for _ in range(nsites)]
    lblocks[0] = np.array([[[1]]], dtype=rblocks[0].dtype)

    # consistency check
    for i, rb in enumerate(rblocks):
        assert is_qsparse(rb, [psi.qbonds[i+1], hamiltonian.qbonds[i+1], -psi.qbonds[i+1]]), \
            "sparsity pattern of operator blocks must match quantum numbers"

    h2 = [mpo_merge_tensor_pair(hamiltonian.a[i], hamiltonian.a[i+1]) for i in range(nsites - 1)]

    for _ in range(numsteps):

        # sweep from left to right
        for i in range(nsites - 2):
            # merge neighboring tensors
            a_cur = mps_merge_tensor_pair(psi.a[i], psi.a[i+1])
            # evolve a_cur forward in time by half a time step
            a_cur = _local_hamiltonian_step(
                lblocks[i], rblocks[i+1], h2[i], a_cur, 0.5*dt, numiter_lanczos)
            # split `a_cur`
            psi.a[i], psi.a[i+1], psi.qbonds[i+1] = mps_split_tensor_svd(
                a_cur, psi.qsite, psi.qsite,
                [psi.qbonds[i], psi.qbonds[i+2]], "right", tol=tol_split)
            # update the left blocks
            lblocks[i+1] = contraction_operator_step_left(
                psi.a[i], psi.a[i], hamiltonian.a[i], lblocks[i])
            # evolve psi.a[i+1] backward in time by half a time step
            psi.a[i+1] = _local_hamiltonian_step(
                lblocks[i+1], rblocks[i+1], hamiltonian.a[i+1],
                psi.a[i+1], -0.5*dt, numiter_lanczos)

        # rightmost tensor pair
        i = nsites - 2
        # merge neighboring tensors
        a_cur = mps_merge_tensor_pair(psi.a[i], psi.a[i+1])
        # evolve a_cur forward in time by a full time step
        a_cur = _local_hamiltonian_step(
            lblocks[i], rblocks[i+1], h2[i], a_cur, dt, numiter_lanczos)
        # split `a_cur`
        psi.a[i], psi.a[i+1], psi.qbonds[i+1] = mps_split_tensor_svd(
            a_cur, psi.qsite, psi.qsite, [psi.qbonds[i], psi.qbonds[i+2]], "left", tol=tol_split)
        # update the right blocks
        rblocks[i] = contraction_operator_step_right(
            psi.a[i+1], psi.a[i+1], hamiltonian.a[i+1], rblocks[i+1])

        # sweep from right to left
        for i in reversed(range(nsites - 2)):
            # evolve psi.a[i+1] backward in time by half a time step
            psi.a[i+1] = _local_hamiltonian_step(
                lblocks[i+1], rblocks[i+1], hamiltonian.a[i+1],
                psi.a[i+1], -0.5*dt, numiter_lanczos)
            # merge neighboring tensors
            a_cur = mps_merge_tensor_pair(psi.a[i], psi.a[i+1])
            # evolve a_cur forward in time by half a time step
            a_cur = _local_hamiltonian_step(
                lblocks[i], rblocks[i+1], h2[i], a_cur, 0.5*dt, numiter_lanczos)
            # split `a_cur`
            psi.a[i], psi.a[i+1], psi.qbonds[i+1] = mps_split_tensor_svd(
                a_cur, psi.qsite, psi.qsite,
                [psi.qbonds[i], psi.qbonds[i+2]], "left", tol=tol_split)
            # update the right blocks
            rblocks[i] = contraction_operator_step_right(
                psi.a[i+1], psi.a[i+1], hamiltonian.a[i+1], rblocks[i+1])

    # return norm of initial psi
    return nrm


def _local_hamiltonian_step(nsites, r, w, a, dt, numiter: int):
    """
    Local time step effected by Hamiltonian, based on a Lanczos iteration.
    """
    return expm_krylov(
        lambda x: apply_local_hamiltonian(nsites, r, w, x.reshape(a.shape)).reshape(-1),
            a.reshape(-1), -dt, numiter, hermitian=True).reshape(a.shape)


def _local_bond_step(nsites, r, c, dt, numiter: int):
    """
    Local "zero-site" bond step, based on a Lanczos iteration.
    """
    return expm_krylov(
        lambda x: apply_local_bond_contraction(nsites, r, x.reshape(c.shape)).reshape(-1),
            c.reshape(-1), -dt, numiter, hermitian=True).reshape(c.shape)
