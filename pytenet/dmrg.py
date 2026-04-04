"""
DMRG algorithm.

Reference:
    Ulrich Schollwöck
    The density-matrix renormalization group in the age of matrix product states
    Ann. Phys. (N. Y.) 326, 96-192 (2011)
"""

import numpy as np
from .mps import (MPS, mps_local_orthonormalize_left_qr, mps_local_orthonormalize_right_qr,
                  mps_merge_tensor_pair, mps_split_tensor_svd)
from .mpo import MPO, mpo_merge_tensor_pair
from .chain_ops import (contraction_operator_step_right, contraction_operator_step_left,
                        compute_right_operator_blocks, apply_local_hamiltonian)
from .krylov import eigh_krylov
from .block_sparse_util import is_qsparse

__all__ = ["dmrg_singlesite", "dmrg_twosite"]


def dmrg_singlesite(hamiltonian: MPO, psi: MPS, numsweeps: int, numiter_lanczos: int = 25):
    """
    Run the single-site DMRG algorithm: Approximate the ground state as MPS
    via left- and right-sweeping and local single-site optimizations.
    The input `psi` is used as the starting state
    and is updated in-place during the optimization.
    Its virtual bond dimensions cannot increase.

    Args:
        hamiltonian: Hamiltonian as MPO
        psi: initial MPS used for optimization; will be overwritten
        numsweeps: maximum number of left and right sweeps
        numiter_lanczos: number of local Lanczos iterations

    Returns:
        numpy.ndarray: array of approximate ground state energies after each iteration
    """

    # number of lattice sites
    nsites = hamiltonian.nsites
    assert nsites == psi.nsites

    # right-normalize input matrix product state
    psi.orthonormalize(mode="right")

    # left and right operator blocks
    # initialize leftmost block by 1 x 1 x 1 identity
    rblocks = compute_right_operator_blocks(psi, hamiltonian)
    lblocks = [None for _ in range(nsites)]
    lblocks[0] = np.array([[[1]]], dtype=rblocks[0].dtype)

    # consistency check
    for i, rb in enumerate(rblocks):
        assert is_qsparse(rb, (psi.qbonds[i+1], hamiltonian.qbonds[i+1], -psi.qbonds[i+1])), \
            "sparsity pattern of operator blocks must match quantum numbers"

    en_min = np.zeros(numsweeps)

    # TODO: number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0

        # sweep from left to right
        for i in range(nsites - 1):
            en, psi.a[i] = _minimize_local_energy(
                hamiltonian.a[i], lblocks[i], rblocks[i], psi.a[i], numiter_lanczos)
            # left-orthonormalize current psi.a[i]
            psi.a[i], psi.a[i+1], psi.qbonds[i+1] = mps_local_orthonormalize_left_qr(
                psi.a[i], psi.a[i+1], psi.qsite, psi.qbonds[i:i+2])
            # update the left blocks
            lblocks[i+1] = contraction_operator_step_left(
                psi.a[i], psi.a[i], hamiltonian.a[i], lblocks[i])

        # sweep from right to left
        for i in reversed(range(1, nsites)):
            en, psi.a[i] = _minimize_local_energy(
                hamiltonian.a[i], lblocks[i], rblocks[i], psi.a[i], numiter_lanczos)
            # right-orthonormalize current psi.a[i]
            psi.a[i], psi.a[i-1], psi.qbonds[i] = mps_local_orthonormalize_right_qr(
                psi.a[i], psi.a[i-1], psi.qsite, psi.qbonds[i:i+2])
            # update the right blocks
            rblocks[i-1] = contraction_operator_step_right(
                psi.a[i], psi.a[i], hamiltonian.a[i], rblocks[i])

        # right-normalize leftmost tensor to ensure that `psi` is normalized
        psi.a[0], _, psi.qbonds[0] = mps_local_orthonormalize_right_qr(
            psi.a[0], np.array([[[1]]]), psi.qsite, psi.qbonds[:2])

        # record energy after each sweep
        en_min[n] = en

    return en_min


def dmrg_twosite(hamiltonian: MPO, psi: MPS, numsweeps: int,
                 numiter_lanczos: int = 25, tol_split: float = 0):
    """
    Run the two-site DMRG algorithm: Approximate the ground state as MPS
    via left- and right-sweeping and local two-site optimizations.
    The input `psi` is used as the starting state
    and is updated in-place during the optimization.

    Args:
        hamiltonian: Hamiltonian as MPO
        psi: initial MPS used for optimization; will be overwritten
        numsweeps: maximum number of left and right sweeps
        numiter_lanczos: number of local Lanczos iterations
        tol_split: tolerance for splitting merged MPS tensors

    Returns:
        numpy.ndarray: array of approximate ground state energies after each iteration
    """

    # number of lattice sites
    nsites = hamiltonian.nsites
    assert nsites == psi.nsites

    # right-normalize input matrix product state
    psi.orthonormalize(mode="right")

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    rblocks = compute_right_operator_blocks(psi, hamiltonian)
    lblocks = [None for _ in range(nsites)]
    lblocks[0] = np.array([[[1]]], dtype=rblocks[0].dtype)

    # consistency check
    for i, rb in enumerate(rblocks):
        assert is_qsparse(rb, (psi.qbonds[i+1], hamiltonian.qbonds[i+1], -psi.qbonds[i+1])), \
            "sparsity pattern of operator blocks must match quantum numbers"

    en_min = np.zeros(numsweeps)

    h2 = [mpo_merge_tensor_pair(hamiltonian.a[i], hamiltonian.a[i+1]) for i in range(nsites - 1)]

    # TODO: number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0

        # sweep from left to right
        for i in range(nsites - 2):
            # merge neighboring tensors
            a_cur = mps_merge_tensor_pair(psi.a[i], psi.a[i+1])
            # minimize local energy
            en, a_cur = _minimize_local_energy(
                h2[i], lblocks[i], rblocks[i+1], a_cur, numiter_lanczos)
            # split a_cur
            psi.a[i], psi.a[i+1], psi.qbonds[i+1] = mps_split_tensor_svd(
                a_cur, psi.qsite, psi.qsite, [psi.qbonds[i], psi.qbonds[i+2]],
                "right", tol=tol_split)
            # update the left blocks
            lblocks[i+1] = contraction_operator_step_left(
                psi.a[i], psi.a[i], hamiltonian.a[i], lblocks[i])

        # sweep from right to left
        for i in reversed(range(nsites - 1)):
            # merge neighboring tensors
            a_cur = mps_merge_tensor_pair(psi.a[i], psi.a[i+1])
            # minimize local energy
            en, a_cur = _minimize_local_energy(
                h2[i], lblocks[i], rblocks[i+1], a_cur, numiter_lanczos)
            # split a_cur
            psi.a[i], psi.a[i+1], psi.qbonds[i+1] = mps_split_tensor_svd(
                a_cur, psi.qsite, psi.qsite, [psi.qbonds[i], psi.qbonds[i+2]],
                "left", tol=tol_split)
            # update the right blocks
            rblocks[i] = contraction_operator_step_right(
                psi.a[i+1], psi.a[i+1], hamiltonian.a[i+1], rblocks[i+1])

        # right-normalize leftmost tensor to ensure that `psi` is normalized
        psi.a[0], _, psi.qbonds[0] = mps_local_orthonormalize_right_qr(
            psi.a[0], np.array([[[1]]]), psi.qsite, psi.qbonds[:2])

        # record energy after each sweep
        en_min[n] = en

    return en_min


def _minimize_local_energy(w, l, r, a_start, numiter: int):
    """
    Minimize single-site local energy by Lanczos iteration.
    """
    w, u_ritz = eigh_krylov(
        lambda x: apply_local_hamiltonian(x.reshape(a_start.shape), w, l, r).reshape(-1),
            a_start.reshape(-1), numiter, 1)
    a_opt = u_ritz[:, 0].reshape(a_start.shape)
    return w[0], a_opt
