import numpy as np
from .mps import MPS, merge_mps_tensor_pair, split_mps_tensor
from .mpo import MPO, merge_mpo_tensor_pair
from .operation import (
        contraction_operator_step_right,
        contraction_operator_step_left,
        compute_right_operator_blocks,
        apply_local_hamiltonian,
        apply_local_bond_contraction)
from .krylov import expm_krylov
from .qnumber import qnumber_flatten, is_qsparse
from .bond_ops import qr

__all__ = ['tdvp_singlesite', 'tdvp_twosite']


def tdvp_singlesite(H: MPO, psi: MPS, dt, numsteps: int, numiter_lanczos: int = 25):
    """
    Symmetric single-site TDVP integration.
    `psi` is overwritten in-place with the time-evolved state.

    Args:
        H: Hamiltonian as MPO
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
    L = H.nsites
    assert L == psi.nsites

    # right-normalize input matrix product state
    nrm = psi.orthonormalize(mode='right')

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    # consistency check
    for i in range(len(BR)):
        assert is_qsparse(BR[i], [psi.qD[i+1], H.qD[i+1], -psi.qD[i+1]]), \
            'sparsity pattern of operator blocks must match quantum numbers'

    for n in range(numsteps):

        # sweep from left to right
        for i in range(L - 1):
            # evolve psi.A[i] forward in time by half a time step
            psi.A[i] = _local_hamiltonian_step(BL[i], BR[i], H.A[i], psi.A[i], 0.5*dt, numiter_lanczos)
            # left-orthonormalize current psi.A[i]
            s = psi.A[i].shape
            (Q, C, psi.qD[i+1]) = qr(psi.A[i].reshape((s[0]*s[1], s[2])),
                                     qnumber_flatten([psi.qd, psi.qD[i]]), psi.qD[i+1])
            psi.A[i] = Q.reshape((s[0], s[1], Q.shape[1]))
            # update the left blocks
            BL[i+1] = contraction_operator_step_left(psi.A[i], psi.A[i], H.A[i], BL[i])
            # evolve C backward in time by half a time step
            C = _local_bond_step(BL[i+1], BR[i], C, -0.5*dt, numiter_lanczos)
            # update psi.A[i+1] tensor: multiply with C from left
            psi.A[i+1] = np.einsum(psi.A[i+1], (0, 3, 2), C, (1, 3), (0, 1, 2), optimize=True)

        # evolve psi.A[L-1] forward in time by a full time step
        i = L - 1
        psi.A[i] = _local_hamiltonian_step(BL[i], BR[i], H.A[i], psi.A[i], dt, numiter_lanczos)

        # sweep from right to left
        for i in reversed(range(1, L)):
            # right-orthonormalize current psi.A[i]
            # flip left and right virtual bond dimensions
            psi.A[i] = psi.A[i].transpose((0, 2, 1))
            # perform QR decomposition
            s = psi.A[i].shape
            (Q, C, qbond) = qr(psi.A[i].reshape((s[0]*s[1], s[2])),
                               qnumber_flatten([psi.qd, -psi.qD[i+1]]), -psi.qD[i])
            psi.qD[i] = -qbond
            # replace psi.A[i] by reshaped Q matrix and undo flip of left and right virtual bond dimensions
            psi.A[i] = Q.reshape((s[0], s[1], Q.shape[1])).transpose((0, 2, 1))
            # update the right blocks
            BR[i-1] = contraction_operator_step_right(psi.A[i], psi.A[i], H.A[i], BR[i])
            # evolve C backward in time by half a time step
            C = np.transpose(C)
            C = _local_bond_step(BL[i], BR[i-1], C, -0.5*dt, numiter_lanczos)
            # update psi.A[i-1] tensor: multiply with C from right
            psi.A[i-1] = np.einsum(psi.A[i-1], (0, 1, 3), C, (3, 2), (0, 1, 2), optimize=True)
            # evolve psi.A[i-1] forward in time by half a time step
            psi.A[i-1] = _local_hamiltonian_step(BL[i-1], BR[i-1], H.A[i-1], psi.A[i-1], 0.5*dt, numiter_lanczos)

    # return norm of initial psi
    return nrm


def tdvp_twosite(H: MPO, psi: MPS, dt, numsteps: int, numiter_lanczos: int = 25, tol_split = 0):
    """
    Symmetric two-site TDVP integration.
    `psi` is overwritten in-place with the time-evolved state.

    Args:
        H: Hamiltonian as MPO
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
    L = H.nsites
    assert L == psi.nsites
    assert L >= 2

    # right-normalize input matrix product state
    nrm = psi.orthonormalize(mode='right')

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    # consistency check
    for i in range(len(BR)):
        assert is_qsparse(BR[i], [psi.qD[i+1], H.qD[i+1], -psi.qD[i+1]]), \
            'sparsity pattern of operator blocks must match quantum numbers'

    for n in range(numsteps):

        # sweep from left to right
        for i in range(L - 2):
            # merge neighboring tensors
            Am = merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
            Hm = merge_mpo_tensor_pair(H.A[i], H.A[i+1])
            # evolve Am forward in time by half a time step
            Am = _local_hamiltonian_step(BL[i], BR[i+1], Hm, Am, 0.5*dt, numiter_lanczos)
            # split Am
            psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'right', tol=tol_split)
            # update the left blocks
            BL[i+1] = contraction_operator_step_left(psi.A[i], psi.A[i], H.A[i], BL[i])
            # evolve psi.A[i+1] backward in time by half a time step
            psi.A[i+1] = _local_hamiltonian_step(BL[i+1], BR[i+1], H.A[i+1], psi.A[i+1], -0.5*dt, numiter_lanczos)

        # rightmost tensor pair
        i = L - 2
        # merge neighboring tensors
        Am = merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
        Hm = merge_mpo_tensor_pair(H.A[i], H.A[i+1])
        # evolve Am forward in time by a full time step
        Am = _local_hamiltonian_step(BL[i], BR[i+1], Hm, Am, dt, numiter_lanczos)
        # split Am
        psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'left', tol=tol_split)
        # update the right blocks
        BR[i] = contraction_operator_step_right(psi.A[i+1], psi.A[i+1], H.A[i+1], BR[i+1])

        # sweep from right to left
        for i in reversed(range(L - 2)):
            # evolve psi.A[i+1] backward in time by half a time step
            psi.A[i+1] = _local_hamiltonian_step(BL[i+1], BR[i+1], H.A[i+1], psi.A[i+1], -0.5*dt, numiter_lanczos)
            # merge neighboring tensors
            Am = merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
            Hm = merge_mpo_tensor_pair(H.A[i], H.A[i+1])
            # evolve Am forward in time by half a time step
            Am = _local_hamiltonian_step(BL[i], BR[i+1], Hm, Am, 0.5*dt, numiter_lanczos)
            # split Am
            psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'left', tol=tol_split)
            # update the right blocks
            BR[i] = contraction_operator_step_right(psi.A[i+1], psi.A[i+1], H.A[i+1], BR[i+1])

    # return norm of initial psi
    return nrm


def _local_hamiltonian_step(L, R, W, A, dt, numiter: int):
    """
    Local time step effected by Hamiltonian, based on a Lanczos iteration.
    """
    return expm_krylov(
        lambda x: apply_local_hamiltonian(L, R, W, x.reshape(A.shape)).reshape(-1),
            A.reshape(-1), -dt, numiter, hermitian=True).reshape(A.shape)


def _local_bond_step(L, R, C, dt, numiter: int):
    """
    Local "zero-site" bond step, based on a Lanczos iteration.
    """
    return expm_krylov(
        lambda x: apply_local_bond_contraction(L, R, x.reshape(C.shape)).reshape(-1),
            C.reshape(-1), -dt, numiter, hermitian=True).reshape(C.shape)
