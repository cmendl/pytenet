import numpy as np
from .operation import (
        contraction_operator_step_right,
        contraction_operator_step_left,
        compute_right_operator_blocks,
        apply_local_hamiltonian,
        apply_local_bond_contraction)
from .krylov import expm_krylov
from .qnumber import qnumber_flatten, is_qsparse
from .bond_ops import qr

__all__ = ['integrate_local_singlesite']


def integrate_local_singlesite(H, psi, dt, numsteps, numiter_lanczos=25):
    """
    Symmetric single-site integration.

    Args:
        H: Hamiltonian as MPO
        psi: initial state as MPS
        dt: time step (without imaginary factor; for real-time evolution
            use purely imaginary dt)
        numsteps: number of time steps
        numiter_lanczos: number of Lanczos iterations for each site-local step

    Returns:
        norm of initial psi;
        psi is overwritten in-place with time-evolved state

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
            BL[i+1] = contraction_operator_step_left(psi.A[i], H.A[i], BL[i])
            # evolve C backward in time by half a time step
            C = _local_bond_step(BL[i+1], BR[i], C, -0.5*dt, numiter_lanczos)
            # update psi.A[i+1] tensor: multiply with C from left
            psi.A[i+1] = np.tensordot(C, psi.A[i+1], (1, 1)).transpose((1, 0, 2))

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
            BR[i-1] = contraction_operator_step_right(psi.A[i], H.A[i], BR[i])
            # evolve C backward in time by half a time step
            C = np.transpose(C)
            C = _local_bond_step(BL[i], BR[i-1], C, -0.5*dt, numiter_lanczos)
            # update psi.A[i-1] tensor: multiply with C from right
            psi.A[i-1] = np.tensordot(psi.A[i-1], C, 1)
            # evolve psi.A[i-1] forward in time by half a time step
            psi.A[i-1] = _local_hamiltonian_step(BL[i-1], BR[i-1], H.A[i-1], psi.A[i-1], 0.5*dt, numiter_lanczos)

    # return norm of initial psi
    return nrm


def _local_hamiltonian_step(L, R, W, A, dt, numiter):
    """Local time step effected by Hamiltonian, based on a Lanczos iteration."""
    return expm_krylov(
        lambda x: apply_local_hamiltonian(L, R, W, x.reshape(A.shape)).flatten(),
            A.flatten(), -dt, numiter).reshape(A.shape)


def _local_bond_step(L, R, C, dt, numiter):
    """Local "zero-site" bond step, based on a Lanczos iteration."""
    return expm_krylov(
        lambda x: apply_local_bond_contraction(L, R, x.reshape(C.shape)).flatten(),
            C.flatten(), -dt, numiter).reshape(C.shape)
