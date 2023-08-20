import numpy as np
from .mps import MPS, local_orthonormalize_left_qr, local_orthonormalize_right_qr, merge_mps_tensor_pair, split_mps_tensor
from .mpo import MPO, merge_mpo_tensor_pair
from .operation import (
        contraction_operator_step_right,
        contraction_operator_step_left,
        compute_right_operator_blocks,
        apply_local_hamiltonian)
from .krylov import eigh_krylov
from .qnumber import is_qsparse

__all__ = ['calculate_ground_state_local_singlesite', 'calculate_ground_state_local_twosite']


def calculate_ground_state_local_singlesite(H: MPO, psi: MPS, numsweeps: int, numiter_lanczos: int = 25):
    """
    Approximate the ground state MPS by left and right sweeps and local single-site optimizations
    (single-site DMRG algorithm). Virtual bond dimensions of starting state `psi` can only decrease.

    Args:
        H: Hamiltonian as MPO
        psi: initial MPS used for optimization; will be overwritten
        numsweeps: maximum number of left and right sweeps
        numiter_lanczos: number of local Lanczos iterations

    Returns:
        numpy.ndarray: array of approximate ground state energies after each iteration

    Reference:
        Ulrich Schollw"ock
        The density-matrix renormalization group in the age of matrix product states
        Annals of Physics 326, 96-192 (2011)
    """

    # number of lattice sites
    L = H.nsites
    assert L == psi.nsites

    # right-normalize input matrix product state
    psi.orthonormalize(mode='right')

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    # consistency check
    for i in range(len(BR)):
        assert is_qsparse(BR[i], [psi.qD[i+1], H.qD[i+1], -psi.qD[i+1]]), \
            'sparsity pattern of operator blocks must match quantum numbers'

    en_min = np.zeros(numsweeps)

    # TODO: number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0

        # sweep from left to right
        for i in range(L - 1):
            en, psi.A[i] = _minimize_local_energy(BL[i], BR[i], H.A[i], psi.A[i], numiter_lanczos)
            # left-orthonormalize current psi.A[i]
            psi.A[i], psi.A[i+1], psi.qD[i+1] = local_orthonormalize_left_qr(
                                        psi.A[i], psi.A[i+1], psi.qd, psi.qD[i:i+2])
            # update the left blocks
            BL[i+1] = contraction_operator_step_left(psi.A[i], H.A[i], BL[i])

        # sweep from right to left
        for i in reversed(range(1, L)):
            en, psi.A[i] = _minimize_local_energy(BL[i], BR[i], H.A[i], psi.A[i], numiter_lanczos)
            # right-orthonormalize current psi.A[i]
            psi.A[i], psi.A[i-1], psi.qD[i] = local_orthonormalize_right_qr(
                                        psi.A[i], psi.A[i-1], psi.qd, psi.qD[i:i+2])
            # update the right blocks
            BR[i-1] = contraction_operator_step_right(psi.A[i], H.A[i], BR[i])

        # right-normalize leftmost tensor to ensure that 'psi' is normalized
        psi.A[0], _, psi.qD[0] = local_orthonormalize_right_qr(
                                psi.A[0], np.array([[[1]]]), psi.qd, psi.qD[:2])

        # record energy after each sweep
        en_min[n] = en

    return en_min


def calculate_ground_state_local_twosite(H: MPO, psi: MPS, numsweeps: int, numiter_lanczos: int = 25, tol_split: float = 0):
    """
    Approximate the ground state MPS by left and right sweeps and local two-site optimizations
    (two-site DMRG algorithm). Virtual bond dimensions of starting state `psi` can only decrease.

    Args:
        H: Hamiltonian as MPO
        psi: initial MPS used for optimization; will be overwritten
        numsweeps: maximum number of left and right sweeps
        numiter_lanczos: number of local Lanczos iterations

    Returns:
        numpy.ndarray: array of approximate ground state energies after each iteration

    Reference:
        Ulrich Schollw"ock
        The density-matrix renormalization group in the age of matrix product states
        Annals of Physics 326, 96-192 (2011)
    """

    # number of lattice sites
    L = H.nsites
    assert L == psi.nsites

    # right-normalize input matrix product state
    psi.orthonormalize(mode='right')

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    # consistency check
    for i in range(len(BR)):
        assert is_qsparse(BR[i], [psi.qD[i+1], H.qD[i+1], -psi.qD[i+1]]), \
            'sparsity pattern of operator blocks must match quantum numbers'

    en_min = np.zeros(numsweeps)

    # TODO: number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0

        # sweep from left to right
        for i in range(L - 2):
            # merge neighboring tensors
            Am = merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
            Hm = merge_mpo_tensor_pair(H.A[i], H.A[i+1])
            # minimize local energy
            en, Am = _minimize_local_energy(BL[i], BR[i+1], Hm, Am, numiter_lanczos)
            # split Am
            psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'right', tol=tol_split)
            # update the left blocks
            BL[i+1] = contraction_operator_step_left(psi.A[i], H.A[i], BL[i])

        # sweep from right to left
        for i in reversed(range(L - 1)):
            # merge neighboring tensors
            Am = merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
            Hm = merge_mpo_tensor_pair(H.A[i], H.A[i+1])
            # minimize local energy
            en, Am = _minimize_local_energy(BL[i], BR[i+1], Hm, Am, numiter_lanczos)
            # split Am
            psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'left', tol=tol_split)
            # update the right blocks
            BR[i] = contraction_operator_step_right(psi.A[i+1], H.A[i+1], BR[i+1])

        # right-normalize leftmost tensor to ensure that 'psi' is normalized
        psi.A[0], _, psi.qD[0] = local_orthonormalize_right_qr(
                                psi.A[0], np.array([[[1]]]), psi.qd, psi.qD[:2])

        # record energy after each sweep
        en_min[n] = en

    return en_min


def _minimize_local_energy(L, R, W, Astart, numiter: int):
    """
    Minimize single-site local energy by Lanczos iteration.
    """
    w, u_ritz = eigh_krylov(
        lambda x: apply_local_hamiltonian(L, R, W, x.reshape(Astart.shape)).reshape(-1),
            Astart.reshape(-1), numiter, 1)

    Aopt = u_ritz[:, 0].reshape(Astart.shape)

    return w[0], Aopt
