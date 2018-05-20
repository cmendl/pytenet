import numpy as np
import mps
import operation
import krylov


def calculate_ground_state_local_singlesite(H, psi, numsweeps, numiter_lanczos=25):
    """Approximate the ground state MPS by left and right sweeps and local single-site optimizations;
    virtual bond dimensions of starting state 'psi' can only decrease.

    Args:
        H:                  Hamiltonian as MPO
        psi:                initial MPS used for optimization; will be overwritten
        numsweeps:          maximum number of left and right sweeps
        numiter_lanczos:    number of local Lanczos iterations
    Returns:
        en_min:             array of approximate ground state energies after each iteration

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
    # initialize left blocks by 1x1x1 identity (only leftmost block actually used)
    BL = [np.array([[[1]]], dtype=complex) for _ in range(L)]
    BR = operation.compute_right_operator_blocks(psi, H)

    en_min = np.zeros(numsweeps)

    # TODO: number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0

        # sweep from left to right
        for i in range(L - 1):
            en, psi.A[i] = _minimize_local_energy(BL[i], BR[i], H.A[i], psi.A[i], numiter_lanczos)
            # left-orthonormalize current psi.A[i]
            psi.A[i], psi.A[i+1] = mps.local_orthonormalize_left_qr(psi.A[i], psi.A[i+1])
            # update the left blocks
            BL[i+1] = operation.contraction_operator_step_left(psi.A[i], H.A[i], BL[i])

        # sweep from right to left
        for i in reversed(range(1, L)):
            en, psi.A[i] = _minimize_local_energy(BL[i], BR[i], H.A[i], psi.A[i], numiter_lanczos)
            # right-orthonormalize current psi.A[i]
            psi.A[i], psi.A[i-1] = mps.local_orthonormalize_right_qr(psi.A[i], psi.A[i-1])
            # update the right blocks
            BR[i-1] = operation.contraction_operator_step_right(psi.A[i], H.A[i], BR[i])

        # right-normalize leftmost tensor to ensure that 'psi' is normalized
        psi.A[0], _ = mps.local_orthonormalize_right_qr(psi.A[0], np.array([[[1]]]))

        # record energy after each sweep
        en_min[n] = en

    return en_min


def _minimize_local_energy(L, R, W, Astart, numiter):
    """Minimize site-local energy by Lanczos iteration."""

    w, u_ritz = krylov.eigh(
        lambda x: operation.apply_local_hamiltonian(L, R, W, np.reshape(x, Astart.shape)).flatten(),
        Astart.flatten(), numiter, 1)

    Aopt = np.reshape(u_ritz[:, 0], Astart.shape)

    return (w[0], Aopt)
