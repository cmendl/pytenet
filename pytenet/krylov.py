"""
Krylov subspace algorithms.
"""

import warnings
import numpy as np
from scipy.linalg import expm

__all__ = ["lanczos_iteration", "arnoldi_iteration", "eigh_krylov", "expm_krylov"]


def lanczos_iteration(afunc, vstart, numiter):
    """
    Perform a "matrix free" Lanczos iteration.

    Args:
        afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        tuple: tuple containing
          - alpha:      diagonal real entries of Hessenberg matrix
          - beta:       off-diagonal real entries of Hessenberg matrix
          - v:          `len(vstart) x numiter` matrix containing the orthonormal Lanczos vectors
    """
    vstart = np.asarray(vstart)
    # normalize starting vector
    nrmv = np.linalg.norm(vstart)
    assert nrmv > 0
    vstart = vstart / nrmv

    alpha = np.zeros(numiter)
    beta  = np.zeros(numiter-1)

    v = np.zeros((numiter, len(vstart)), dtype=vstart.dtype)
    v[0] = vstart

    for j in range(numiter-1):
        w = afunc(v[j])
        alpha[j] = np.vdot(w, v[j]).real
        w -= alpha[j]*v[j] + (beta[j-1]*v[j-1] if j > 0 else 0)
        beta[j] = np.linalg.norm(w)
        if beta[j] < 100*len(vstart)*np.finfo(float).eps:
            warnings.warn(
                f"beta[{j}] ~= 0 encountered during Lanczos iteration.",
                RuntimeWarning)
            # premature end of iteration
            numiter = j + 1
            return (alpha[:numiter], beta[:numiter-1], v[:numiter, :].T)
        v[j+1] = w / beta[j]

    # complete final iteration
    j = numiter-1
    w = afunc(v[j])
    alpha[j] = np.vdot(w, v[j]).real
    return (alpha, beta, v.T)


def arnoldi_iteration(afunc, vstart, numiter):
    """
    Perform a "matrix free" Arnoldi iteration.

    Args:
        afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        tuple: tuple containing
          - hess:   `numiter x numiter` upper Hessenberg matrix
          - v:      `len(vstart) x numiter` matrix containing the orthonormal Arnoldi vectors
    """
    vstart = np.asarray(vstart)
    # normalize starting vector
    nrmv = np.linalg.norm(vstart)
    assert nrmv > 0
    vstart = vstart / nrmv

    hess = np.zeros((numiter, numiter), dtype=vstart.dtype)
    v = np.zeros((numiter, len(vstart)), dtype=vstart.dtype)
    v[0] = vstart

    for j in range(numiter-1):
        w = afunc(v[j])
        # subtract the projections on previous vectors
        for k in range(j+1):
            hess[k, j] = np.vdot(v[k], w)
            w -= hess[k, j]*v[k]
        hess[j+1, j] = np.linalg.norm(w)
        if hess[j+1, j] < 100*len(vstart)*np.finfo(float).eps:
            warnings.warn(
                f"H[{j+1}, {j}] ~= 0 encountered during Arnoldi iteration.",
                RuntimeWarning)
            # premature end of iteration
            numiter = j + 1
            return hess[:numiter, :numiter], v[:numiter, :].T
        v[j+1] = w / hess[j+1, j]

    # complete final iteration
    j = numiter-1
    w = afunc(v[j])
    for k in range(j+1):
        hess[k, j] = np.vdot(v[k], w)
        w -= hess[k, j]*v[k]

    return hess, v.T


def eigh_krylov(afunc, vstart, numiter, numeig):
    """
    Compute Krylov subspace approximation of eigenvalues and vectors.
    """
    alpha, beta, v = lanczos_iteration(afunc, vstart, numiter)
    # diagonalize Hessenberg matrix
    w_hess, u_hess = eigh_tridiag(alpha, beta)
    # compute Ritz eigenvectors
    u_ritz = v @ u_hess[:, 0:numeig]
    return (w_hess[0:numeig], u_ritz)


def expm_krylov(afunc, vec, dt, numiter, hermitian=False):
    """
    Compute Krylov subspace approximation of the matrix exponential
    applied to input vector: `expm(dt*A)*vec`.

    Reference:
        M. Hochbruck and C. Lubich
        On Krylov subspace approximations to the matrix exponential operator
        SIAM J. Numer. Anal. 34, 1911 (1997)
    """
    if hermitian:
        alpha, beta, v = lanczos_iteration(afunc, vec, numiter)
        # diagonalize Hessenberg matrix
        w_hess, u_hess = eigh_tridiag(alpha, beta)
        return v @ (u_hess @ (np.linalg.norm(vec) * np.exp(dt*w_hess) * u_hess[0]))
    else:
        hess, v = arnoldi_iteration(afunc, vec, numiter)
        return v @ (np.linalg.norm(vec) * expm(dt*hess)[:, 0])


def eigh_tridiag(d, e):
    """
    Solve the eigenvalue problem for a real symmetric tridiagonal matrix.

    This function is a work-around of `scipy.linalg.eigh_tridiagonal`,
    which runs into convergence problems in some cases.
    """
    a = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    return np.linalg.eigh(a)
