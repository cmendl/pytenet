import numpy as np
from scipy.linalg import eigh_tridiagonal, expm
import warnings

__all__ = ['lanczos_iteration', 'arnoldi_iteration', 'eigh_krylov', 'expm_krylov']


def lanczos_iteration(Afunc, vstart, numiter):
    """
    Perform a "matrix free" Lanczos iteration.

    Args:
        Afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        tuple: tuple containing
          - alpha:      diagonal real entries of Hessenberg matrix
          - beta:       off-diagonal real entries of Hessenberg matrix
          - V:          `len(vstart) x numiter` matrix containing the orthonormal Lanczos vectors
    """
    # normalize starting vector
    nrmv = np.linalg.norm(vstart)
    assert nrmv > 0
    vstart /= nrmv

    alpha = np.zeros(numiter)
    beta  = np.zeros(numiter-1)

    V = np.zeros((numiter, len(vstart)), dtype=complex)
    V[0] = vstart

    for j in range(numiter-1):
        w = Afunc(V[j])
        alpha[j] = np.vdot(w, V[j]).real
        w -= alpha[j]*V[j] + (beta[j-1]*V[j-1] if j > 0 else 0)
        beta[j] = np.linalg.norm(w)
        if beta[j] < 100*len(vstart)*np.finfo(float).eps:
            warnings.warn(
                'beta[{}] ~= 0 encountered during Lanczos iteration.'.format(j),
                RuntimeWarning)
            # premature end of iteration
            numiter = j + 1
            return (alpha[:numiter], beta[:numiter-1], V[:numiter, :].T)
        V[j+1] = w / beta[j]

    # complete final iteration
    j = numiter-1
    w = Afunc(V[j])
    alpha[j] = np.vdot(w, V[j]).real
    return (alpha, beta, V.T)


def arnoldi_iteration(Afunc, vstart, numiter):
    """
    Perform a "matrix free" Arnoldi iteration.

    Args:
        Afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        tuple: tuple containing
          - H:      `numiter x numiter` upper Hessenberg matrix
          - V:      `len(vstart) x numiter` matrix containing the orthonormal Arnoldi vectors
    """
    # normalize starting vector
    nrmv = np.linalg.norm(vstart)
    assert nrmv > 0
    vstart /= nrmv

    H = np.zeros((numiter, numiter), dtype=complex)
    V = np.zeros((numiter, len(vstart)), dtype=complex)
    V[0] = vstart

    for j in range(numiter-1):
        w = Afunc(V[j])
        # subtract the projections on previous vectors
        for k in range(j+1):
            H[k, j] = np.vdot(V[k], w)
            w -= H[k, j]*V[k]
        H[j+1, j] = np.linalg.norm(w)
        if H[j+1, j] < 100*len(vstart)*np.finfo(float).eps:
            warnings.warn(
                'H[{}, {}] ~= 0 encountered during Arnoldi iteration.'.format(j+1, j),
                RuntimeWarning)
            # premature end of iteration
            numiter = j + 1
            return H[:numiter, :numiter], V[:numiter, :].T
        V[j+1] = w / H[j+1, j]

    # complete final iteration
    j = numiter-1
    w = Afunc(V[j])
    for k in range(j+1):
        H[k, j] = np.vdot(V[k], w)
        w -= H[k, j]*V[k]

    return H, V.T


def eigh_krylov(Afunc, vstart, numiter, numeig):
    """
    Compute Krylov subspace approximation of eigenvalues and vectors.
    """
    alpha, beta, V = lanczos_iteration(Afunc, vstart, numiter)

    # diagonalize Hessenberg matrix
    w_hess, u_hess = eigh_tridiagonal(alpha, beta)

    # compute Ritz eigenvectors
    u_ritz = np.dot(V, u_hess[:, 0:numeig])

    return (w_hess[0:numeig], u_ritz)


def expm_krylov(Afunc, v, dt, numiter, hermitian=False):
    """
    Compute Krylov subspace approximation of the matrix exponential
    applied to input vector: `expm(dt*A)*v`.

    Reference:
        M. Hochbruck and C. Lubich
        On Krylov subspace approximations to the matrix exponential operator
        SIAM J. Numer. Anal. 34, 1911 (1997)
    """
    if hermitian:
        alpha, beta, V = lanczos_iteration(Afunc, v, numiter)
        # diagonalize Hessenberg matrix
        w_hess, u_hess = eigh_tridiagonal(alpha, beta)
        return np.dot(V, np.dot(u_hess, np.linalg.norm(v) * np.exp(dt*w_hess) * u_hess[0]))
    else:
        H, V = arnoldi_iteration(Afunc, v, numiter)
        return np.dot(V, np.linalg.norm(v) * expm(dt*H)[:, 0])
