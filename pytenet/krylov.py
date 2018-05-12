import numpy as np


def lanczos_iteration(Afunc, vstart, numiter):
    """Perform a "matrix free" Lanczos iteration.

    Args:
        Afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        alpha:      diagonal real entries of Hessenberg matrix
        beta:       off-diagonal real entries of Hessenberg matrix
        V:          len(vstart) x numiter matrix containing the orthonormal Lanczos vectors
    """

    # normalize starting vector
    assert np.linalg.norm(vstart) > 0
    vstart = vstart / np.linalg.norm(vstart)

    alpha = np.zeros(numiter)
    beta  = np.zeros(numiter-1)

    V = np.zeros((numiter, len(vstart)), dtype=complex)
    V[0] = vstart

    for j in range(numiter-1):
        w = Afunc(V[j])
        alpha[j] = np.vdot(w, V[j]).real
        w -= alpha[j]*V[j] + (beta[j-1]*V[j-1] if j > 0 else 0)
        beta[j] = np.linalg.norm(w)
        assert beta[j] > 0
        V[j+1] = w / beta[j]

    # complete final iteration
    j = numiter-1
    w = Afunc(V[j])
    alpha[j] = np.vdot(w, V[j]).real

    return (alpha, beta, V.T)
