import numpy as np

__all__ = ['iMPS', 'left_orthonormalize_uniform', 'right_orthonormalize_uniform']


class iMPS(object):
    """
    Uniform ("infinite") matrix product state (MPS) class.
    """

    def __init__(self, qd, qD, fill=0.0):
        """
        Create a matrix product state.

        Args:
            qd: physical quantum numbers at each site
        """
        # require NumPy arrays
        self.qd = np.array(qd)
        self.qD = np.array(qD)
        d = len(qd)
        D = len(qD)
        self.A = np.full((d, D, D), fill)


def left_orthonormalize_uniform(A):
    """
    Left-orthonormalize a uniform MPS tensor (direct numerical solution, as reference).

    Args:
        A: input MPS tensor, of shape (d, D, D)

    Returns:
        tuple: tuple containing
          - Aleft: left-orthonormal uniform MPS tensor describing same state as A
          - L: corresponding L matrix such that Aleft == L A L^{-1}
          - nrm: normalization factor of A, such that dominant
                 eigenvalue of transfer matrix is 1

    Reference:
        L. Vanderstraeten, J. Haegeman, F. Verstraete
        Tangent-space methods for uniform matrix product states
        arXiv:1810.07006
    """
    assert A.ndim == 3

    s = A.shape
    assert s[1] == s[2], 'left and right virtual bond dimensions must agree'
    D = s[1]

    # construct transfer matrix
    E = np.tensordot(A, A.conj(), axes=(0, 0)).transpose((0, 2, 1, 3)).reshape((D**2, D**2))

    # dominant left eigenvector
    w, v = np.linalg.eig(E.T)
    i = np.argmax(abs(w))

    # normalization factor
    nrm = np.sqrt(abs(w[i]))
    A = A / nrm

    # eigenvector (independent of rescaling)
    l = v[:, i].reshape((D, D))
    # l must be Hermitian
    assert np.linalg.norm(l - l.conj().T) < len(l) * 1e-14
    L = np.linalg.cholesky(l).T

    # L A L^{-1}, with L acting on virtual bond dimensions
    s = A.shape
    LA = np.tensordot(L, A, axes=(1, 1)).transpose((1, 0, 2))
    Aleft = np.linalg.solve(L.T, LA.transpose((2, 0, 1)).reshape((s[2], s[0]*s[1]))).reshape((s[2], s[0], s[1])).transpose((1, 2, 0))

    return Aleft, L, nrm


def right_orthonormalize_uniform(A):
    """
    Right-orthonormalize a uniform MPS tensor (direct numerical solution, as reference).

    Args:
        A: input MPS tensor, of shape (d, D, D)

    Returns:
        tuple: tuple containing
          - Aright: right-orthonormal uniform MPS tensor describing same state as A
          - R: corresponding R matrix such that Aright == R^{-1} A R
          - nrm: normalization factor of A, such that dominant
                 eigenvalue of transfer matrix is 1

    Reference:
        L. Vanderstraeten, J. Haegeman, F. Verstraete
        Tangent-space methods for uniform matrix product states
        arXiv:1810.07006
    """
    assert A.ndim == 3

    s = A.shape
    assert s[1] == s[2], 'left and right virtual bond dimensions must agree'
    D = s[1]

    # construct transfer matrix
    E = np.tensordot(A, A.conj(), axes=(0, 0)).transpose((0, 2, 1, 3)).reshape((D**2, D**2))

    # dominant right eigenvector
    w, v = np.linalg.eig(E)
    i = np.argmax(abs(w))

    # normalization factor
    nrm = np.sqrt(abs(w[i]))
    A = A / nrm

    # eigenvector (independent of rescaling)
    r = v[:, i].reshape((D, D))
    # r must be Hermitian
    assert np.linalg.norm(r - r.conj().T) < len(r) * 1e-14
    R = np.linalg.cholesky(r)

    # R^{-1} A R, with R acting on virtual bond dimensions
    s = A.shape
    AR = np.tensordot(A, R, axes=(2, 0))
    Aright = np.linalg.solve(R, AR.transpose((1, 0, 2)).reshape((s[1], s[0]*s[2]))).reshape((s[1], s[0], s[2])).transpose((1, 0, 2))

    return Aright, R, nrm
