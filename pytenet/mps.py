import numpy as np


class MPS(object):
    """Matrix product state (MPS) class.

    The i-th MPS tensor has dimension [d, D[i], D[i+1]] with d the physical dimension
    and D the list of virtual bond dimensions.
    """

    def __init__(self, d, D, fill=0.0):
        # create list of MPS tensors
        if isinstance(fill, int) or isinstance(fill, float) or isinstance(fill, complex):
            self.A = [np.full((d, D[i], D[i+1]), fill) for i in range(len(D)-1)]
        elif fill == 'random':
            # random complex entries
            self.A = [
                    np.random.normal(size=(d, D[i], D[i+1]), scale=1./np.sqrt(d*D[i]*D[i+1])) +
                 1j*np.random.normal(size=(d, D[i], D[i+1]), scale=1./np.sqrt(d*D[i]*D[i+1])) for i in range(len(D)-1)]
        else:
            raise ValueError('fill = {} invalid; must be a number or "random".'.format(fill))

    @property
    def nsites(self):
        """Number of lattice sites."""
        return len(self.A)

    @property
    def bond_dims(self):
        """Virtual bond dimensions."""
        if len(self.A) == 0:
            return []
        else:
            D = [self.A[i].shape[1] for i in range(len(self.A))]
            D.append(self.A[-1].shape[2])
            return D

    def orthonormalize(self, mode='left'):
        """Left- or right-orthonormalize the MPS."""
        if len(self.A) == 0:
            return

        if mode == 'left':
            for i in range(len(self.A) - 1):
                self.A[i], self.A[i+1] = local_orthonormalize_left_qr(self.A[i], self.A[i+1])
            # last tensor
            self.A[-1], T = local_orthonormalize_left_qr(self.A[-1], np.array([[[1]]]))
            # return normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            return T[0, 0, 0].real
        elif mode == 'right':
            for i in reversed(range(1, len(self.A))):
                self.A[i], self.A[i-1] = local_orthonormalize_right_qr(self.A[i], self.A[i-1])
            # first tensor
            self.A[0], T = local_orthonormalize_right_qr(self.A[0], np.array([[[1]]]))
            # return normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            return T[0, 0, 0].real
        else:
            raise ValueError('mode = {} invalid; must be "left" or "right".'.format(mode))

    def as_vector(self):
        """Merge all tensors to obtain the vector representation on the full Hilbert space."""
        psi = self.A[0]
        for i in range(1, len(self.A)):
            psi = _merge_MPS_tensor_pair(psi, self.A[i])
        assert psi.ndim == 3
        assert psi.shape[1] == 1 and psi.shape[2] == 1
        psi = psi.flatten()
        return psi


def local_orthonormalize_left_qr(A, Anext):
    """Left-orthonormalize local site tensor A by a QR decomposition,
    and update tensor at next site."""
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    Q, R = np.linalg.qr(np.reshape(A, (s[0]*s[1], s[2])), mode='reduced')
    A = np.reshape(Q, (s[0], s[1], Q.shape[1]))
    # update Anext tensor: multiply with R from left
    Anext = np.tensordot(R, Anext, (1, 1)).transpose((1, 0, 2))
    return (A, Anext)


def local_orthonormalize_right_qr(A, Aprev):
    """Right-orthonormalize local site tensor A by a QR decomposition,
    and update tensor at previous site."""
    # flip left and right virtual bond dimensions
    A = A.transpose((0, 2, 1))
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    Q, R = np.linalg.qr(np.reshape(A, (s[0]*s[1], s[2])), mode='reduced')
    A = np.reshape(Q, (s[0], s[1], Q.shape[1])).transpose((0, 2, 1))
    # update Aprev tensor: multiply with R from right
    Aprev = np.tensordot(Aprev, R, (2, 1))
    return (A, Aprev)


def _merge_MPS_tensor_pair(A0, A1):
    """Merge two neighboring MPS tensors."""
    A = np.tensordot(A0, A1, (2, 1))
    # pair original physical dimensions of A0 and A1
    A = A.transpose((0, 2, 1, 3))
    # combine original physical dimensions
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2], A.shape[3]))
    return A
