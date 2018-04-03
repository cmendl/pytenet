import numpy as np


def _merge_MPS_tensor_pair(A0, A1):
    """Merge two neighboring MPS tensors."""
    A = np.tensordot(A0, A1, (2, 1))
    # pair original physical dimensions of A0 and A1
    A = A.transpose((0, 2, 1, 3))
    # combine original physical dimensions
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2], A.shape[3]))
    return A


class MPS:
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
            C = np.identity(self.A[0].shape[1])
            for i in range(len(self.A)):
                # multiply with C from left
                B = np.tensordot(C, self.A[i], (1, 1))
                B = B.transpose((1, 0, 2))
                s = B.shape
                assert len(s) == 3
                Q, C = np.linalg.qr(np.reshape(B, (s[0]*s[1], s[2])), mode='reduced')
                self.A[i] = np.reshape(Q, (s[0], s[1], Q.shape[1]))
            # return normalization factor
            assert C.shape == (1, 1)
            return C[0, 0]
        elif mode == 'right':
            C = np.identity(self.A[-1].shape[2])
            for i in reversed(range(len(self.A))):
                # multiply with C from right
                B = np.tensordot(self.A[i], C, (2, 0))
                # flip left and right virtual bond dimensions
                B = B.transpose((0, 2, 1))
                s = B.shape
                assert len(s) == 3
                Q, C = np.linalg.qr(np.reshape(B, (s[0]*s[1], s[2])), mode='reduced')
                self.A[i] = np.reshape(Q, (s[0], s[1], Q.shape[1])).transpose((0, 2, 1))
                C = C.T
            # return normalization factor
            assert C.shape == (1, 1)
            return C[0, 0]
        else:
            raise ValueError('mode = {} invalid; must be "left" or "right".'.format(mode))

    def merge_full(self):
        """Merge all tensors to obtain the vector representation on the full Hilbert space.
        (Only use for small number of lattice sites!)"""
        psi = self.A[0]
        for i in range(1, len(self.A)):
            psi = _merge_MPS_tensor_pair(psi, self.A[i])
        assert len(psi.shape) == 3
        assert psi.shape[1] == 1 and psi.shape[2] == 1
        psi = psi.flatten()
        return psi
