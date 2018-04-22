import numpy as np


class OpChain(object):
    """Operator chain op_i x op_{i+1} x ... x op_{i+n-1},
    with op_i acting on lattice site i."""

    def __init__(self, istart=0, oplist=[]):
        self.istart = istart
        self.oplist = oplist

    @property
    def iend(self):
        """Largest lattice index of operator chain."""
        return self.istart + len(self.oplist)

    @property
    def length(self):
        """Length of operator chain."""
        return len(self.oplist)

    def pad_identities_right(self, d, L):
        """Pad identity matrices on the right."""
        npad = L - self.iend
        # concatenate lists
        self.oplist += [np.identity(d) for _ in range(npad)]

    def pad_identities_left(self, d):
        """Pad identity matrices on the left."""
        # concatenate lists
        self.oplist = [np.identity(d) for _ in range(self.istart)] + self.oplist
        self.istart = 0

    def as_matrix(self, d, L):
        """Construct matrix representation on full Hilbert space (dimension d^L x d^L)."""
        A = np.identity(d**(self.istart))
        for op in self.oplist:
            A = np.kron(A, op)
        A = np.kron(A, np.identity(d**(L - self.iend)))
        return A
