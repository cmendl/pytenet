import numpy as np

__all__ = ['OpChain']


class OpChain(object):
    """
    Operator chain `op_i x op_{i+1} x ... x op_{i+n-1}`,
    with `op_i` acting on lattice site `i`.

    A single bond quantum number is interleaved between each `op_i` and `op_{i+1}`;
    set all quantum numbers to zero to effectively disable them.
    """

    def __init__(self, oplist, qD, istart=0):
        """
        Create an operator chain.

        Args:
            oplist: list of the local op_i operators
            qD: list of bond quantum numbers
            istart: first lattice site the operator chain acts on
        """
        assert len(oplist) == len(qD) + 1
        self.oplist = oplist
        self.qD = list(qD)
        self.istart = istart

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
        self.qD     += npad*[0]

    def pad_identities_left(self, d):
        """Pad identity matrices on the left."""
        # concatenate lists
        self.oplist = [np.identity(d) for _ in range(self.istart)] + self.oplist
        self.qD     = self.istart*[0] + self.qD
        self.istart = 0

    def as_matrix(self, d, L):
        """Construct matrix representation on full Hilbert space (dimension `d^L x d^L`)."""
        if len(self.oplist) == 0:
            A = np.zeros((d**L, d**L))
        else:
            A = np.identity(d**(self.istart))
            for op in self.oplist:
                A = np.kron(A, op)
            A = np.kron(A, np.identity(d**(L - self.iend)))
        return A
