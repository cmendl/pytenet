from collections.abc import Sequence, Mapping
import numpy as np

__all__ = ['OpChain']


class OpChain:
    """
    Symbolic operator chain `coeff op_i x op_{i+1} x ... x op_{i+n-1}`,
    with `op_i` acting on lattice site `i`.

    A single bond quantum number is interleaved between each `op_i` and `op_{i+1}`;
    set all quantum numbers to zero to effectively disable them.
    """
    def __init__(self, oids: Sequence[int], qnums: Sequence[int], coeff: float, istart: int):
        """
        Create an operator chain.

        Args:
            oids: list of local op_i operator IDs
            qnums: interleaved bond quantum numbers, including a leading and trailing quantum number
            coeff: coefficient (scalar factor)
            istart: first lattice site the operator chain acts on
        """
        if len(oids) + 1 != len(qnums):
            raise ValueError('incompatible lengths of operator and quantum number lists')
        if istart < 0:
            raise ValueError('start index cannot be negative')
        self.oids  = list(oids)
        self.qnums = list(qnums)
        self.coeff = coeff
        self.istart = istart

    @property
    def length(self) -> int:
        """
        Length of operator chain.
        """
        return len(self.oids)

    def as_matrix(self, opmap: Mapping) -> np.ndarray:
        """
        Represent the logical operation of the operator chain as a matrix.
        """
        op = self.coeff * np.identity(1)
        for oid in self.oids:
            op = np.kron(op, opmap[oid])
        return op
