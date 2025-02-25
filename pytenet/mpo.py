from collections.abc import Sequence, Mapping
import numpy as np
from scipy import sparse
from .qnumber import qnumber_outer_sum, qnumber_flatten, is_qsparse
from .bond_ops import qr
from .opgraph import OpGraph
from .util import crandn

__all__ = ['MPO', 'merge_mpo_tensor_pair']


class MPO:
    """
    Matrix product operator (MPO) class.

    The i-th MPO tensor has dimension `[d, d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.

    Quantum numbers are assumed to be additive and stored as integers.
    `qd` stores the list of physical quantum numbers at each site (assumed to agree
    for first and second physical dimension), and `qD` the virtual bond quantum numbers.
    The sum of first physical and left virtual bond quantum number of each
    non-zero tensor entry must be equal to the sum of second physical and
    right virtual bond quantum number.
    """

    def __init__(self, qd: Sequence[int], qD: Sequence[Sequence[int]], fill=0.0, rng: np.random.Generator=None):
        """
        Create a matrix product operator.

        Args:
            qd: physical quantum numbers at each site (same for all sites)
            qD: virtual bond quantum numbers (list of quantum number lists)
            fill: explicit scalar number to fill MPO tensors with, or
                  'random' to initialize tensors with random complex entries, or
                  'postpone' to leave MPO tensors unallocated
            rng: (optional) random number generator for drawing entries
        """
        # require NumPy arrays
        self.qd = np.array(qd)
        self.qD = [np.array(qDi) for qDi in qD]
        # create list of MPS tensors
        d = len(qd)
        D = [len(qDi) for qDi in qD]
        if isinstance(fill, (int, float, complex)):
            self.A = [np.full((d, d, D[i], D[i+1]), fill) for i in range(len(D)-1)]
        elif fill == 'random':
            # random complex entries
            if rng is None:
                rng = np.random.default_rng()
            self.A = [crandn((d, d, D[i], D[i+1]), rng) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)]
        elif fill == 'random real':
            # random real entries
            if rng is None:
                rng = np.random.default_rng()
            self.A = [rng.normal(size=(d, d, D[i], D[i+1])) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)]
        elif fill == 'postpone':
            self.A = (len(D) - 1) * [None]
        else:
            raise ValueError(f'fill = {fill} invalid; must be a number, "random", "random real" or "postpone".')
        if fill != 'postpone':
            # enforce block sparsity structure dictated by quantum numbers
            for i in range(len(self.A)):
                mask = qnumber_outer_sum([self.qd, -self.qd, self.qD[i], -self.qD[i+1]])
                self.A[i] = np.where(mask == 0, self.A[i], 0)

    @classmethod
    def identity(cls, qd: Sequence[int], L: int, scale: float = 1, dtype=float):
        """
        Construct MPO representation of the identity operation.
        """
        d = len(qd)
        mpo = cls(qd, (L+1)*[[0]])
        for i in range(L):
            mpo.A[i] = scale * np.identity(d, dtype=dtype).reshape((d, d, 1, 1))
        return mpo

    @classmethod
    def from_opgraph(cls, qd: Sequence[int], graph: OpGraph, opmap: Mapping, compute_nid_map=False):
        """
        Construct a MPO from an operator graph.

        Args:
            qd: physical quantum numbers at each site (same for all sites)
            graph: symbolic operator graph
            opmap: local operators as dictionary, using operator IDs as keys
            compute_nid_map: whether to construct the map from node IDs to bond location and index

        Returns:
            MPO: MPO representation of the operator graph
        """
        d = len(qd)
        if d == 0:
            raise ValueError('require at least one physical quantum number')
        Alist = []
        qD = []
        # node IDs at current bond site
        nids0 = [graph.nid_terminal[0]]
        qD.append([graph.nodes[graph.nid_terminal[0]].qnum])
        if compute_nid_map:
            nid_map = {}
            nid_map[graph.nid_terminal[0]] = (0, 0)
            l = 1
        while True:
            # node IDs at next bond site
            nids1 = []
            for nid in nids0:
                node = graph.nodes[nid]
                for eid in node.eids[1]:
                    edge = graph.edges[eid]
                    assert edge.nids[0] == nid
                    if edge.nids[1] not in nids1:
                        nids1.append(edge.nids[1])
            if not nids1:   # reached final site
                break
            # sort by node ID
            nids1 = sorted(nids1)
            qD.append([graph.nodes[nid].qnum for nid in nids1])
            if compute_nid_map:
                for i, nid in enumerate(nids1):
                    # record bond information (site location and index)
                    nid_map[nid] = (l, i)
            A = np.zeros((d, d, len(nids0), len(nids1)))
            for i, nid in enumerate(nids0):
                node = graph.nodes[nid]
                for eid in node.eids[1]:
                    edge = graph.edges[eid]
                    j = nids1.index(edge.nids[1])
                    # update local operator in MPO tensor (supporting multiple edges between same pair of nodes)
                    dAij = sum(c * opmap[k] for k, c in edge.opics)
                    if np.iscomplexobj(dAij):
                        A = A.astype(complex)
                    A[:, :, i, j] += dAij
            Alist.append(A)
            nids0 = nids1
            if compute_nid_map:
                # next bond
                l += 1
        assert len(Alist) + 1 == len(qD)
        op = cls(qd, qD, fill='postpone')
        op.A = Alist
        # consistency check
        for i in range(op.nsites):
            assert is_qsparse(op.A[i], [op.qd, -op.qd, op.qD[i], -op.qD[i+1]]), \
                'sparsity pattern of MPO tensor does not match quantum numbers'
        # store node ID map in MPO
        if compute_nid_map:
            op.nid_map = nid_map
        return op

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return len(self.A)

    @property
    def bond_dims(self) -> list:
        """
        Virtual bond dimensions.
        """
        if len(self.A) == 0:
            return []
        D = [self.A[i].shape[2] for i in range(len(self.A))]
        D.append(self.A[-1].shape[3])
        return D

    def zero_qnumbers(self):
        """
        Set all quantum numbers to zero (effectively disabling them).
        """
        self.qd.fill(0)
        for i in range(len(self.qD)):
            self.qD[i].fill(0)
        # enable chaining
        return self

    def orthonormalize(self, mode='left'):
        """
        Left- or right-orthonormalize the MPO (Frobenius norm) using QR decompositions.
        """
        if len(self.A) == 0:
            return 1
        if mode == 'left':
            for i in range(len(self.A) - 1):
                self.A[i], self.A[i+1], self.qD[i+1] = local_orthonormalize_left_qr(
                    self.A[i], self.A[i+1], self.qd, self.qD[i:i+2])
            # last tensor
            self.A[-1], T, self.qD[-1] = local_orthonormalize_left_qr(self.A[-1], np.array([[[[1]]]]), self.qd, self.qD[-2:])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1, 1)
            nrm = T[0, 0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.A[-1] = -self.A[-1]
                nrm = -nrm
            return nrm
        if mode == 'right':
            for i in reversed(range(1, len(self.A))):
                self.A[i], self.A[i-1], self.qD[i] = local_orthonormalize_right_qr(
                    self.A[i], self.A[i-1], self.qd, self.qD[i:i+2])
            # first tensor
            self.A[0], T, self.qD[0] = local_orthonormalize_right_qr(self.A[0], np.array([[[[1]]]]), self.qd, self.qD[:2])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1, 1)
            nrm = T[0, 0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.A[0] = -self.A[0]
                nrm = -nrm
            return nrm
        raise ValueError(f'mode = {mode} invalid; must be "left" or "right".')

    def as_matrix(self, sparse_format:bool=False):
        """
        Merge all tensors to obtain the matrix representation on the full Hilbert space.
        """
        if not sparse_format:
            op = self.A[0]
            for i in range(1, len(self.A)):
                op = merge_mpo_tensor_pair(op, self.A[i])
            assert op.ndim == 4
            assert op.shape[2] == 1 and op.shape[3] == 1
            op = op.reshape((op.shape[0], op.shape[1]))
            return op
        else:
            n = len(self.qd)
            op = self.A[0]
            assert op.shape[2] == 1
            # keep right virtual bond dimension as column dimension
            op = sparse.csr_array(op.reshape((-1, op.shape[3])))
            for i in range(1, len(self.A)):
                T = self.A[i]
                assert T.shape[0] == len(self.qd)
                op_next_list = []
                for j in range(len(self.qd)):
                    # explicitly index physical output axis;
                    # compressed sparse column format for subsequent multiplication
                    Tj = sparse.csc_array(T[j].transpose((1, 0, 2)).reshape(T.shape[2], -1))
                    # contract along virtual bond and isolate physical output axis of 'op'
                    op_next_list.append((op @ Tj).reshape((n, -1)))
                op = sparse.csr_array(sparse.hstack(op_next_list))
                n *= len(self.qd)
                op = op.reshape((n**2, -1))
            assert op.shape[1] == 1
            # restore physical input and output dimensions
            op = sparse.csr_array(op.reshape((n, n)))
            return op

    def __add__(self, other):
        """
        Add MPO to another.
        """
        return add_mpo(self, other)

    def __sub__(self, other):
        """
        Subtract another MPO.
        """
        return add_mpo(self, other, alpha=-1)

    def __matmul__(self, other):
        """
        Multiply MPO with another (composition along physical dimension).
        """
        return multiply_mpo(self, other)


def local_orthonormalize_left_qr(A: np.ndarray, Anext: np.ndarray, qd: Sequence[int], qD: Sequence[Sequence[int]]):
    """
    Left-orthonormalize local site tensor `A` by a QR decomposition,
    and update tensor at next site.
    """
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 4
    q0 = qnumber_flatten([qd, -qd, qD[0]])
    Q, R, qbond = qr(A.reshape((s[0]*s[1]*s[2], s[3])), q0, qD[1])
    A = Q.reshape((s[0], s[1], s[2], Q.shape[1]))
    # update Anext tensor: multiply with R from left
    Anext = np.tensordot(R, Anext, (1, 2)).transpose((1, 2, 0, 3))
    return (A, Anext, qbond)


def local_orthonormalize_right_qr(A: np.ndarray, Aprev: np.ndarray, qd: Sequence[int], qD: Sequence[Sequence[int]]):
    """
    Right-orthonormalize local site tensor `A` by a QR decomposition,
    and update tensor at previous site.
    """
    # flip left and right virtual bond dimensions
    A = A.transpose((0, 1, 3, 2))
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 4
    q0 = qnumber_flatten([qd, -qd, -qD[1]])
    Q, R, qbond = qr(A.reshape((s[0]*s[1]*s[2], s[3])), q0, -qD[0])
    A = Q.reshape((s[0], s[1], s[2], Q.shape[1])).transpose((0, 1, 3, 2))
    # update Aprev tensor: multiply with R from right
    Aprev = np.tensordot(Aprev, R, (3, 1))
    return (A, Aprev, -qbond)


def merge_mpo_tensor_pair(A0: np.ndarray, A1: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPO tensors.
    """
    A = np.einsum(A0, (0, 2, 4, 6), A1, (1, 3, 6, 5), (0, 1, 2, 3, 4, 5), optimize=True)
    # combine original physical dimensions
    s = A.shape
    A = A.reshape((s[0]*s[1], s[2]*s[3], s[4], s[5]))
    return A


def add_mpo(op0: MPO, op1: MPO, alpha=1) -> MPO:
    """
    Logical addition of two MPOs (effectively sum virtual bond dimensions)
    with the second MPO scaled by 'alpha'.
    """
    # number of lattice sites must agree
    assert op0.nsites == op1.nsites
    L = op0.nsites
    # physical quantum numbers must agree
    assert np.array_equal(op0.qd, op1.qd)
    d = len(op0.qd)

    # initialize with dummy tensors and bond quantum numbers
    op = MPO(op0.qd, (L+1)*[[0]])

    if L == 1:
        # single site
        # dummy bond quantum numbers must agree
        assert np.array_equal(op0.qD[0], op1.qD[0])
        assert np.array_equal(op0.qD[1], op1.qD[1])
        op.qD[0] = op0.qD[0].copy()
        op.qD[1] = op0.qD[1].copy()
        # simply add MPO tensors
        op.A[0] = op0.A[0] + alpha*op1.A[0]
        # consistency check
        assert is_qsparse(op.A[0], [op.qd, -op.qd, op.qD[0], -op.qD[1]]), \
            'sparsity pattern of MPO tensor does not match quantum numbers'
    elif L > 1:
        # combine virtual bond quantum numbers
        # leading and trailing (dummy) bond quantum numbers must agree
        assert np.array_equal(op0.qD[ 0], op1.qD[ 0])
        assert np.array_equal(op0.qD[-1], op1.qD[-1])
        op.qD[ 0] = op0.qD[ 0].copy()
        op.qD[-1] = op0.qD[-1].copy()
        # intermediate bond quantum numbers
        for i in range(1, L):
            op.qD[i] = np.concatenate((op0.qD[i], op1.qD[i]))

        # leftmost tensor
        op.A[0] = np.block([op0.A[0], alpha*op1.A[0]])
        # intermediate tensors
        for i in range(1, L - 1):
            s0 = op0.A[i].shape
            s1 = op1.A[i].shape
            # form block-diagonal tensor
            op.A[i] = np.block([[op0.A[i], np.zeros((d, d, s0[2], s1[3]))], [np.zeros((d, d, s1[2], s0[3])), op1.A[i]]])
        # rightmost tensor
        op.A[-1] = np.block([[op0.A[-1]], [op1.A[-1]]])

        # consistency check
        for i in range(1, L):
            assert is_qsparse(op.A[i], [op.qd, -op.qd, op.qD[i], -op.qD[i+1]]), \
                'sparsity pattern of MPO tensor does not match quantum numbers'
    return op


def multiply_mpo(op0: MPO, op1: MPO) -> MPO:
    """
    Multiply two MPOs (composition along physical dimension).
    """
    # number of lattice sites must agree
    assert op0.nsites == op1.nsites
    L = op0.nsites
    # physical quantum numbers must agree
    assert np.array_equal(op0.qd, op1.qd)

    # initialize with dummy tensors and bond quantum numbers
    op = MPO(op0.qd, (L+1)*[[0]])

    # combine virtual bond quantum numbers
    for i in range(L + 1):
        op.qD[i] = qnumber_flatten([op0.qD[i], op1.qD[i]])

    for i in range(L):
        # multiply physical dimensions and reorder dimensions
        op.A[i] = np.tensordot(op0.A[i], op1.A[i], (1, 0)).transpose((0, 3, 1, 4, 2, 5))
        # merge virtual bonds
        s = op.A[i].shape
        assert len(s) == 6
        op.A[i] = op.A[i].reshape((s[0], s[1], s[2]*s[3], s[4]*s[5]))
        # consistency check
        assert is_qsparse(op.A[i], [op.qd, -op.qd, op.qD[i], -op.qD[i+1]]), \
            'sparsity pattern of MPO tensor does not match quantum numbers'
    return op
