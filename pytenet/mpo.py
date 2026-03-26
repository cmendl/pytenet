"""
Matrix product operator (MPO) class and associated functionality.
"""

from collections.abc import Sequence, Mapping
import numpy as np
from scipy import sparse
from .block_sparse_util import qnumber_flatten, is_qsparse, enforce_qsparsity, block_sparse_qr
from .opgraph import OpGraph
from .util import crandn

__all__ = ["MPO", "mpo_merge_tensor_pair"]


class MPO:
    """
    Matrix product operator (MPO) class.

    The i-th MPO tensor has dimension `[d, d, b[i], b[i+1]]` with `d` the physical
    dimension at each site and `b` the list of virtual bond dimensions.

    Quantum numbers are assumed to be additive and stored as integers.
    `qsite` stores the list of physical quantum numbers at each site (assumed to agree
    for first and second physical dimension), and `qbonds` the virtual bond quantum numbers.
    The sum of first physical and left virtual bond quantum number of each
    non-zero tensor entry must be equal to the sum of second physical and
    right virtual bond quantum number.
    """

    def __init__(self, qsite, qbonds, fill=0.0, rng=None):
        """
        Create a matrix product operator.

        Args:
            qsite: physical quantum numbers at each site (same for all sites)
            qbonds: virtual bond quantum numbers (list of quantum number lists)
            fill: explicit scalar number to fill MPO tensors with, or
                  "random" to initialize tensors with random complex entries, or
                  "random real" to initialize tensors with random real entries, or
                  "postpone" to leave MPO tensors unallocated
            rng: (optional) random number generator for drawing entries
        """
        # require NumPy arrays
        self.qsite = np.array(qsite)
        self.qbonds = [np.array(qb) for qb in qbonds]
        # create list of MPS tensors
        d = len(qsite)
        b = [len(qb) for qb in qbonds]
        if isinstance(fill, (int, float, complex)):
            self.a = [np.full((d, d, b[i], b[i+1]), fill) for i in range(len(b)-1)]
        elif fill == "random":
            # random complex entries
            if rng is None:
                rng = np.random.default_rng()
            self.a = [crandn((d, d, b[i], b[i+1]), rng) / np.sqrt(d*b[i]*b[i+1])
                      for i in range(len(b) - 1)]
        elif fill == "random real":
            # random real entries
            if rng is None:
                rng = np.random.default_rng()
            self.a = [rng.normal(size=(d, d, b[i], b[i+1])) / np.sqrt(d*b[i]*b[i+1])
                      for i in range(len(b) - 1)]
        elif fill == "postpone":
            self.a = (len(b) - 1) * [None]
        else:
            raise ValueError(f'`fill` = {fill} invalid; must be a number, '
                             f'"random", "random real" or "postpone".')
        if fill != "postpone":
            # enforce block sparsity structure dictated by quantum numbers
            for i, ai in enumerate(self.a):
                enforce_qsparsity(ai, [self.qsite, -self.qsite, self.qbonds[i], -self.qbonds[i+1]])
        # can be set by `from_opgraph`
        self.nid_map = None

    @classmethod
    def identity(cls, qsite: Sequence[int], nsites: int, scale: float = 1, dtype=float):
        """
        Construct MPO representation of the identity operation.
        """
        d = len(qsite)
        mpo = cls(qsite, (nsites + 1)*[[0]])
        for i in range(nsites):
            mpo.a[i] = scale * np.identity(d, dtype=dtype).reshape((d, d, 1, 1))
        return mpo

    @classmethod
    def from_opgraph(cls, qsite: Sequence[int], graph: OpGraph, opmap: Mapping,
                     compute_nid_map=False):
        """
        Construct a MPO from an operator graph.

        Args:
            qsite: physical quantum numbers at each site (same for all sites)
            graph: symbolic operator graph
            opmap: local operators as dictionary, using operator IDs as keys
            compute_nid_map: whether to construct the map from node IDs to bond location and index

        Returns:
            MPO: MPO representation of the operator graph
        """
        d = len(qsite)
        if d == 0:
            raise ValueError("require at least one physical quantum number")
        a_list = []
        qbonds = []
        # node IDs at current bond site
        nids0 = [graph.nid_terminal[0]]
        qbonds.append([graph.nodes[graph.nid_terminal[0]].qnum])
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
            qbonds.append([graph.nodes[nid].qnum for nid in nids1])
            if compute_nid_map:
                for i, nid in enumerate(nids1):
                    # record bond information (site location and index)
                    nid_map[nid] = (l, i)
            a = np.zeros((d, d, len(nids0), len(nids1)))
            for i, nid in enumerate(nids0):
                node = graph.nodes[nid]
                for eid in node.eids[1]:
                    edge = graph.edges[eid]
                    j = nids1.index(edge.nids[1])
                    # update local operator in MPO tensor
                    # (supporting multiple edges between same pair of nodes)
                    daij = sum(c * opmap[k] for k, c in edge.opics)
                    if np.iscomplexobj(daij):
                        a = a.astype(complex)
                    a[:, :, i, j] += daij
            a_list.append(a)
            nids0 = nids1
            if compute_nid_map:
                # next bond
                l += 1
        assert len(a_list) + 1 == len(qbonds)
        op = cls(qsite, qbonds, fill="postpone")
        op.a = a_list
        # consistency check
        for i in range(op.nsites):
            assert is_qsparse(op.a[i], [op.qsite, -op.qsite, op.qbonds[i], -op.qbonds[i+1]]), \
                "sparsity pattern of MPO tensor does not match quantum numbers"
        # store node ID map in MPO
        if compute_nid_map:
            op.nid_map = nid_map
        return op

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return len(self.a)

    @property
    def bond_dims(self) -> list:
        """
        Virtual bond dimensions.
        """
        if len(self.a) == 0:
            return []
        return [self.a[i].shape[2] for i in range(len(self.a))] + [self.a[-1].shape[3]]

    def zero_qnumbers(self):
        """
        Set all quantum numbers to zero (effectively disabling them).
        """
        self.qsite.fill(0)
        for qb in self.qbonds:
            qb.fill(0)
        # enable chaining
        return self

    def orthonormalize(self, mode="left"):
        """
        Left- or right-orthonormalize the MPO (Frobenius norm) using QR decompositions.
        """
        if len(self.a) == 0:
            return 1
        if mode == "left":
            for i in range(len(self.a) - 1):
                self.a[i], self.a[i+1], self.qbonds[i+1] = mpo_local_orthonormalize_left_qr(
                    self.a[i], self.a[i+1], self.qsite, self.qbonds[i:i+2])
            # last tensor
            self.a[-1], t, self.qbonds[-1] = mpo_local_orthonormalize_left_qr(
                self.a[-1], np.array([[[[1]]]]), self.qsite, self.qbonds[-2:])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert t.shape == (1, 1, 1, 1)
            nrm = t[0, 0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.a[-1] = -self.a[-1]
                nrm = -nrm
            return nrm
        if mode == "right":
            for i in reversed(range(1, len(self.a))):
                self.a[i], self.a[i-1], self.qbonds[i] = mpo_local_orthonormalize_right_qr(
                    self.a[i], self.a[i-1], self.qsite, self.qbonds[i:i+2])
            # first tensor
            self.a[0], t, self.qbonds[0] = mpo_local_orthonormalize_right_qr(
                self.a[0], np.array([[[[1]]]]), self.qsite, self.qbonds[:2])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert t.shape == (1, 1, 1, 1)
            nrm = t[0, 0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.a[0] = -self.a[0]
                nrm = -nrm
            return nrm
        raise ValueError(f'`mode` = {mode} invalid; must be "left" or "right".')

    def to_matrix(self, sparse_format: bool = False):
        """
        Merge all tensors to obtain the matrix representation on the full Hilbert space.
        """
        if not sparse_format:
            op = self.a[0]
            for i in range(1, len(self.a)):
                op = mpo_merge_tensor_pair(op, self.a[i])
            assert op.ndim == 4
            assert op.shape[2] == 1 and op.shape[3] == 1
            op = op.reshape((op.shape[0], op.shape[1]))
            return op
        else:
            n = len(self.qsite)
            op = self.a[0]
            assert op.shape[2] == 1
            # keep right virtual bond dimension as column dimension
            op = sparse.csr_array(op.reshape((-1, op.shape[3])))
            for i in range(1, len(self.a)):
                t = self.a[i]
                assert t.shape[0] == len(self.qsite)
                op_next_list = []
                for j in range(len(self.qsite)):
                    # explicitly index physical output axis;
                    # compressed sparse column format for subsequent multiplication
                    tj = sparse.csc_array(t[j].transpose((1, 0, 2)).reshape(t.shape[2], -1))
                    # contract along virtual bond and isolate physical output axis of "op"
                    op_next_list.append((op @ tj).reshape((n, -1)))
                op = sparse.csr_array(sparse.hstack(op_next_list))
                n *= len(self.qsite)
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


def mpo_local_orthonormalize_left_qr(a: np.ndarray, a_next: np.ndarray,
                                 qsite: Sequence[int], qbonds: Sequence[Sequence[int]]):
    """
    Left-orthonormalize local site tensor `A` by a QR decomposition,
    and update tensor at next site.
    """
    # perform QR decomposition and replace A by reshaped Q matrix
    s = a.shape
    assert len(s) == 4
    q0 = qnumber_flatten([qsite, -qsite, qbonds[0]])
    q, r, qbond = block_sparse_qr(a.reshape((s[0]*s[1]*s[2], s[3])), q0, qbonds[1])
    a = q.reshape((s[0], s[1], s[2], q.shape[1]))
    # update a_next tensor: multiply with R from left
    a_next = np.tensordot(r, a_next, (1, 2)).transpose((1, 2, 0, 3))
    return (a, a_next, qbond)


def mpo_local_orthonormalize_right_qr(a: np.ndarray, a_prev: np.ndarray,
                                      qsite: Sequence[int], qbonds: Sequence[Sequence[int]]):
    """
    Right-orthonormalize local site tensor `a` by a QR decomposition,
    and update tensor at previous site.
    """
    # flip left and right virtual bond dimensions
    a = a.transpose((0, 1, 3, 2))
    # perform QR decomposition and replace A by reshaped Q matrix
    s = a.shape
    assert len(s) == 4
    q0 = qnumber_flatten([qsite, -qsite, -qbonds[1]])
    q, r, qbond = block_sparse_qr(a.reshape((s[0]*s[1]*s[2], s[3])), q0, -qbonds[0])
    a = q.reshape((s[0], s[1], s[2], q.shape[1])).transpose((0, 1, 3, 2))
    # update a_prev tensor: multiply with R from right
    a_prev = np.tensordot(a_prev, r, (3, 1))
    return (a, a_prev, -qbond)


def mpo_merge_tensor_pair(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPO tensors.
    """
    a = np.einsum(a0, (0, 2, 4, 6), a1, (1, 3, 6, 5), (0, 1, 2, 3, 4, 5), optimize=True)
    # combine original physical dimensions
    s = a.shape
    a = a.reshape((s[0]*s[1], s[2]*s[3], s[4], s[5]))
    return a


def add_mpo(op0: MPO, op1: MPO, alpha=1) -> MPO:
    """
    Logical addition of two MPOs (effectively sum virtual bond dimensions)
    with the second MPO scaled by "alpha".
    """
    # number of lattice sites must agree
    assert op0.nsites == op1.nsites
    nsites = op0.nsites
    # physical quantum numbers must agree
    assert np.array_equal(op0.qsite, op1.qsite)
    d = len(op0.qsite)

    # initialize with dummy tensors and bond quantum numbers
    op = MPO(op0.qsite, (nsites + 1)*[[0]])

    if nsites == 1:
        # single site
        # dummy bond quantum numbers must agree
        assert np.array_equal(op0.qbonds[0], op1.qbonds[0])
        assert np.array_equal(op0.qbonds[1], op1.qbonds[1])
        op.qbonds[0] = op0.qbonds[0].copy()
        op.qbonds[1] = op0.qbonds[1].copy()
        # simply add MPO tensors
        op.a[0] = op0.a[0] + alpha*op1.a[0]
        # consistency check
        assert is_qsparse(op.a[0], [op.qsite, -op.qsite, op.qbonds[0], -op.qbonds[1]]), \
            "sparsity pattern of MPO tensor does not match quantum numbers"
    elif nsites > 1:
        # combine virtual bond quantum numbers
        # leading and trailing (dummy) bond quantum numbers must agree
        assert np.array_equal(op0.qbonds[ 0], op1.qbonds[ 0])
        assert np.array_equal(op0.qbonds[-1], op1.qbonds[-1])
        op.qbonds[ 0] = op0.qbonds[ 0].copy()
        op.qbonds[-1] = op0.qbonds[-1].copy()
        # intermediate bond quantum numbers
        for i in range(1, nsites):
            op.qbonds[i] = np.concatenate((op0.qbonds[i], op1.qbonds[i]))

        # leftmost tensor
        op.a[0] = np.block([op0.a[0], alpha*op1.a[0]])
        # intermediate tensors
        for i in range(1, nsites - 1):
            s0 = op0.a[i].shape
            s1 = op1.a[i].shape
            # form block-diagonal tensor
            op.a[i] = np.block([[op0.a[i], np.zeros((d, d, s0[2], s1[3]))],
                                [np.zeros((d, d, s1[2], s0[3])), op1.a[i]]])
        # rightmost tensor
        op.a[-1] = np.block([[op0.a[-1]], [op1.a[-1]]])

        # consistency check
        for i in range(1, nsites):
            assert is_qsparse(op.a[i], [op.qsite, -op.qsite, op.qbonds[i], -op.qbonds[i+1]]), \
                "sparsity pattern of MPO tensor does not match quantum numbers"
    return op


def multiply_mpo(op0: MPO, op1: MPO) -> MPO:
    """
    Multiply two MPOs (composition along physical dimension).
    """
    # number of lattice sites must agree
    assert op0.nsites == op1.nsites
    nsites = op0.nsites
    # physical quantum numbers must agree
    assert np.array_equal(op0.qsite, op1.qsite)

    # initialize with dummy tensors and bond quantum numbers
    op = MPO(op0.qsite, (nsites + 1)*[[0]])

    # combine virtual bond quantum numbers
    for i in range(nsites + 1):
        op.qbonds[i] = qnumber_flatten([op0.qbonds[i], op1.qbonds[i]])

    for i in range(nsites):
        # multiply physical dimensions and reorder dimensions
        op.a[i] = np.tensordot(op0.a[i], op1.a[i], (1, 0)).transpose((0, 3, 1, 4, 2, 5))
        # merge virtual bonds
        s = op.a[i].shape
        assert len(s) == 6
        op.a[i] = op.a[i].reshape((s[0], s[1], s[2]*s[3], s[4]*s[5]))
        # consistency check
        assert is_qsparse(op.a[i], [op.qsite, -op.qsite, op.qbonds[i], -op.qbonds[i+1]]), \
            "sparsity pattern of MPO tensor does not match quantum numbers"
    return op
