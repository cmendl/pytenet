"""
Matrix product state (MPS) class and associated functionality.
"""

from collections.abc import Sequence
import numpy as np
from .block_sparse_util import (qnumber_outer_sum, qnumber_flatten,
    is_qsparse, enforce_qsparsity, block_sparse_qr, block_sparse_eigh)
from .bond_ops import retained_bond_indices, split_block_sparse_matrix_svd
from .util import crandn

__all__ = ["MPS", "mps_merge_tensor_pair", "mps_split_tensor_svd"]


class MPS:
    """
    Matrix product state (MPS) class.

    The i-th MPS tensor has dimension `[d, b[i], b[i+1]]` with `d` the physical
    dimension at each site and `b` the list of virtual bond dimensions.

    Quantum numbers are assumed to be additive and stored as integers.
    `qsite` stores the list of physical quantum numbers at each site,
    and `qbonds` the virtual bond quantum numbers.
    The sum of physical and left virtual bond quantum number of each non-zero
    tensor entry must be equal to the right virtual bond quantum number.
    """

    def __init__(self, qsite, qbonds, fill=0.0, rng=None):
        """
        Create a matrix product state.

        Args:
            qsite: physical quantum numbers at each site (same for all sites)
            qbonds: virtual bond quantum numbers (list of quantum number lists)
            fill: explicit scalar number to fill MPS tensors with, or
                  "random" to initialize tensors with random complex entries, or
                  "random real" to initialize tensors with random real entries, or
                  "postpone" to leave MPS tensors unallocated
            rng: (optional) random number generator for drawing entries
        """
        # require NumPy arrays
        self.qsite = np.asarray(qsite)
        self.qbonds = [np.asarray(qDi) for qDi in qbonds]
        # create list of MPS tensors
        d = len(qsite)
        b = [len(qb) for qb in qbonds]
        # leading and trailing bond dimensions must be 1
        assert b[0] == 1 and b[-1] == 1
        if isinstance(fill, (int, float, complex)):
            self.a = [np.full((d, b[i], b[i+1]), fill) for i in range(len(b) - 1)]
        elif fill == "random":
            # random complex entries
            if rng is None:
                rng = np.random.default_rng()
            self.a = [crandn((d, b[i], b[i+1]), rng) / np.sqrt(d*b[i]*b[i+1])
                      for i in range(len(b) - 1)]
        elif fill == "random real":
            # random real entries
            if rng is None:
                rng = np.random.default_rng()
            self.a = [rng.normal(size=(d, b[i], b[i+1])) / np.sqrt(d*b[i]*b[i+1])
                      for i in range(len(b) - 1)]
        elif fill == "postpone":
            self.a = (len(b) - 1) * [None]
        else:
            raise ValueError(f'`fill` = {fill} invalid; must be a number, '
                             f'"random", "random real" or "postpone".')
        if fill != "postpone":
            # enforce block sparsity structure dictated by quantum numbers
            for i, ai in enumerate(self.a):
                enforce_qsparsity(ai, [self.qsite, self.qbonds[i], -self.qbonds[i+1]])

    @classmethod
    def construct_random(cls, nsites: int, qsite: Sequence[int], qnum_sector: int,
                         max_vdim: int=256, dtype="complex", rng: np.random.Generator=None):
        """
        Construct a matrix product state with random normal tensor entries,
        given an overall quantum number sector and maximum virtual bond dimension.
        """
        assert nsites > 0
        # require NumPy array
        qsite = np.asarray(qsite)
        if rng is None:
            rng = np.random.default_rng()
        qbonds = (nsites + 1) * [None]
        # dummy left virtual bond; set quantum number to zero
        qbonds[0] = [0]
        # dummy right virtual bond; set quantum number to overall quantum number sector
        qbonds[-1] = [qnum_sector]
        # virtual bond quantum numbers on left half
        for l in range(1, (nsites + 1) // 2):
            # enumerate all combinations of left bond quantum numbers
            # and local physical quantum numbers
            qnums_full = qnumber_outer_sum([qbonds[l - 1], qsite]).reshape(-1)
            if len(qnums_full) <= max_vdim:
                qbonds[l] = qnums_full
            else:
                # randomly select quantum numbers
                idx = rng.choice(len(qnums_full), size=max_vdim, replace=False)
                qbonds[l] = qnums_full[idx]
        # virtual bond quantum numbers on right half
        for l in reversed(range((nsites + 1) // 2, nsites)):
            # enumerate all combinations of right bond quantum numbers
            # and local physical quantum numbers
            qnums_full = qnumber_outer_sum([qbonds[l + 1], -qsite]).reshape(-1)
            if len(qnums_full) <= max_vdim:
                qbonds[l] = qnums_full
            else:
                # randomly select quantum numbers
                idx = rng.choice(len(qnums_full), size=max_vdim, replace=False)
                qbonds[l] = qnums_full[idx]
        return cls(qsite, qbonds, fill=("random" if (dtype in (complex, "complex")) \
                                        else "random real"), rng=rng)

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
        return [self.a[i].shape[1] for i in range(len(self.a))] + [self.a[-1].shape[2]]

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
        Left- or right-orthonormalize the MPS using QR decompositions.
        """
        if len(self.a) == 0:
            return 1

        if mode == "left":
            for i in range(len(self.a) - 1):
                self.a[i], self.a[i+1], self.qbonds[i+1] = mps_local_orthonormalize_left_qr(
                    self.a[i], self.a[i+1], self.qsite, self.qbonds[i:i+2])
            # last tensor
            self.a[-1], t, self.qbonds[-1] = mps_local_orthonormalize_left_qr(
                self.a[-1], np.array([[[1]]]), self.qsite, self.qbonds[-2:])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert t.shape == (1, 1, 1)
            nrm = t[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.a[-1] = -self.a[-1]
                nrm = -nrm
            return nrm
        if mode == "right":
            for i in reversed(range(1, len(self.a))):
                self.a[i], self.a[i-1], self.qbonds[i] = mps_local_orthonormalize_right_qr(
                    self.a[i], self.a[i-1], self.qsite, self.qbonds[i:i+2])
            # first tensor
            self.a[0], t, self.qbonds[0] = mps_local_orthonormalize_right_qr(
                self.a[0], np.array([[[1]]]), self.qsite, self.qbonds[:2])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert t.shape == (1, 1, 1)
            nrm = t[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.a[0] = -self.a[0]
                nrm = -nrm
            return nrm
        raise ValueError(f'`mode` = {mode} invalid; must be "left" or "right".')

    def compress(self, tol: float, mode="svd", direction="left"):
        """
        Compress and orthonormalize an MPS.

        For "mode == "svd"", the function performs site-local SVDs and singular value truncations.
        For "mode == "density"", the function implements the rounding compression
        based on the local density matrix.

        Returns original norm and scaling factor due to compression.
        """
        if mode == "svd":
            return self._compress_svd(tol, direction)
        if mode == "density":
            return self._compress_density(tol)
        raise ValueError(f'`mode` = {mode} invalid; must be "svd" or "density".')

    def _compress_svd(self, tol: float, direction: str):
        """
        Compress and orthonormalize an MPS by site-local SVDs and singular value truncations.

        Returns original norm and scaling factor due to compression.
        """
        if direction == "left":
            # transform to right-canonical form first
            nrm = self.orthonormalize(mode="right")
            for i in range(len(self.a) - 1):
                self.a[i], self.a[i+1], self.qbonds[i+1] = mps_local_orthonormalize_left_svd(
                    self.a[i], self.a[i+1], self.qsite, self.qbonds[i:i+2], tol)
                assert is_qsparse(self.a[i], [self.qsite, self.qbonds[i], -self.qbonds[i+1]]), \
                    "sparsity pattern of MPS tensor does not match quantum numbers"
            # last tensor
            self.a[-1], t, self.qbonds[-1] = mps_local_orthonormalize_left_svd(
                self.a[-1], np.array([[[1]]]), self.qsite, self.qbonds[-2:], tol)
            assert is_qsparse(self.a[-1], [self.qsite, self.qbonds[-2], -self.qbonds[-1]]), \
                "sparsity pattern of MPS tensor does not match quantum numbers"
            assert t.shape == (1, 1, 1)
            # absorb potential phase factor into MPS tensor
            self.a[-1] *= t[0, 0, 0] / abs(t[0, 0, 0])
            return (nrm, abs(t[0, 0, 0]))
        if direction == "right":
            # transform to left-canonical form first
            nrm = self.orthonormalize(mode="left")
            for i in reversed(range(1, len(self.a))):
                self.a[i], self.a[i-1], self.qbonds[i] = mps_local_orthonormalize_right_svd(
                    self.a[i], self.a[i-1], self.qsite, self.qbonds[i:i+2], tol)
                assert is_qsparse(self.a[i], [self.qsite, self.qbonds[i], -self.qbonds[i+1]]), \
                    "sparsity pattern of MPS tensor does not match quantum numbers"
            # first tensor
            self.a[0], t, self.qbonds[0] = mps_local_orthonormalize_right_svd(
                self.a[0], np.array([[[1]]]), self.qsite, self.qbonds[:2], tol)
            assert is_qsparse(self.a[0], [self.qsite, self.qbonds[0], -self.qbonds[1]]), \
                "sparsity pattern of MPS tensor does not match quantum numbers"
            assert t.shape == (1, 1, 1)
            # absorb potential phase factor into MPS tensor
            self.a[0] *= t[0, 0, 0] / abs(t[0, 0, 0])
            return (nrm, abs(t[0, 0, 0]))
        raise ValueError(f'`direction` = {direction} invalid; must be "left" or "right".')

    def _compress_density(self, tol: float):
        """
        Compress and orthonormalize an MPS, implementing the rounding compression
        discussed in the paper
        "From density-matrix renormalization group to matrix product states" (cited below),
        see also https://tensornetwork.org/mps/#toc_6

        Returns original norm and scaling factor due to compression.

        Reference:
            Ian P. McCulloch
            From density-matrix renormalization group to matrix product states
            J. Stat. Mech. (2007) P10014
            DOI: 10.1088/1742-5468/2007/10/P10014
            https://arxiv.org/abs/cond-mat/0701428
        """
        from .operation import compute_left_state_blocks
        left_envs = compute_left_state_blocks(self, self)
        assert left_envs[-1].shape == (1, 1)
        assert left_envs[-1][0, 0].real > 0
        nrm = np.sqrt(left_envs[-1][0, 0].real)
        # trivial initial tensors
        b = np.array([1], dtype=self.a[-1].dtype).reshape(1, 1, 1)
        # current `u` matrix
        u = np.array([1], dtype=self.a[-1].dtype).reshape(1, 1, 1)
        for i in reversed(range(1, self.nsites)):
            # compute new `b` block
            b = np.tensordot(b, u.conj(), axes=([0, 2], [0, 2]))
            b = np.tensordot(self.a[i], b, axes=([2], [0]))
            # compute density matrix
            rho = np.tensordot(b, left_envs[i], axes=([1], [0]))
            rho = np.tensordot(rho, b.conj(), axes=([2], [1]))
            # diagonalize density matrix
            assert rho.ndim == 4
            orig_shape = rho.shape[0:2]
            rho = rho.reshape((np.prod(rho.shape[0:2]), np.prod(rho.shape[2:4])))
            qnums_rho = qnumber_flatten([self.qsite, -self.qbonds[i+1]])
            assert is_qsparse(rho, [qnums_rho, -qnums_rho])
            u, evals, qnums_eig = block_sparse_eigh(rho, qnums_rho)
            # truncate small eigenvalues;
            # eigenvalues are real, but can be negative
            idx = retained_bond_indices(np.abs(evals), tol)
            u = u[:, idx]
            evals = evals[idx]
            qnums_eig = -qnums_eig[idx]
            assert is_qsparse(u, [qnums_rho, qnums_eig])
            u = u.reshape(orig_shape[0], orig_shape[1], u.shape[1]).transpose((0, 2, 1))
            assert is_qsparse(u, [self.qsite, qnums_eig, -self.qbonds[i+1]])
            self.a[i] = u
            self.qbonds[i] = qnums_eig
        # for the leftmost site, we merely need to find the block, as the bond is already truncated
        b = np.tensordot(b, u.conj(), axes=([0, 2], [0, 2]))
        b = np.tensordot(self.a[0], b, axes=([2], [0]))
        s = np.linalg.norm(b.reshape(-1))
        self.a[0] = b / s
        return (nrm, s / nrm)

    def to_vector(self) -> np.ndarray:
        """
        Merge all tensors to obtain the vector representation on the full Hilbert space.
        """
        psi = self.a[0]
        for i in range(1, len(self.a)):
            psi = mps_merge_tensor_pair(psi, self.a[i])
        assert psi.ndim == 3
        assert psi.shape[1] == 1 and psi.shape[2] == 1
        return psi.reshape(-1)

    @classmethod
    def from_vector(cls, d: int, nsites: int, v: np.ndarray, tol: float = 0):
        """
        Construct the MPS representation of the vector `v` via the TT-SVD algorithm,
        for local dimension `d`.

        All quantum numbers are set to zero for simplicity.
        """
        v = np.array(v)
        assert v.ndim == 1 and len(v) == d**nsites, \
            f"`v` has length {len(v)}, expecting {d**nsites}."
        # allocate an MPS with dummy virtual bonds of dimension 1
        mps = cls(d*[0], [[0] for _ in range(nsites + 1)], fill="postpone")
        # endow `v` with a dummy left virtual bond dimension
        v = np.reshape(v, (1, len(v)))
        for i in range(nsites):
            assert v.shape[1] == d**(nsites - i)
            bleft = v.shape[0]
            # partition physical dimension into local dimension and the rest,
            # split by SVD and assign the right SVD matrix to `v`
            u, s, v = np.linalg.svd(v.reshape((bleft*d, d**(nsites-i-1))), full_matrices=False)
            # truncate small singular values
            idx = retained_bond_indices(s, tol)
            u = u[:, idx]
            v = v[idx, :]
            s = s[idx]
            # use broadcasting to distribute singular values to the right
            v = v * s[:, None]
            # assign MPS tensor at current site i, transposing physical dimension to the front
            mps.a[i] = u.reshape((bleft, d, len(s))).transpose((1, 0, 2))
            mps.qbonds[i + 1] = len(s) * [0]
        assert v.shape == (1, 1)
        # include scalar factor in last MPS tensor
        mps.a[-1] *= v[0, 0]
        return mps

    def __add__(self, other):
        """
        Add MPS to another.
        """
        return add_mps(self, other)

    def __sub__(self, other):
        """
        Subtract another MPS.
        """
        return add_mps(self, other, alpha=-1)


def mps_local_orthonormalize_left_qr(a: np.ndarray, a_next: np.ndarray,
                                     qsite: Sequence[int], qbonds: Sequence[Sequence[int]]):
    """
    Left-orthonormalize the local site tensor `a` by a QR decomposition,
    and update the tensor at the next site.
    """
    # perform QR decomposition and replace `a` by reshaped Q matrix
    s = a.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qsite, qbonds[0]])
    q, r, qbond = block_sparse_qr(a.reshape((s[0]*s[1], s[2])), q0, qbonds[1])
    a = q.reshape((s[0], s[1], q.shape[1]))
    # update a_next tensor: multiply with R from left
    a_next = np.tensordot(r, a_next, (1, 1)).transpose((1, 0, 2))
    return (a, a_next, qbond)


def mps_local_orthonormalize_right_qr(a: np.ndarray, a_prev: np.ndarray,
                                      qsite: Sequence[int], qbonds: Sequence[Sequence[int]]):
    """
    Right-orthonormalize the local site tensor `a` by a QR decomposition,
    and update the tensor at the previous site.
    """
    # flip left and right virtual bond dimensions
    a = a.transpose((0, 2, 1))
    # perform QR decomposition and replace A by reshaped Q matrix
    s = a.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qsite, -qbonds[1]])
    q, r, qbond = block_sparse_qr(a.reshape((s[0]*s[1], s[2])), q0, -qbonds[0])
    a = q.reshape((s[0], s[1], q.shape[1])).transpose((0, 2, 1))
    # update a_prev tensor: multiply with R from right
    a_prev = np.tensordot(a_prev, r, (2, 1))
    return (a, a_prev, -qbond)


def mps_local_orthonormalize_left_svd(a, a_next, qsite: Sequence[int],
                                      qbonds: Sequence[Sequence[int]], tol: float):
    """
    Left-orthonormalize the local site tensor `a` by a SVD,
    and update the tensor at the next site.
    """
    # perform SVD and replace `a` by reshaped U matrix
    s = a.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qsite, qbonds[0]])
    u, sigma, v, qbond = split_block_sparse_matrix_svd(
        a.reshape((s[0]*s[1], s[2])), q0, qbonds[1], tol)
    a = u.reshape((s[0], s[1], u.shape[1]))
    # update a_next tensor: multiply with (sigma @ v) from left
    a_next = np.tensordot(sigma[:, None] * v, a_next, (1, 1)).transpose((1, 0, 2))
    return (a, a_next, qbond)


def mps_local_orthonormalize_right_svd(a, a_prev, qsite: Sequence[int],
                                       qbonds: Sequence[Sequence[int]], tol: float):
    """
    Right-orthonormalize the local site tensor `a` by a SVD,
    and update the tensor at the previous site.
    """
    # flip physical and left virtual bond dimension
    a = a.transpose((1, 0, 2))
    # perform SVD and replace a by reshaped V matrix
    s = a.shape
    assert len(s) == 3
    q1 = qnumber_flatten([-qsite, qbonds[1]])
    u, sigma, v, qbond = split_block_sparse_matrix_svd(
        a.reshape((s[0], s[1]*s[2])), qbonds[0], q1, tol)
    a = v.reshape((v.shape[0], s[1], s[2]))
    # undo flip of physical and left virtual bond dimension
    a = a.transpose((1, 0, 2))
    # update a_prev tensor: multiply with (u @ sigma) from right
    a_prev = np.tensordot(a_prev, u * sigma, (2, 0))
    return (a, a_prev, qbond)


def mps_merge_tensor_pair(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPS tensors.
    """
    a = np.einsum(a0, (0, 2, 3), a1, (1, 3, 4), (0, 1, 2, 4), optimize=True)
    # combine original physical dimensions
    a = a.reshape((a.shape[0]*a.shape[1], a.shape[2], a.shape[3]))
    return a


def mps_split_tensor_svd(a: np.ndarray, qsite0: Sequence[int], qsite1: Sequence[int],
                         qbonds: Sequence[Sequence[int]], svd_distr: str, tol=0):
    """
    Split an MPS tensor with dimension `d0*d1 x b0 x b2` into two MPS tensors
    with dimensions `d0 x b0 x b1` and `d1 x b1 x b2`, respectively.
    """
    assert a.ndim == 3
    d0 = len(qsite0)
    d1 = len(qsite1)
    assert d0 * d1 == a.shape[0], "physical dimension of MPS tensor must be equal to d0 * d1"
    # reshape as matrix and split by SVD
    a = a.reshape((d0, d1, a.shape[1], a.shape[2])).transpose((0, 2, 1, 3))
    s = a.shape
    q0 = qnumber_flatten([ qsite0, qbonds[0]])
    q1 = qnumber_flatten([-qsite1, qbonds[1]])
    a0, sigma, a1, qbond = split_block_sparse_matrix_svd(
        a.reshape((s[0]*s[1], s[2]*s[3])), q0, q1, tol)
    a0.shape = (s[0], s[1], len(sigma))
    a1.shape = (len(sigma), s[2], s[3])
    # use broadcasting to distribute singular values
    if svd_distr == "left":
        a0 = a0 * sigma
    elif svd_distr == "right":
        a1 = a1 * sigma[:, None, None]
    elif svd_distr == "sqrt":
        s = np.sqrt(sigma)
        a0 = a0 * s
        a1 = a1 * s[:, None, None]
    else:
        raise ValueError('`svd_distr` parameter must be "left", "right" or "sqrt".')
    # move physical dimension to the front
    a1 = a1.transpose((1, 0, 2))
    return (a0, a1, qbond)


def add_mps(mps0: MPS, mps1: MPS, alpha=1) -> MPS:
    """
    Logical addition of two matrix product states (effectively sum virtual bond dimensions),
    with the second MPS scaled by "alpha".
    """
    # number of lattice sites must agree
    assert mps0.nsites == mps1.nsites
    nsites = mps0.nsites
    # physical quantum numbers must agree
    assert np.array_equal(mps0.qsite, mps1.qsite)
    d = len(mps0.qsite)

    # initialize with dummy tensors and bond quantum numbers
    mps = MPS(mps0.qsite, (nsites+1)*[[0]])

    if nsites == 1:
        # single site
        # dummy bond quantum numbers must agree
        assert np.array_equal(mps0.qbonds[0], mps1.qbonds[0])
        assert np.array_equal(mps0.qbonds[1], mps1.qbonds[1])
        mps.qbonds[0] = mps0.qbonds[0].copy()
        mps.qbonds[1] = mps0.qbonds[1].copy()
        # simply add MPS tensors
        mps.a[0] = mps0.a[0] + alpha*mps1.a[0]
        # consistency check
        assert is_qsparse(mps.a[0], [mps.qsite, mps.qbonds[0], -mps.qbonds[1]]), \
            "sparsity pattern of MPS tensor does not match quantum numbers"
    elif nsites > 1:
        # combine virtual bond quantum numbers
        # leading and trailing (dummy) bond quantum numbers must agree
        assert np.array_equal(mps0.qbonds[ 0], mps1.qbonds[ 0])
        assert np.array_equal(mps0.qbonds[-1], mps1.qbonds[-1])
        mps.qbonds[ 0] = mps0.qbonds[ 0].copy()
        mps.qbonds[-1] = mps0.qbonds[-1].copy()
        # intermediate bond quantum numbers
        for i in range(1, nsites):
            mps.qbonds[i] = np.concatenate((mps0.qbonds[i], mps1.qbonds[i]))

        # leftmost tensor
        mps.a[0] = np.block([mps0.a[0], alpha*mps1.a[0]])
        # intermediate tensors
        for i in range(1, nsites - 1):
            s0 = mps0.a[i].shape
            s1 = mps1.a[i].shape
            # form block-diagonal tensor
            mps.a[i] = np.block([[mps0.a[i], np.zeros((d, s0[1], s1[2]))],
                                 [np.zeros((d, s1[1], s0[2])), mps1.a[i]]])
        # rightmost tensor
        mps.a[-1] = np.block([[mps0.a[-1]], [mps1.a[-1]]])

        # consistency check
        for i in range(1, nsites):
            assert is_qsparse(mps.a[i], [mps.qsite, mps.qbonds[i], -mps.qbonds[i+1]]), \
                "sparsity pattern of MPS tensor does not match quantum numbers"
    return mps
