from collections.abc import Sequence
import numpy as np
from .qnumber import qnumber_outer_sum, qnumber_flatten, is_qsparse, enforce_qsparsity
from .bond_ops import qr, eigh, retained_bond_indices, split_matrix_svd
from .util import crandn

__all__ = ['MPS', 'merge_mps_tensor_pair', 'split_mps_tensor']


class MPS:
    """
    Matrix product state (MPS) class.

    The i-th MPS tensor has dimension `[d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.

    Quantum numbers are assumed to be additive and stored as integers.
    `qd` stores the list of physical quantum numbers at each site,
    and `qD` the virtual bond quantum numbers.
    The sum of physical and left virtual bond quantum number of each non-zero
    tensor entry must be equal to the right virtual bond quantum number.
    """

    def __init__(self, qd: Sequence[int], qD: Sequence[Sequence[int]], fill=0.0, rng: np.random.Generator=None):
        """
        Create a matrix product state.

        Args:
            qd: physical quantum numbers at each site (same for all sites)
            qD: virtual bond quantum numbers (list of quantum number lists)
            fill: explicit scalar number to fill MPS tensors with, or
                  'random' to initialize tensors with random complex entries, or
                  'postpone' to leave MPS tensors unallocated
            rng: (optional) random number generator for drawing entries
        """
        # require NumPy arrays
        self.qd = np.asarray(qd)
        self.qD = [np.asarray(qDi) for qDi in qD]
        # create list of MPS tensors
        d = len(qd)
        D = [len(qb) for qb in qD]
        # leading and trailing bond dimensions must be 1
        assert D[0] == 1 and D[-1] == 1
        if isinstance(fill, (int, float, complex)):
            self.A = [np.full((d, D[i], D[i+1]), fill) for i in range(len(D)-1)]
        elif fill == 'random':
            # random complex entries
            if rng is None:
                rng = np.random.default_rng()
            self.A = [crandn((d, D[i], D[i+1]), rng) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)]
        elif fill == 'random real':
            # random real entries
            if rng is None:
                rng = np.random.default_rng()
            self.A = [rng.normal(size=(d, D[i], D[i+1])) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)]
        elif fill == 'postpone':
            self.A = (len(D) - 1) * [None]
        else:
            raise ValueError(f'fill = {fill} invalid; must be a number, "random", "random real" or "postpone".')
        if fill != 'postpone':
            # enforce block sparsity structure dictated by quantum numbers
            for i in range(len(self.A)):
                enforce_qsparsity(self.A[i], [self.qd, self.qD[i], -self.qD[i+1]])

    @classmethod
    def construct_random(cls, nsites: int, qd: Sequence[int], qnum_sector: int, max_vdim: int=256, dtype='complex', rng: np.random.Generator=None):
        """
        Construct a matrix product state with random normal tensor entries,
        given an overall quantum number sector and maximum virtual bond dimension.
        """
        assert nsites > 0
        # require NumPy array
        qd = np.asarray(qd)
        if rng is None:
            rng = np.random.default_rng()
        qD = (nsites + 1) * [None]
        # dummy left virtual bond; set quantum number to zero
        qD[0] = [0]
        # dummy right virtual bond; set quantum number to overall quantum number sector
        qD[-1] = [qnum_sector]
        # virtual bond quantum numbers on left half
        for l in range(1, (nsites + 1) // 2):
            # enumerate all combinations of left bond quantum numbers and local physical quantum numbers
            qnums_full = qnumber_outer_sum([qD[l - 1], qd]).reshape(-1)
            if len(qnums_full) <= max_vdim:
                qD[l] = qnums_full
            else:
                # randomly select quantum numbers
                idx = rng.choice(len(qnums_full), size=max_vdim, replace=False)
                qD[l] = qnums_full[idx]
        # virtual bond quantum numbers on right half
        for l in reversed(range((nsites + 1) // 2, nsites)):
            # enumerate all combinations of right bond quantum numbers and local physical quantum numbers
            qnums_full = qnumber_outer_sum([qD[l + 1], -qd]).reshape(-1)
            if len(qnums_full) <= max_vdim:
                qD[l] = qnums_full
            else:
                # randomly select quantum numbers
                idx = rng.choice(len(qnums_full), size=max_vdim, replace=False)
                qD[l] = qnums_full[idx]
        return cls(qd, qD, fill=('random' if (dtype == complex or dtype == 'complex') else 'random real'), rng=rng)

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
        D = [self.A[i].shape[1] for i in range(len(self.A))]
        D.append(self.A[-1].shape[2])
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
        Left- or right-orthonormalize the MPS using QR decompositions.
        """
        if len(self.A) == 0:
            return 1

        if mode == 'left':
            for i in range(len(self.A) - 1):
                self.A[i], self.A[i+1], self.qD[i+1] = local_orthonormalize_left_qr(self.A[i], self.A[i+1], self.qd, self.qD[i:i+2])
            # last tensor
            self.A[-1], T, self.qD[-1] = local_orthonormalize_left_qr(self.A[-1], np.array([[[1]]]), self.qd, self.qD[-2:])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            nrm = T[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.A[-1] = -self.A[-1]
                nrm = -nrm
            return nrm
        if mode == 'right':
            for i in reversed(range(1, len(self.A))):
                self.A[i], self.A[i-1], self.qD[i] = local_orthonormalize_right_qr(self.A[i], self.A[i-1], self.qd, self.qD[i:i+2])
            # first tensor
            self.A[0], T, self.qD[0] = local_orthonormalize_right_qr(self.A[0], np.array([[[1]]]), self.qd, self.qD[:2])
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            nrm = T[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.A[0] = -self.A[0]
                nrm = -nrm
            return nrm
        raise ValueError(f'mode = {mode} invalid; must be "left" or "right".')

    def compress(self, tol: float, mode='svd', direction='left'):
        """
        Compress and orthonormalize a MPS.

        For 'mode == "svd"', the function performs site-local SVDs and singular value truncations.
        For 'mode == "density"', the function implements the rounding compression
        based on the local density matrix.

        Returns original norm and scaling factor due to compression.
        """
        if mode == 'svd':
            return self._compress_svd(tol, direction)
        if mode == 'density':
            return self._compress_density(tol)
        raise ValueError(f'mode = {mode} invalid; must be "svd" or "density".')

    def _compress_svd(self, tol: float, direction: str):
        """
        Compress and orthonormalize a MPS by site-local SVDs and singular value truncations.

        Returns original norm and scaling factor due to compression.
        """
        if direction == 'left':
            # transform to right-canonical form first
            nrm = self.orthonormalize(mode='right')
            for i in range(len(self.A) - 1):
                self.A[i], self.A[i+1], self.qD[i+1] = local_orthonormalize_left_svd(self.A[i], self.A[i+1], self.qd, self.qD[i:i+2], tol)
                assert is_qsparse(self.A[i], [self.qd, self.qD[i], -self.qD[i+1]]), \
                    'sparsity pattern of MPS tensor does not match quantum numbers'
            # last tensor
            self.A[-1], T, self.qD[-1] = local_orthonormalize_left_svd(self.A[-1], np.array([[[1]]]), self.qd, self.qD[-2:], tol)
            assert is_qsparse(self.A[-1], [self.qd, self.qD[-2], -self.qD[-1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
            assert T.shape == (1, 1, 1)
            # absorb potential phase factor into MPS tensor
            self.A[-1] *= T[0, 0, 0] / abs(T[0, 0, 0])
            return (nrm, abs(T[0, 0, 0]))
        if direction == 'right':
            # transform to left-canonical form first
            nrm = self.orthonormalize(mode='left')
            for i in reversed(range(1, len(self.A))):
                self.A[i], self.A[i-1], self.qD[i] = local_orthonormalize_right_svd(self.A[i], self.A[i-1], self.qd, self.qD[i:i+2], tol)
                assert is_qsparse(self.A[i], [self.qd, self.qD[i], -self.qD[i+1]]), \
                    'sparsity pattern of MPS tensor does not match quantum numbers'
            # first tensor
            self.A[0], T, self.qD[0] = local_orthonormalize_right_svd(self.A[0], np.array([[[1]]]), self.qd, self.qD[:2], tol)
            assert is_qsparse(self.A[0], [self.qd, self.qD[0], -self.qD[1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
            assert T.shape == (1, 1, 1)
            # absorb potential phase factor into MPS tensor
            self.A[0] *= T[0, 0, 0] / abs(T[0, 0, 0])
            return (nrm, abs(T[0, 0, 0]))
        raise ValueError(f'direction = {direction} invalid; must be "left" or "right".')

    def _compress_density(self, tol: float):
        """
        Compress and orthonormalize a MPS, implementing the rounding compression
        discussed in the paper "From density-matrix renormalization group to matrix product states" (cited below),
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
        b = np.array([1], dtype=self.A[-1].dtype).reshape(1, 1, 1)
        # current 'U' matrix
        u = np.array([1], dtype=self.A[-1].dtype).reshape(1, 1, 1)
        for i in reversed(range(1, self.nsites)):
            # compute new 'B' block
            b = np.tensordot(b, u.conj(), axes=([0, 2], [0, 2]))
            b = np.tensordot(self.A[i], b, axes=([2], [0]))
            # compute density matrix
            rho = np.tensordot(b, left_envs[i], axes=([1], [0]))
            rho = np.tensordot(rho, b.conj(), axes=([2], [1]))
            # diagonalize density matrix
            assert rho.ndim == 4
            orig_shape = rho.shape[0:2]
            rho = rho.reshape((np.prod(rho.shape[0:2]), np.prod(rho.shape[2:4])))
            qnums_rho = qnumber_flatten([self.qd, -self.qD[i+1]])
            assert is_qsparse(rho, [qnums_rho, -qnums_rho])
            u, _, qnums_eig = eigh(rho, qnums_rho, tol=tol)
            qnums_eig = -qnums_eig
            assert is_qsparse(u, [qnums_rho, qnums_eig])
            u = u.reshape(orig_shape[0], orig_shape[1], u.shape[1]).transpose((0, 2, 1))
            assert is_qsparse(u, [self.qd, qnums_eig, -self.qD[i+1]])
            self.A[i] = u
            self.qD[i] = qnums_eig
        # for the leftmost site, we merely need to find the block, as the bond is already truncated
        b = np.tensordot(b, u.conj(), axes=([0, 2], [0, 2]))
        b = np.tensordot(self.A[0], b, axes=([2], [0]))
        s = np.linalg.norm(b.reshape(-1))
        self.A[0] = b / s
        return (nrm, s / nrm)

    def as_vector(self) -> np.ndarray:
        """
        Merge all tensors to obtain the vector representation on the full Hilbert space.
        """
        psi = self.A[0]
        for i in range(1, len(self.A)):
            psi = merge_mps_tensor_pair(psi, self.A[i])
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
        assert v.ndim == 1 and len(v) == d**nsites, f"`v` has length {len(v)}, expecting {d**nsites}."
        # allocate a MPS with dummy virtual bonds of dimension 1
        mps = cls(d*[0], [[0] for _ in range(nsites + 1)], fill='postpone')
        # endow `v` with a dummy left virtual bond dimension
        v = np.reshape(v, (1, len(v)))
        for i in range(nsites):
            assert v.shape[1] == d**(nsites - i)
            Dleft = v.shape[0]
            # partition physical dimension into local dimension and the rest,
            # split by SVD and assign the right SVD matrix to `v`
            u, s, v = np.linalg.svd(v.reshape((Dleft*d, d**(nsites-i-1))), full_matrices=False)
            # truncate small singular values
            idx = retained_bond_indices(s, tol)
            u = u[:, idx]
            v = v[idx, :]
            s = s[idx]
            # use broadcasting to distribute singular values to the right
            v = v * s[:, None]
            # assign MPS tensor at current site i, transposing physical dimension to the front
            mps.A[i] = u.reshape((Dleft, d, len(s))).transpose((1, 0, 2))
            mps.qD[i + 1] = len(s) * [0]
        assert v.shape == (1, 1)
        # include scalar factor in last MPS tensor
        mps.A[-1] *= v[0, 0]
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


def local_orthonormalize_left_qr(A: np.ndarray, Anext: np.ndarray, qd: Sequence[int], qD: Sequence[Sequence[int]]):
    """
    Left-orthonormalize the local site tensor `A` by a QR decomposition,
    and update the tensor at the next site.
    """
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qd, qD[0]])
    Q, R, qbond = qr(A.reshape((s[0]*s[1], s[2])), q0, qD[1])
    A = Q.reshape((s[0], s[1], Q.shape[1]))
    # update Anext tensor: multiply with R from left
    Anext = np.tensordot(R, Anext, (1, 1)).transpose((1, 0, 2))
    return (A, Anext, qbond)


def local_orthonormalize_right_qr(A: np.ndarray, Aprev: np.ndarray, qd: Sequence[int], qD: Sequence[Sequence[int]]):
    """
    Right-orthonormalize the local site tensor `A` by a QR decomposition,
    and update the tensor at the previous site.
    """
    # flip left and right virtual bond dimensions
    A = A.transpose((0, 2, 1))
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qd, -qD[1]])
    Q, R, qbond = qr(A.reshape((s[0]*s[1], s[2])), q0, -qD[0])
    A = Q.reshape((s[0], s[1], Q.shape[1])).transpose((0, 2, 1))
    # update Aprev tensor: multiply with R from right
    Aprev = np.tensordot(Aprev, R, (2, 1))
    return (A, Aprev, -qbond)


def local_orthonormalize_left_svd(A, Anext, qd: Sequence[int], qD: Sequence[Sequence[int]], tol: float):
    """
    Left-orthonormalize the local site tensor `A` by a SVD,
    and update the tensor at the next site.
    """
    # perform SVD and replace A by reshaped U matrix
    s = A.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qd, qD[0]])
    U, sigma, V, qbond = split_matrix_svd(A.reshape((s[0]*s[1], s[2])), q0, qD[1], tol)
    A = U.reshape((s[0], s[1], U.shape[1]))
    # update Anext tensor: multiply with (sigma @ V) from left
    Anext = np.tensordot(sigma[:, None] * V, Anext, (1, 1)).transpose((1, 0, 2))
    return (A, Anext, qbond)


def local_orthonormalize_right_svd(A, Aprev, qd: Sequence[int], qD: Sequence[Sequence[int]], tol: float):
    """
    Right-orthonormalize the local site tensor `A` by a SVD,
    and update the tensor at the previous site.
    """
    # flip physical and left virtual bond dimension
    A = A.transpose((1, 0, 2))
    # perform SVD and replace A by reshaped V matrix
    s = A.shape
    assert len(s) == 3
    q1 = qnumber_flatten([-qd, qD[1]])
    U, sigma, V, qbond = split_matrix_svd(A.reshape((s[0], s[1]*s[2])), qD[0], q1, tol)
    A = V.reshape((V.shape[0], s[1], s[2]))
    # undo flip of physical and left virtual bond dimension
    A = A.transpose((1, 0, 2))
    # update Aprev tensor: multiply with (U @ sigma) from right
    Aprev = np.tensordot(Aprev, U * sigma, (2, 0))
    return (A, Aprev, qbond)


def merge_mps_tensor_pair(A0: np.ndarray, A1: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPS tensors.
    """
    A = np.einsum(A0, (0, 2, 3), A1, (1, 3, 4), (0, 1, 2, 4), optimize=True)
    # combine original physical dimensions
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2], A.shape[3]))
    return A


def split_mps_tensor(A: np.ndarray, qd0: Sequence[int], qd1: Sequence[int], qD: Sequence[Sequence[int]], svd_distr: str, tol=0):
    """
    Split a MPS tensor with dimension `d0*d1 x D0 x D2` into two MPS tensors
    with dimensions `d0 x D0 x D1` and `d1 x D1 x D2`, respectively.
    """
    assert A.ndim == 3
    d0 = len(qd0)
    d1 = len(qd1)
    assert d0 * d1 == A.shape[0], 'physical dimension of MPS tensor must be equal to d0 * d1'
    # reshape as matrix and split by SVD
    A = A.reshape((d0, d1, A.shape[1], A.shape[2])).transpose((0, 2, 1, 3))
    s = A.shape
    q0 = qnumber_flatten([ qd0, qD[0]])
    q1 = qnumber_flatten([-qd1, qD[1]])
    A0, sigma, A1, qbond = split_matrix_svd(A.reshape((s[0]*s[1], s[2]*s[3])), q0, q1, tol)
    A0.shape = (s[0], s[1], len(sigma))
    A1.shape = (len(sigma), s[2], s[3])
    # use broadcasting to distribute singular values
    if svd_distr == 'left':
        A0 = A0 * sigma
    elif svd_distr == 'right':
        A1 = A1 * sigma[:, None, None]
    elif svd_distr == 'sqrt':
        s = np.sqrt(sigma)
        A0 = A0 * s
        A1 = A1 * s[:, None, None]
    else:
        raise ValueError('svd_distr parameter must be "left", "right" or "sqrt".')
    # move physical dimension to the front
    A1 = A1.transpose((1, 0, 2))
    return (A0, A1, qbond)


def add_mps(mps0: MPS, mps1: MPS, alpha=1) -> MPS:
    """
    Logical addition of two matrix product states (effectively sum virtual bond dimensions),
    with the second MPS scaled by 'alpha'.
    """
    # number of lattice sites must agree
    assert mps0.nsites == mps1.nsites
    L = mps0.nsites
    # physical quantum numbers must agree
    assert np.array_equal(mps0.qd, mps1.qd)
    d = len(mps0.qd)

    # initialize with dummy tensors and bond quantum numbers
    mps = MPS(mps0.qd, (L+1)*[[0]])

    if L == 1:
        # single site
        # dummy bond quantum numbers must agree
        assert np.array_equal(mps0.qD[0], mps1.qD[0])
        assert np.array_equal(mps0.qD[1], mps1.qD[1])
        mps.qD[0] = mps0.qD[0].copy()
        mps.qD[1] = mps0.qD[1].copy()
        # simply add MPS tensors
        mps.A[0] = mps0.A[0] + alpha*mps1.A[0]
        # consistency check
        assert is_qsparse(mps.A[0], [mps.qd, mps.qD[0], -mps.qD[1]]), \
            'sparsity pattern of MPS tensor does not match quantum numbers'
    elif L > 1:
        # combine virtual bond quantum numbers
        # leading and trailing (dummy) bond quantum numbers must agree
        assert np.array_equal(mps0.qD[ 0], mps1.qD[ 0])
        assert np.array_equal(mps0.qD[-1], mps1.qD[-1])
        mps.qD[ 0] = mps0.qD[ 0].copy()
        mps.qD[-1] = mps0.qD[-1].copy()
        # intermediate bond quantum numbers
        for i in range(1, L):
            mps.qD[i] = np.concatenate((mps0.qD[i], mps1.qD[i]))

        # leftmost tensor
        mps.A[0] = np.block([mps0.A[0], alpha*mps1.A[0]])
        # intermediate tensors
        for i in range(1, L - 1):
            s0 = mps0.A[i].shape
            s1 = mps1.A[i].shape
            # form block-diagonal tensor
            mps.A[i] = np.block([[mps0.A[i], np.zeros((d, s0[1], s1[2]))], [np.zeros((d, s1[1], s0[2])), mps1.A[i]]])
        # rightmost tensor
        mps.A[-1] = np.block([[mps0.A[-1]], [mps1.A[-1]]])

        # consistency check
        for i in range(1, L):
            assert is_qsparse(mps.A[i], [mps.qd, mps.qD[i], -mps.qD[i+1]]), \
                'sparsity pattern of MPS tensor does not match quantum numbers'
    return mps
