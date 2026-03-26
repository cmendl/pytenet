import numpy as np
import pytenet as ptn


def test_block_sparse_qr():

    rng = np.random.default_rng()

    a = ptn.crandn((23, 15), rng)

    # fictitious quantum numbers
    q0 = rng.integers(-2, 3, size=a.shape[0])
    q1 = rng.integers(-2, 3, size=a.shape[1])

    # enforce block sparsity structure dictated by quantum numbers
    ptn.enforce_qsparsity(a, [q0, -q1])

    # perform QR decomposition, taking quantum numbers into account
    q, r, qinterm = ptn.block_sparse_qr(a, q0, q1)

    assert np.allclose(q @ r, a, rtol=1e-14), "q @ r must match `a` matrix"

    assert np.allclose(q.conj().T @ q, np.identity(q.shape[1]), rtol=1e-14), \
        "columns of `q` matrix must be orthonormalized"

    assert ptn.is_qsparse(q, [q0, -qinterm]), \
        "sparsity pattern of `q` matrix must match quantum numbers"
    assert ptn.is_qsparse(r, [qinterm, -q1]), \
        "sparsity pattern of `r` matrix must match quantum numbers"


def test_block_sparse_eigh():

    rng = np.random.default_rng()

    a = ptn.crandn((13, 13), rng)
    a = 0.5 * (a + a.conj().T)  # ensure that `a` is Hermitian

    # fictitious quantum numbers
    q0 = rng.integers(-2, 3, size=a.shape[0])

    # enforce block sparsity structure dictated by quantum numbers
    ptn.enforce_qsparsity(a, [q0, -q0])

    # perform diagonalization
    u, evals, qinterm = ptn.block_sparse_eigh(a, q0)

    assert np.allclose(u @ np.diag(evals) @ u.conj().T, a, rtol=1e-13), \
        "u @ evals @ u^H must match `a` matrix"

    assert np.allclose(u.conj().T @ u, np.identity(u.shape[1]), rtol=1e-14), \
        "`u` matrix must be an isometry"

    assert ptn.is_qsparse(u, [q0, -qinterm]), \
        "sparsity pattern of `u` matrix must match quantum numbers"


def test_block_sparse_svd():

    rng = np.random.default_rng()

    a = ptn.crandn((73, 52), rng)

    # fictitious quantum numbers
    q0 = rng.integers(-2, 3, size=a.shape[0])
    q1 = rng.integers(-2, 3, size=a.shape[1])

    # enforce block sparsity structure dictated by quantum numbers
    ptn.enforce_qsparsity(a, [q0, -q1])

    # perform SVD decomposition
    u, s, v, qinterm = ptn.block_sparse_svd(a, q0, q1)

    assert np.allclose((u * s) @ v, a, rtol=1e-13), \
        "u @ s @ v must match `a` matrix"

    assert np.allclose(u.conj().T @ u, np.identity(u.shape[1]), rtol=1e-14), \
        "`u` matrix must be an isometry"
    assert np.allclose(v @ v.conj().T, np.identity(v.shape[0]), rtol=1e-14), \
        "`v` matrix must be an isometry"

    assert ptn.is_qsparse(u, [q0, -qinterm]), \
        "sparsity pattern of `u` matrix must match quantum numbers"
    assert ptn.is_qsparse(v, [qinterm, -q1]), \
        "sparsity pattern of `v` matrix must match quantum numbers"
