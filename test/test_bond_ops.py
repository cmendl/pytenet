import numpy as np
import pytenet as ptn


def test_split_matrix_svd():

    rng = np.random.default_rng()

    a = ptn.crandn((17, 26), rng)

    # fictitious quantum numbers
    q0 = rng.integers(-2, 3, size=a.shape[0])
    q1 = rng.integers(-2, 3, size=a.shape[1])

    # enforce block sparsity structure dictated by quantum numbers
    ptn.enforce_qsparsity(a, [q0, -q1])

    # perform SVD decomposition without truncation
    u, s, v, qinterm = ptn.split_block_sparse_matrix_svd(a, q0, q1, 0.)

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

    s_norm = np.linalg.norm(s)

    # perform SVD decomposition with truncation
    u, s, v, qinterm = ptn.split_block_sparse_matrix_svd(a, q0, q1, 0.15)

    assert abs(np.linalg.norm((u * s) @ v - a)
               - np.sqrt(s_norm**2 - np.linalg.norm(s)**2)) < 1e-14, \
        "weight of truncated singular values must agree with norm of matrix difference"

    assert np.allclose(u.conj().T @ u, np.identity(u.shape[1]), rtol=1e-14), \
        "`u` matrix must be an isometry"
    assert np.allclose(v @ v.conj().T, np.identity(v.shape[0]), rtol=1e-14), \
        "`v` matrix must be an isometry"

    assert ptn.is_qsparse(u, [q0, -qinterm]), \
        "sparsity pattern of `u` matrix must match quantum numbers"
    assert ptn.is_qsparse(v, [qinterm, -q1]), \
        "sparsity pattern of `v` matrix must match quantum numbers"
