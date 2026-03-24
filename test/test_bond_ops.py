import numpy as np
import pytenet as ptn


def test_qr():

    rng = np.random.default_rng()

    A = ptn.crandn((23, 15), rng)

    # fictitious quantum numbers
    q0 = rng.integers(-2, 3, size=A.shape[0])
    q1 = rng.integers(-2, 3, size=A.shape[1])

    # enforce block sparsity structure dictated by quantum numbers
    ptn.enforce_qsparsity(A, [q0, -q1])

    # perform QR decomposition, taking quantum numbers into account
    Q, R, qinterm = ptn.qr(A, q0, q1)

    assert np.allclose(Q @ R, A, rtol=1e-14), "Q @ R must match A matrix"

    assert np.allclose(Q.conj().T @ Q, np.identity(Q.shape[1]), rtol=1e-14), \
        "columns of Q matrix must be orthonormalized"

    assert ptn.is_qsparse(Q, [q0, -qinterm]), \
        "sparsity pattern of Q matrix must match quantum numbers"
    assert ptn.is_qsparse(R, [qinterm, -q1]), \
        "sparsity pattern of R matrix must match quantum numbers"


def test_split_matrix_svd():

    rng = np.random.default_rng()

    A = ptn.crandn((17, 26), rng)

    # fictitious quantum numbers
    q0 = rng.integers(-2, 3, size=A.shape[0])
    q1 = rng.integers(-2, 3, size=A.shape[1])

    # enforce block sparsity structure dictated by quantum numbers
    ptn.enforce_qsparsity(A, [q0, -q1])

    # perform SVD decomposition without truncation
    u, s, v, qinterm = ptn.split_matrix_svd(A, q0, q1, 0.)

    assert np.allclose((u * s) @ v, A, rtol=1e-13), \
        "U @ S @ V must match A matrix"

    assert np.allclose(u.conj().T @ u, np.identity(u.shape[1]), rtol=1e-14), \
        "columns of U matrix must be orthonormalized"

    assert ptn.is_qsparse(u, [q0, -qinterm]), \
        "sparsity pattern of U matrix must match quantum numbers"
    assert ptn.is_qsparse(v, [qinterm, -q1]), \
        "sparsity pattern of V matrix must match quantum numbers"

    s_norm = np.linalg.norm(s)

    # perform SVD decomposition with truncation
    u, s, v, qinterm = ptn.split_matrix_svd(A, q0, q1, 0.15)

    assert abs(np.linalg.norm((u * s) @ v - A)
               - np.sqrt(s_norm**2 - np.linalg.norm(s)**2)) < 1e-14, \
        "weight of truncated singular values must agree with norm of matrix difference"

    assert np.allclose(u.conj().T @ u, np.identity(u.shape[1]), rtol=1e-14), \
        "columns of U matrix must be orthonormalized"

    assert ptn.is_qsparse(u, [q0, -qinterm]), \
        "sparsity pattern of U matrix must match quantum numbers"
    assert ptn.is_qsparse(v, [qinterm, -q1]), \
        "sparsity pattern of V matrix must match quantum numbers"


def test_eigh():

    rng = np.random.default_rng()

    A = ptn.crandn((13, 13), rng)
    A = 0.5 * (A + A.conj().T)  # ensure that 'A' is Hermitian

    # fictitious quantum numbers
    q0 = rng.integers(-2, 3, size=A.shape[0])

    # enforce block sparsity structure dictated by quantum numbers
    ptn.enforce_qsparsity(A, [q0, -q0])

    # perform diagonalisation without truncation
    U, evals, qinterm = ptn.eigh(A, q0)

    assert np.allclose(U @ np.diag(evals) @ U.conj().T, A, rtol=1e-13), \
        "U @ eval @ U^H must match A matrix"

    assert np.allclose(U.conj().T @ U, np.identity(U.shape[1]), rtol=1e-14), \
        "columns of U matrix must be orthonormalized"

    assert ptn.is_qsparse(U, [q0, -qinterm]), \
        "sparsity pattern of U matrix must match quantum numbers"

    eig_norm = np.linalg.norm(evals)

    # perform diagonalization with truncation
    U, evals, qinterm = ptn.eigh(A, q0, tol=0.15)

    assert abs(np.linalg.norm(U @ np.diag(evals) @ U.conj().T - A)
               - np.sqrt(eig_norm**2 - np.linalg.norm(evals)**2)) < 1e-14, \
        "weight of truncated singular values must agree with norm of matrix difference"

    assert np.allclose(U.conj().T @ U, np.identity(U.shape[1]), rtol=1e-14), \
        "columns of U matrix must be orthonormalized"

    assert ptn.is_qsparse(U, [q0, -qinterm]), \
        "sparsity pattern of U matrix must match quantum numbers"
