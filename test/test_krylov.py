import numpy as np
from scipy.linalg import expm
import pytenet as ptn


def test_lanczos_iteration():

    rng = np.random.default_rng()

    n = 256
    numiter = 24

    # random Hermitian matrix
    a = ptn.crandn((n, n), rng) / np.sqrt(n)
    a = 0.5 * (a + a.conj().T)

    # random complex starting vector
    vstart = ptn.crandn(n, rng) / np.sqrt(n)

    # simply use `a` as linear transformation
    alpha, beta, v = ptn.lanczos_iteration(lambda x: a @ x, vstart, numiter)

    # check orthogonality of Lanczos vectors
    assert np.allclose(v.T.conj() @ v, np.identity(numiter), rtol=1e-12), \
        "matrix of Lanczos vectors must be orthonormalized"

    # Lanczos vectors must tridiagonalize `a`
    t = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    assert np.allclose(v.conj().T @ a @ v, t, rtol=1e-12), \
        "Lanczos vectors must tridiagonalize `a`"


def test_arnoldi_iteration():

    rng = np.random.default_rng()

    n = 256
    numiter = 24

    # random matrix
    a = ptn.crandn((n, n), rng)
    # random complex starting vector
    vstart = ptn.crandn(n, rng) / np.sqrt(n)

    # simply use `a` as linear transformation
    hess, v = ptn.arnoldi_iteration(lambda x: a @ x, vstart, numiter)

    # check orthogonality of Arnoldi vectors
    assert np.allclose(v.conj().T @ v, np.identity(v.shape[1]), rtol=1e-12), \
        "matrix of Arnoldi vectors must be orthonormalized"

    assert np.allclose(v.conj().T @ a @ v, hess, rtol=1e-12), \
        "Arnoldi vectors must transform `a` to upper Hessenberg form"


def test_eigh_krylov():

    rng = np.random.default_rng()

    n = 196
    numiter = 30
    numeig  = 2

    # random Hermitian matrix
    a = ptn.crandn((n, n), rng) / np.sqrt(n)
    a = 0.5 * (a + a.conj().T)

    # random complex starting vector
    vstart = ptn.crandn(n, rng) / np.sqrt(n)

    # simply use `a` as linear transformation;
    w, u_ritz = ptn.eigh_krylov(lambda x: a @ x, vstart, numiter, numeig)

    # check orthogonality of Ritz matrix
    assert np.allclose(u_ritz.conj().T @ u_ritz, np.identity(numeig), rtol=1e-12), \
        "matrix of Ritz eigenvectors must be orthonormalized"

    # check U^H a U = diag(w)
    assert np.allclose(u_ritz.conj().T @ a @ u_ritz, np.diag(w), rtol=1e-12), \
        "Ritz eigenvectors must diagonalize a within Krylov subspace"

    # reference eigenvalues
    w_ref = np.linalg.eigvalsh(a)

    # compare lowest eigenvalues
    assert abs(w[0] - w_ref[0]) < 0.001, \
        "lowest Lanczos eigenvalue should approximate exact eigenvalue"

    assert abs(w[1] - w_ref[1]) < 0.02, \
        "second-lowest Lanczos eigenvalue should approximate exact eigenvalue"


def test_expm_krylov():

    rng = np.random.default_rng()

    n = 320
    numiter = 12
    # time step
    dt = 0.4 + 0.2j

    # random complex matrix
    a = ptn.crandn((n, n), rng) / np.sqrt(n)

    # random complex vector
    vec = ptn.crandn(n, rng) / np.sqrt(n)

    # Krylov subspace approximation of expm(dt*a) @ vec, general case
    vt = ptn.expm_krylov(lambda x: a @ x, vec, dt, numiter, hermitian=False)
    # reference
    vt_ref = expm(dt*a) @ vec
    assert np.allclose(vt, vt_ref, rtol=1e-11), \
        "Krylov subspace approximation of expm(dt*a) @ vec should match reference"

    # symmetrize
    a = 0.5 * (a + a.conj().T)
    # Krylov subspace approximation of expm(dt*a) @ vec, Hermitian case
    vt = ptn.expm_krylov(lambda x: a @ x, vec, dt, numiter, hermitian=True)
    # reference
    vt_ref = expm(dt*a) @ vec
    assert np.allclose(vt, vt_ref, rtol=1e-11), \
        "Krylov subspace approximation of expm(dt*a) @ vec should match reference"
