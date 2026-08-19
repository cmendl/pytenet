import math
import time
import functools
import autoray as ar
from autoray import numpy as np
import torch
import pytenet as ptn


def test_lanczos_iteration():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            n = 256
            numiter = 24

            # random Hermitian matrix
            a = ptn.crandn((n, n), rng=rng, scale=1/math.sqrt(n))
            a = 0.5 * (a + a.conj().T)

            # random complex starting vector
            vstart = ptn.crandn(n, rng=rng, scale=1/math.sqrt(n))

            if backend == "torch" and torch.cuda.is_available():
                a = ar.to_device(a, "gpu")
                vstart = ar.to_device(vstart, "gpu")

            # simply use `a` as linear transformation
            alpha, beta, v = ptn.lanczos_iteration(functools.partial(lambda mat, x: mat @ x, a),
                                                   vstart, numiter)

            # check orthogonality of Lanczos vectors
            assert np.allclose(v.T.conj() @ v,
                               np.identity(numiter, like=v), rtol=1e-12), \
                "matrix of Lanczos vectors must be orthonormalized"

            # Lanczos vectors must tridiagonalize `a`
            t = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
            assert np.allclose(v.conj().T @ a @ v, ar.astype(t, a.dtype), rtol=1e-12), \
                "Lanczos vectors must tridiagonalize `a`"


def test_arnoldi_iteration():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            n = 256
            numiter = 24

            # random matrix
            a = ptn.crandn((n, n), rng)
            # random complex starting vector
            vstart = ptn.crandn(n, rng=rng, scale=1/math.sqrt(n))

            if backend == "torch" and torch.cuda.is_available():
                a = ar.to_device(a, "gpu")
                vstart = ar.to_device(vstart, "gpu")

            # simply use `a` as linear transformation
            hess, v = ptn.arnoldi_iteration(functools.partial(lambda mat, x: mat @ x, a),
                                            vstart, numiter)

            # check orthogonality of Arnoldi vectors
            assert np.allclose(v.conj().T @ v, np.identity(v.shape[1], like=v), rtol=1e-12), \
                "matrix of Arnoldi vectors must be orthonormalized"

            assert np.allclose(v.conj().T @ a @ v, hess, rtol=1e-12), \
                "Arnoldi vectors must transform `a` to upper Hessenberg form"


def test_eigh_krylov():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            n = 196
            numiter = 30
            numeig  = 2

            # random Hermitian matrix
            a = ptn.crandn((n, n), rng=rng, scale=1/math.sqrt(n))
            a = 0.5 * (a + a.conj().T)

            # random complex starting vector
            vstart = ptn.crandn(n, rng=rng, scale=1/math.sqrt(n))

            if backend == "torch" and torch.cuda.is_available():
                a = ar.to_device(a, "gpu")
                vstart = ar.to_device(vstart, "gpu")

            # simply use `a` as linear transformation;
            w, u_ritz = ptn.eigh_krylov(functools.partial(lambda mat, x: mat @ x, a),
                                        vstart, numiter, numeig)

            # check orthogonality of Ritz matrix
            assert np.allclose(u_ritz.conj().T @ u_ritz,
                               np.identity(numeig, like=u_ritz), rtol=1e-12), \
                "matrix of Ritz eigenvectors must be orthonormalized"

            # check U^H a U = diag(w)
            assert np.allclose(u_ritz.conj().T @ a @ u_ritz,
                               np.diag(ar.astype(w, a.dtype)), rtol=1e-12), \
                "Ritz eigenvectors must diagonalize a within Krylov subspace"

            # reference eigenvalues
            w_ref = np.linalg.eigvalsh(a)

            # compare lowest eigenvalues
            assert abs(w[0] - w_ref[0]) < 0.001, \
                "lowest Lanczos eigenvalue should approximate exact eigenvalue"

            assert abs(w[1] - w_ref[1]) < 0.02, \
                "second-lowest Lanczos eigenvalue should approximate exact eigenvalue"


def test_expm_krylov():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            n = 320
            numiter = 12
            # time step
            dt = 0.4 + 0.2j

            # random complex matrix
            a = ptn.crandn((n, n), rng=rng, scale=1/math.sqrt(n))

            # random complex vector
            vec = ptn.crandn(n, rng=rng, scale=1/math.sqrt(n))

            if backend == "torch" and torch.cuda.is_available():
                a = ar.to_device(a, "gpu")
                vec = ar.to_device(vec, "gpu")

            # Krylov subspace approximation of expm(dt*a) @ vec, general case
            vt = ptn.expm_krylov(functools.partial(lambda mat, x: mat @ x, a),
                                 vec, dt, numiter, hermitian=False)
            # reference
            vt_ref = ar.do("linalg.expm", dt*a) @ vec
            assert np.allclose(vt, vt_ref, rtol=1e-11), \
                "Krylov subspace approximation of expm(dt*a) @ vec should match reference"

            # symmetrize
            a = 0.5 * (a + a.conj().T)
            # Krylov subspace approximation of expm(dt*a) @ vec, Hermitian case
            vt = ptn.expm_krylov(functools.partial(lambda mat, x: mat @ x, a),
                                 vec, dt, numiter, hermitian=True)
            # reference
            vt_ref = ar.do("linalg.expm", dt*a) @ vec
            assert np.allclose(vt, vt_ref, rtol=1e-11), \
                "Krylov subspace approximation of expm(dt*a) @ vec should match reference"
