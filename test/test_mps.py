import math
import time
import autoray as ar
from autoray import numpy as np
import torch
import pytenet as ptn


def test_mps_orthonormalize():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # create random matrix product state
            d = 7
            b = [1, 4, 15, 13, 7, 1]
            mps0 = ptn.MPS(ptn.random_qnumbers(-2, 3, size=d, rng=rng),
                           [ptn.random_qnumbers(-2, 3, size=bi, rng=rng) for bi in b],
                        fill="random", rng=rng)
            if backend == "torch" and torch.cuda.is_available():
                mps0.to_device("gpu")

            assert mps0.bond_dims == b, "virtual bond dimensions do not match reference values"

            # wavefunction on full Hilbert space
            psi = mps0.to_vector()

            # performing left-orthonormalization...
            c_left = mps0.orthonormalize(mode="left")

            assert mps0.bond_dims[1] <= d, \
                "virtual bond dimension can only increase by a factor of `d` per site"

            for i in range(mps0.nsites):
                assert ptn.is_qsparse(mps0.a[i], [mps0.qbonds[i], mps0.qsite,
                                                  ptn.neg_qnumbers(mps0.qbonds[i+1])]), \
                    "sparsity pattern of MPS tensors must match quantum numbers"

            psi_left = mps0.to_vector()
            # wavefunction should now be normalized
            assert abs(np.linalg.norm(psi_left) - 1.) < 1e-12, \
                "MPS wavefunction must be normalized"

            # wavefunctions before and after left-normalization must match
            # (up to normalization factor)
            assert np.linalg.norm(c_left*psi_left - psi) < 1e-10, \
                "wavefunctions before and after left-normalization must match"

            # check left-orthonormalization
            for i in range(mps0.nsites):
                s = mps0.a[i].shape
                assert s[1] == d
                q = mps0.a[i].reshape((s[0]*s[1], s[2]))
                assert np.allclose(q.conj().T @ q, np.identity(s[2], like=q), rtol=1e-12), \
                    "MPS tensor is not left-orthonormalized"

            # performing right-orthonormalization...
            c_right = mps0.orthonormalize(mode="right")

            assert mps0.bond_dims[-2] <= d, \
                "virtual bond dimension can only increase by a factor of `d` per site"

            for i in range(mps0.nsites):
                assert ptn.is_qsparse(mps0.a[i], [mps0.qbonds[i], mps0.qsite,
                                                  ptn.neg_qnumbers(mps0.qbonds[i+1])]), \
                    "sparsity pattern of MPS tensors must match quantum numbers"

            assert abs(abs(c_right) - 1.) <= 1e-12, \
                "normalization factor must have magnitude 1 due to previous left-orthonormalization"

            psi_right = mps0.to_vector()
            # wavefunctions must match
            assert np.allclose(psi_left, c_right*psi_right, rtol=1e-10), \
                "wavefunctions after left- and right-orthonormalization must match"

            # check right-orthonormalization
            for i in range(mps0.nsites):
                s = mps0.a[i].shape
                assert s[1] == d
                q = np.transpose(mps0.a[i], (2, 1, 0)).reshape((s[1]*s[2], s[0]))
                assert np.allclose(q.conj().T @ q, np.identity(s[0], like=q), rtol=1e-12), \
                    "MPS tensor is not right-orthonormalized"


def test_mps_norm():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # physical quantum numbers
            qsite = ptn.random_qnumbers(-2, 3, size=4, rng=rng)
            # number of lattice sites
            nsites = 6

            # create a random matrix product state
            psi = ptn.MPS.construct_random(
                nsites, qsite, qnum_sector=1, max_vdim=15, rng=rng)
            # rescale to achieve norm of order 1
            for i in range(nsites):
                psi.a[i] *= 5

            if backend == "torch" and torch.cuda.is_available():
                psi.to_device("gpu")

            # calculate the norm of psi using the MPS representation
            nrm = ptn.mps_norm(psi)

            # reference value
            nrm_ref = np.linalg.norm(psi.to_vector())

            # compare
            assert abs(nrm - nrm_ref) / max(abs(nrm_ref), 1e-12) < 1e-12, \
                "matrix product state norm must match reference value"


def test_mps_compress():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # physical quantum numbers
            qsite = [-1, 1, 0]

            for tol in (0., 1e-4):
                for mode, direction in [("svd", "left"), ("svd", "right"), ("density", "any")]:
                    # create random matrix product state
                    b = [1, 30, 83, 102, 75, 23, 1]
                    qbonds = [ptn.random_qnumbers(-1, 2, size=bi, rng=rng) for bi in b]
                    psi = ptn.MPS(qsite, qbonds,
                                  fill="random", rng=rng)
                    if backend == "torch" and torch.cuda.is_available():
                        psi.to_device("gpu")
                    for i in range(psi.nsites):
                        # imitate small entanglement by multiplying bonds with small scaling factors
                        s = np.exp(-30*(rng.uniform(size=psi.bond_dims[i + 1])))
                        s /= np.linalg.norm(s)
                        if backend == "torch" and torch.cuda.is_available():
                            s = ar.to_device(s, "gpu")
                        psi.a[i] = psi.a[i] * s
                        # rescale to achieve norm of order 1
                        psi.a[i] *= 5 / np.linalg.norm(psi.a[i])

                    psi_ref = psi.to_vector()
                    nrm_ref = np.linalg.norm(psi_ref)

                    nrm, scale = psi.compress(tol, mode=mode, direction=direction)

                    # norm of input state vector
                    assert abs(nrm - nrm_ref) < 1e-13

                    assert abs(scale - 1) < (1e-13 if tol == 0 else 1e-2)
                    # must be normalized after compression
                    assert abs(ptn.mps_norm(psi) - 1) < 1e-13

                    # compare with original state vector
                    ctol = (1e-13 if tol == 0 else 0.1)
                    assert np.allclose(nrm*psi.to_vector(), psi_ref, atol=ctol, rtol=ctol)


def test_mps_split_tensor_svd():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # physical dimensions
            d0, d1 = 3, 5
            # outer virtual bond dimensions
            b0, b2 = 14, 17

            a_pair = ptn.crandn((b0, d0*d1, b2), scale=1/math.sqrt(b0*d0*d1*b2), rng=rng)
            if backend == "torch" and torch.cuda.is_available():
                a_pair = ar.to_device(a_pair, "gpu")

            # fictitious quantum numbers
            qsite0 = ptn.random_qnumbers(-2, 3, size=d0, rng=rng)
            qsite1 = ptn.random_qnumbers(-2, 3, size=d1, rng=rng)
            qbonds = [ptn.random_qnumbers(-2, 3, size=b0, rng=rng),
                      ptn.random_qnumbers(-2, 3, size=b2, rng=rng)]

            # enforce block sparsity structure dictated by quantum numbers
            ptn.enforce_qsparsity(a_pair,
                [qbonds[0], ptn.qnumber_flatten([qsite0, qsite1]), ptn.neg_qnumbers(qbonds[1])])

            for svd_distr in ["left", "right", "sqrt"]:
                a0, a1, qbond = ptn.mps_split_tensor_svd(
                    a_pair, qsite0, qsite1, qbonds, svd_distr=svd_distr, tol=0)

                assert ptn.is_qsparse(a0, [qbonds[0], qsite0, ptn.neg_qnumbers(qbond)]), \
                    "sparsity pattern of `a0` tensors must match quantum numbers"
                assert ptn.is_qsparse(a1, [qbond, qsite1, ptn.neg_qnumbers(qbonds[1])]), \
                    "sparsity pattern of `a1` tensors must match quantum numbers"

                # merged tensor must agree with the original tensor
                a_mrg = ptn.mps_merge_tensor_pair(a0, a1)
                assert np.allclose(a_mrg, a_pair, rtol=1e-13), \
                    "splitting and subsequent merging must give the same tensor"


def test_mps_from_vector():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # physical local dimension
            d = 3
            # number of lattice sites
            nsites = 7
            # random vector
            v = ptn.crandn(d**nsites, rng)
            if backend == "torch" and torch.cuda.is_available():
                v = ar.to_device(v, "gpu")
            mps = ptn.MPS.from_vector(d, nsites, v)
            assert np.allclose(mps.to_vector(), v, rtol=1e-13), \
                "MPS constructed from a vector must match original vector"


def test_mps_add():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # physical quantum numbers
            qsite = ptn.random_qnumbers(-2, 3, size=5, rng=rng)

            # create random matrix product states
            qbonds0 = [ptn.random_qnumbers(-2, 3, size=bi, rng=rng)
                       for bi in [1, 8, 15, 23, 18,  9, 1]]
            qbonds1 = [ptn.random_qnumbers(-2, 3, size=bi, rng=rng)
                       for bi in [1, 7, 23, 11, 17, 13, 1]]
            # leading and trailing (dummy) virtual bond quantum numbers must agree
            qbonds1[ 0] = qbonds0[ 0]
            qbonds1[-1] = qbonds0[-1]
            mps0 = ptn.MPS(qsite, qbonds0, fill="random", rng=rng)
            mps1 = ptn.MPS(qsite, qbonds1, fill="random", rng=rng)
            if backend == "torch" and torch.cuda.is_available():
                mps0.to_device("gpu")
                mps1.to_device("gpu")

            # MPS addition
            mps = mps0 + mps1

            # reference calculation
            mps_ref = mps0.to_vector() + mps1.to_vector()

            # compare
            assert np.allclose(mps.to_vector(), mps_ref, rtol=1e-13), \
                "addition of two matrix product states must agree with vector representation"


def test_mps_add_singlesite():

    # separate test for a single site since implementation is a special case

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # physical quantum numbers
            qsite = ptn.random_qnumbers(-2, 3, size=7, rng=rng)

            # create random matrix product states for a single site
            # leading and trailing (dummy) virtual bond quantum numbers
            qbonds = [[-1], [-2]]
            mps0 = ptn.MPS(qsite, qbonds, fill="random", rng=rng)
            mps1 = ptn.MPS(qsite, qbonds, fill="random", rng=rng)
            if backend == "torch" and torch.cuda.is_available():
                mps0.to_device("gpu")
                mps1.to_device("gpu")

            # MPS addition
            mps = mps0 + mps1

            # reference calculation
            mps_ref = mps0.to_vector() + mps1.to_vector()

            # compare
            assert np.allclose(mps.to_vector(), mps_ref, rtol=1e-13), \
                "addition of two matrix product states must agree with vector representation"


def test_mps_sub():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # physical quantum numbers
            qsite = ptn.random_qnumbers(-2, 3, size=5, rng=rng)

            # create random matrix product states
            qbonds0 = [ptn.random_qnumbers(-2, 3, size=bi, rng=rng)
                       for bi in [1, 8, 15, 23, 18,  9, 1]]
            qbonds1 = [ptn.random_qnumbers(-2, 3, size=bi, rng=rng)
                       for bi in [1, 7, 23, 11, 17, 13, 1]]
            # leading and trailing (dummy) virtual bond quantum numbers must agree
            qbonds1[ 0] = qbonds0[ 0]
            qbonds1[-1] = qbonds0[-1]
            mps0 = ptn.MPS(qsite, qbonds0, fill="random", rng=rng)
            mps1 = ptn.MPS(qsite, qbonds1, fill="random", rng=rng)
            if backend == "torch" and torch.cuda.is_available():
                mps0.to_device("gpu")
                mps1.to_device("gpu")

            # MPS subtraction
            mps = mps0 - mps1

            # reference calculation
            mps_ref = mps0.to_vector() - mps1.to_vector()

            # compare
            assert np.allclose(mps.to_vector(), mps_ref, rtol=1e-12), \
                "subtraction of two matrix product states must agree with vector representation"


def test_mps_sub_singlesite():

    # separate test for a single site since implementation is a special case

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # physical quantum numbers
            qsite = ptn.random_qnumbers(-2, 3, size=7, rng=rng)

            # create random matrix product states for a single site
            # leading and trailing (dummy) virtual bond quantum numbers
            qbonds = [[-1], [-2]]
            mps0 = ptn.MPS(qsite, qbonds, fill="random", rng=rng)
            mps1 = ptn.MPS(qsite, qbonds, fill="random", rng=rng)
            if backend == "torch" and torch.cuda.is_available():
                mps0.to_device("gpu")
                mps1.to_device("gpu")

            # MPS subtraction
            mps = mps0 - mps1

            # reference calculation
            mps_ref = mps0.to_vector() - mps1.to_vector()

            # compare
            assert np.allclose(mps.to_vector(), mps_ref, rtol=1e-12), \
                "subtraction of two matrix product states must agree with vector representation"


def test_mps_vdot():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # physical quantum numbers
            qsite = ptn.random_qnumbers(-2, 3, size=5, rng=rng)
            # number of lattice sites
            nsites = 6

            for dtype in ("real", "complex"):
                # create random matrix product states
                psi = ptn.MPS.construct_random(
                    nsites, qsite, qnum_sector=2, max_vdim=13, dtype=dtype, rng=rng)
                chi = ptn.MPS.construct_random(
                    nsites, qsite, qnum_sector=2, max_vdim=17, dtype=dtype, rng=rng)
                # rescale to achieve norm of order 1
                for i in range(nsites):
                    psi.a[i] *= 7
                    chi.a[i] *= 7
                if backend == "torch" and torch.cuda.is_available():
                    psi.to_device("gpu")
                    chi.to_device("gpu")

                # calculate dot product <chi | psi>
                s = ptn.mps_vdot(chi, psi)

                # reference value
                s_ref = np.vdot(chi.to_vector(), psi.to_vector())

                # compare
                assert abs(s - s_ref) / max(abs(s_ref), 1e-12) < 1e-12, \
                    "dot product of two matrix product states must match reference value"
