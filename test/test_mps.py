import numpy as np
import pytenet as ptn


def test_mps_orthonormalize():

    rng = np.random.default_rng()

    # create random matrix product state
    d = 7
    b = [1, 4, 15, 13, 7, 1]
    mps0 = ptn.MPS(rng.integers(-2, 3, size=d),
                   [rng.integers(-2, 3, size=bi) for bi in b],
                   fill="random", rng=rng)

    assert mps0.bond_dims == b, "virtual bond dimensions do not match reference values"

    # wavefunction on full Hilbert space
    psi = mps0.to_vector()

    # performing left-orthonormalization...
    c_left = mps0.orthonormalize(mode="left")

    assert mps0.bond_dims[1] <= d, \
        "virtual bond dimension can only increase by a factor of `d` per site"

    for i in range(mps0.nsites):
        assert ptn.is_qsparse(mps0.a[i], (mps0.qbonds[i], mps0.qsite, -mps0.qbonds[i+1])), \
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
        assert np.allclose(q.conj().T @ q, np.identity(s[2]), rtol=1e-12), \
            "MPS tensor is not left-orthonormalized"

    # performing right-orthonormalization...
    c_right = mps0.orthonormalize(mode="right")

    assert mps0.bond_dims[-2] <= d, \
        "virtual bond dimension can only increase by a factor of `d` per site"

    for i in range(mps0.nsites):
        assert ptn.is_qsparse(mps0.a[i], (mps0.qbonds[i], mps0.qsite, -mps0.qbonds[i+1])), \
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
        q = mps0.a[i].transpose((2, 1, 0)).reshape((s[1]*s[2], s[0]))
        assert np.allclose(q.conj().T @ q, np.identity(s[0]), rtol=1e-12), \
            "MPS tensor is not right-orthonormalized"


def test_mps_compress():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = [-1, 1, 0]

    for tol in (0., 1e-4):
        for mode, direction in [("svd", "left"), ("svd", "right"), ("density", "any")]:
            # create random matrix product state
            b = [1, 30, 83, 102, 75, 23, 1]
            psi = ptn.MPS(qsite, [rng.integers(-1, 2, size=bi) for bi in b], fill="random", rng=rng)
            for i in range(psi.nsites):
                # imitate small entanglement by multiplying bonds with small scaling factors
                s = np.exp(-30*(rng.uniform(size=psi.bond_dims[i + 1])))
                s /= np.linalg.norm(s)
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
            ctol = (1e-13 if tol == 0 else 0.08)
            assert np.allclose(nrm*psi.to_vector(), psi_ref, atol=ctol, rtol=ctol)


def test_mps_split_tensor_svd():

    rng = np.random.default_rng()

    # physical dimensions
    d0, d1 = 3, 5
    # outer virtual bond dimensions
    b0, b2 = 14, 17

    a_pair = ptn.crandn((b0, d0*d1, b2), rng) / np.sqrt(b0*d0*d1*b2)

    # fictitious quantum numbers
    qsite0 = rng.integers(-2, 3, size=d0)
    qsite1 = rng.integers(-2, 3, size=d1)
    qbonds = [rng.integers(-2, 3, size=b0), rng.integers(-2, 3, size=b2)]

    # enforce block sparsity structure dictated by quantum numbers
    ptn.enforce_qsparsity(a_pair, (qbonds[0], ptn.qnumber_flatten((qsite0, qsite1)), -qbonds[1]))

    for svd_distr in ["left", "right", "sqrt"]:
        a0, a1, qbond = ptn.mps_split_tensor_svd(
            a_pair, qsite0, qsite1, qbonds, svd_distr=svd_distr, tol=0)

        assert ptn.is_qsparse(a0, (qbonds[0], qsite0, -qbond)), \
            "sparsity pattern of `a0` tensors must match quantum numbers"
        assert ptn.is_qsparse(a1, (qbond, qsite1, -qbonds[1])), \
            "sparsity pattern of `a1` tensors must match quantum numbers"

        # merged tensor must agree with the original tensor
        a_mrg = ptn.mps_merge_tensor_pair(a0, a1)
        assert np.allclose(a_mrg, a_pair, rtol=1e-13), \
            "splitting and subsequent merging must give the same tensor"


def test_mps_from_vector():

    rng = np.random.default_rng()

    # physical local dimension
    d = 3
    # number of lattice sites
    nsites = 7
    # random vector
    v = ptn.crandn(d**nsites, rng)
    mps = ptn.MPS.from_vector(d, nsites, v)
    assert np.allclose(mps.to_vector(), v, rtol=1e-13), \
        "MPS constructed from a vector must match original vector"


def test_mps_add():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=5)

    # create random matrix product states
    qbonds0 = [rng.integers(-2, 3, size=bi) for bi in [1, 8, 15, 23, 18,  9, 1]]
    qbonds1 = [rng.integers(-2, 3, size=bi) for bi in [1, 7, 23, 11, 17, 13, 1]]
    # leading and trailing (dummy) virtual bond quantum numbers must agree
    qbonds1[ 0] = qbonds0[ 0].copy()
    qbonds1[-1] = qbonds0[-1].copy()
    mps0 = ptn.MPS(qsite, qbonds0, fill="random", rng=rng)
    mps1 = ptn.MPS(qsite, qbonds1, fill="random", rng=rng)

    # MPS addition
    mps = mps0 + mps1

    # reference calculation
    mps_ref = mps0.to_vector() + mps1.to_vector()

    # compare
    assert np.allclose(mps.to_vector(), mps_ref, rtol=1e-13), \
        "addition of two matrix product states must agree with vector representation"


def test_mps_add_singlesite():

    # separate test for a single site since implementation is a special case

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=7)

    # create random matrix product states for a single site
    # leading and trailing (dummy) virtual bond quantum numbers
    qbonds = [np.array([-1]), np.array([-2])]
    mps0 = ptn.MPS(qsite, qbonds, fill="random", rng=rng)
    mps1 = ptn.MPS(qsite, qbonds, fill="random", rng=rng)

    # MPS addition
    mps = mps0 + mps1

    # reference calculation
    mps_ref = mps0.to_vector() + mps1.to_vector()

    # compare
    assert np.allclose(mps.to_vector(), mps_ref, rtol=1e-13), \
        "addition of two matrix product states must agree with vector representation"


def test_mps_sub():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=5)

    # create random matrix product states
    qbonds0 = [rng.integers(-2, 3, size=bi) for bi in [1, 8, 15, 23, 18,  9, 1]]
    qbonds1 = [rng.integers(-2, 3, size=bi) for bi in [1, 7, 23, 11, 17, 13, 1]]
    # leading and trailing (dummy) virtual bond quantum numbers must agree
    qbonds1[ 0] = qbonds0[ 0].copy()
    qbonds1[-1] = qbonds0[-1].copy()
    mps0 = ptn.MPS(qsite, qbonds0, fill="random", rng=rng)
    mps1 = ptn.MPS(qsite, qbonds1, fill="random", rng=rng)

    # MPS subtraction
    mps = mps0 - mps1

    # reference calculation
    mps_ref = mps0.to_vector() - mps1.to_vector()

    # compare
    assert np.allclose(mps.to_vector(), mps_ref, rtol=1e-12), \
        "subtraction of two matrix product states must agree with vector representation"


def test_mps_sub_singlesite():

    # separate test for a single site since implementation is a special case

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=7)

    # create random matrix product states for a single site
    # leading and trailing (dummy) virtual bond quantum numbers
    qbonds = [np.array([-1]), np.array([-2])]
    mps0 = ptn.MPS(qsite, qbonds, fill="random", rng=rng)
    mps1 = ptn.MPS(qsite, qbonds, fill="random", rng=rng)

    # MPS subtraction
    mps = mps0 - mps1

    # reference calculation
    mps_ref = mps0.to_vector() - mps1.to_vector()

    # compare
    assert np.allclose(mps.to_vector(), mps_ref, rtol=1e-12), \
        "subtraction of two matrix product states must agree with vector representation"


def test_mps_vdot():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=5)
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

        # calculate dot product <chi | psi>
        s = ptn.mps_vdot(chi, psi)

        # reference value
        s_ref = np.vdot(chi.to_vector(), psi.to_vector())

        # compare
        assert abs(s - s_ref) / max(abs(s_ref), 1e-12) < 1e-12, \
            "dot product of two matrix product states must match reference value"
