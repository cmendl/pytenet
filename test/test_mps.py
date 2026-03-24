import numpy as np
import pytenet as ptn


def test_mps_orthonormalization():

    rng = np.random.default_rng()

    # create random matrix product state
    d = 7
    D = [1, 4, 15, 13, 7, 1]
    mps0 = ptn.MPS(rng.integers(-2, 3, size=d),
                   [rng.integers(-2, 3, size=Di) for Di in D],
                   fill="random", rng=rng)

    assert mps0.bond_dims == D, "virtual bond dimensions do not match reference values"

    # wavefunction on full Hilbert space
    psi = mps0.as_vector()

    # performing left-orthonormalization...
    cL = mps0.orthonormalize(mode="left")

    assert mps0.bond_dims[1] <= d, \
        "virtual bond dimension can only increase by a factor of \"d\" per site"

    for i in range(mps0.nsites):
        assert ptn.is_qsparse(mps0.A[i], [mps0.qd, mps0.qD[i], -mps0.qD[i+1]]), \
            "sparsity pattern of MPS tensors must match quantum numbers"

    psiL = mps0.as_vector()
    # wavefunction should now be normalized
    assert abs(np.linalg.norm(psiL) - 1.) < 1e-12, \
        "MPS wavefunction must be normalized"

    # wavefunctions before and after left-normalization must match
    # (up to normalization factor)
    assert np.linalg.norm(cL*psiL - psi) < 1e-10, \
        "wavefunctions before and after left-normalization must match"

    # check left-orthonormalization
    for i in range(mps0.nsites):
        s = mps0.A[i].shape
        assert s[0] == d
        Q = mps0.A[i].reshape((s[0]*s[1], s[2]))
        assert np.allclose(Q.conj().T @ Q, np.identity(s[2]), rtol=1e-12), \
            "MPS tensor is not left-orthonormalized"

    # performing right-orthonormalization...
    cR = mps0.orthonormalize(mode="right")

    assert mps0.bond_dims[-2] <= d, \
        "virtual bond dimension can only increase by a factor of \"d\" per site"

    for i in range(mps0.nsites):
        assert ptn.is_qsparse(mps0.A[i], [mps0.qd, mps0.qD[i], -mps0.qD[i+1]]), \
            "sparsity pattern of MPS tensors must match quantum numbers"

    assert abs(abs(cR) - 1.) <= 1e-12, \
        "normalization factor must have magnitude 1 due to previous left-orthonormalization"

    psiR = mps0.as_vector()
    # wavefunctions must match
    assert np.allclose(psiL, cR*psiR, rtol=1e-10), \
        "wavefunctions after left- and right-orthonormalization must match"

    # check right-orthonormalization
    for i in range(mps0.nsites):
        s = mps0.A[i].shape
        assert s[0] == d
        Q = mps0.A[i].transpose((0, 2, 1)).reshape((s[0]*s[2], s[1]))
        assert np.allclose(Q.conj().T @ Q, np.identity(s[1]), rtol=1e-12), \
            "MPS tensor is not right-orthonormalized"


def test_mps_compression():

    rng = np.random.default_rng()

    # physical quantum numbers
    qd = [-1, 1, 0]

    for tol in (0., 1e-4):
        for mode, direction in [("svd", "left"), ("svd", "right"), ("density", "any")]:
            # create random matrix product state
            D = [1, 30, 83, 102, 75, 23, 1]
            psi = ptn.MPS(qd, [rng.integers(-1, 2, size=Di) for Di in D], fill="random", rng=rng)
            for i in range(psi.nsites):
                # imitate small entanglement by multiplying bonds with small scaling factors
                s = np.exp(-30*(rng.uniform(size=psi.bond_dims[i + 1])))
                s /= np.linalg.norm(s)
                psi.A[i] = psi.A[i] * s
                # rescale to achieve norm of order 1
                psi.A[i] *= 5 / np.linalg.norm(psi.A[i])

            psi_ref = psi.as_vector()
            nrm_ref = np.linalg.norm(psi_ref)

            nrm, scale = psi.compress(tol, mode=mode, direction=direction)

            # norm of input state vector
            assert abs(nrm - nrm_ref) < 1e-13

            assert abs(scale - 1) < (1e-13 if tol == 0 else 1e-2)
            # must be normalized after compression
            assert abs(ptn.norm(psi) - 1) < 1e-13

            # compare with original state vector
            ctol = (1e-13 if tol == 0 else 0.05)
            assert np.allclose(nrm*psi.as_vector(), psi_ref, atol=ctol, rtol=ctol)


def test_mps_split_tensor():

    rng = np.random.default_rng()

    # physical dimensions
    d0, d1 = 3, 5
    # outer virtual bond dimensions
    D0, D2 = 14, 17

    Apair = ptn.crandn((d0*d1, D0, D2), rng) / np.sqrt(d0*d1*D0*D2)

    # fictitious quantum numbers
    qd0 = rng.integers(-2, 3, size=d0)
    qd1 = rng.integers(-2, 3, size=d1)
    qD = [rng.integers(-2, 3, size=D0), rng.integers(-2, 3, size=D2)]

    # enforce block sparsity structure dictated by quantum numbers
    ptn.enforce_qsparsity(Apair, [ptn.qnumber_flatten([qd0, qd1]), qD[0], -qD[1]])

    for svd_distr in ["left", "right", "sqrt"]:
        A0, A1, qbond = ptn.split_mps_tensor(Apair, qd0, qd1, qD, svd_distr=svd_distr, tol=0)

        assert ptn.is_qsparse(A0, [qd0, qD[0], -qbond]), \
            "sparsity pattern of A0 tensors must match quantum numbers"
        assert ptn.is_qsparse(A1, [qd1, qbond, -qD[1]]), \
            "sparsity pattern of A1 tensors must match quantum numbers"

        # merged tensor must agree with the original tensor
        Amrg = ptn.merge_mps_tensor_pair(A0, A1)
        assert np.allclose(Amrg, Apair, rtol=1e-13), \
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
    assert np.allclose(mps.as_vector(), v, rtol=1e-13), \
        "MPS constructed from a vector must match original vector"


def test_mps_add():

    rng = np.random.default_rng()

    # physical quantum numbers
    qd = rng.integers(-2, 3, size=5)

    # create random matrix product states
    qD0 = [rng.integers(-2, 3, size=Di) for Di in [1, 8, 15, 23, 18,  9, 1]]
    qD1 = [rng.integers(-2, 3, size=Di) for Di in [1, 7, 23, 11, 17, 13, 1]]
    # leading and trailing (dummy) virtual bond quantum numbers must agree
    qD1[ 0] = qD0[ 0].copy()
    qD1[-1] = qD0[-1].copy()
    mps0 = ptn.MPS(qd, qD0, fill="random", rng=rng)
    mps1 = ptn.MPS(qd, qD1, fill="random", rng=rng)

    # MPS addition
    mps = mps0 + mps1

    # reference calculation
    mps_ref = mps0.as_vector() + mps1.as_vector()

    # compare
    assert np.allclose(mps.as_vector(), mps_ref, rtol=1e-13), \
        "addition of two matrix product states must agree with vector representation"


def test_mps_add_singlesite():

    # separate test for a single site since implementation is a special case

    rng = np.random.default_rng()

    # physical quantum numbers
    qd = rng.integers(-2, 3, size=7)

    # create random matrix product states for a single site
    # leading and trailing (dummy) virtual bond quantum numbers
    qD = [np.array([-1]), np.array([-2])]
    mps0 = ptn.MPS(qd, qD, fill="random", rng=rng)
    mps1 = ptn.MPS(qd, qD, fill="random", rng=rng)

    # MPS addition
    mps = mps0 + mps1

    # reference calculation
    mps_ref = mps0.as_vector() + mps1.as_vector()

    # compare
    assert np.allclose(mps.as_vector(), mps_ref, rtol=1e-13), \
        "addition of two matrix product states must agree with vector representation"


def test_mps_sub():

    rng = np.random.default_rng()

    # physical quantum numbers
    qd = rng.integers(-2, 3, size=5)

    # create random matrix product states
    qD0 = [rng.integers(-2, 3, size=Di) for Di in [1, 8, 15, 23, 18,  9, 1]]
    qD1 = [rng.integers(-2, 3, size=Di) for Di in [1, 7, 23, 11, 17, 13, 1]]
    # leading and trailing (dummy) virtual bond quantum numbers must agree
    qD1[ 0] = qD0[ 0].copy()
    qD1[-1] = qD0[-1].copy()
    mps0 = ptn.MPS(qd, qD0, fill="random", rng=rng)
    mps1 = ptn.MPS(qd, qD1, fill="random", rng=rng)

    # MPS subtraction
    mps = mps0 - mps1

    # reference calculation
    mps_ref = mps0.as_vector() - mps1.as_vector()

    # compare
    assert np.allclose(mps.as_vector(), mps_ref, rtol=1e-12), \
        "subtraction of two matrix product states must agree with vector representation"


def test_mps_sub_singlesite():

    # separate test for a single site since implementation is a special case

    rng = np.random.default_rng()

    # physical quantum numbers
    qd = rng.integers(-2, 3, size=7)

    # create random matrix product states for a single site
    # leading and trailing (dummy) virtual bond quantum numbers
    qD = [np.array([-1]), np.array([-2])]
    mps0 = ptn.MPS(qd, qD, fill="random", rng=rng)
    mps1 = ptn.MPS(qd, qD, fill="random", rng=rng)

    # MPS subtraction
    mps = mps0 - mps1

    # reference calculation
    mps_ref = mps0.as_vector() - mps1.as_vector()

    # compare
    assert np.allclose(mps.as_vector(), mps_ref, rtol=1e-12), \
        "subtraction of two matrix product states must agree with vector representation"
