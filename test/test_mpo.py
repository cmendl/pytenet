import numpy as np
import pytenet as ptn


def test_mpo_orthonormalize():

    rng = np.random.default_rng()

    # create random matrix product operator
    d = 4
    b = [1, 10, 13, 14, 7, 1]
    mpo0 = ptn.MPO(rng.integers(-2, 3, size=d),
                   [rng.integers(-2, 3, size=bi) for bi in b], fill="random", rng=rng)

    assert mpo0.bond_dims == b, "virtual bond dimensions do not match reference values"

    # density matrix on full Hilbert space
    rho = mpo0.to_matrix()

    # performing left-orthonormalization...
    c_left = mpo0.orthonormalize(mode="left")

    assert mpo0.bond_dims[1] <= d**2, \
        "virtual bond dimension can only increase by factor of `d^2` per site"

    for i in range(mpo0.nsites):
        assert ptn.is_qsparse(mpo0.a[i],
            (mpo0.qbonds[i], mpo0.qsite, -mpo0.qsite, -mpo0.qbonds[i+1])), \
            "sparsity pattern of MPO tensors must match quantum numbers"

    rho_left = mpo0.to_matrix()
    # density matrix should now be normalized
    assert np.allclose(np.linalg.norm(rho_left, "fro"), 1., rtol=1e-12), \
        "operator must be normalized after left-orthonormalization"

    # density matrices before and after left-normalization must match
    # (up to normalization factor)
    assert np.allclose(c_left*rho_left, rho, rtol=1e-10), \
        "density matrices before and after left-normalization must match"

    # check left-orthonormalization
    for i in range(mpo0.nsites):
        s = mpo0.a[i].shape
        assert s[1] == d and s[2] == d
        q = mpo0.a[i].reshape((s[0]*s[1]*s[2], s[3]))
        assert np.allclose(q.conj().T @ q, np.identity(s[3]), rtol=1e-12), \
            "MPO tensor is not an isometry after left-orthonormalization"

    # performing right-orthonormalization...
    c_right = mpo0.orthonormalize(mode="right")

    assert mpo0.bond_dims[-2] <= d**2, \
        "virtual bond dimension can only increase by a factor of `d^2` per site"

    for i in range(mpo0.nsites):
        assert ptn.is_qsparse(mpo0.a[i],
            (mpo0.qbonds[i], mpo0.qsite, -mpo0.qsite, -mpo0.qbonds[i+1])), \
            "sparsity pattern of MPO tensors must match quantum numbers"

    assert abs(abs(c_right) - 1.) <= 1e-12, \
        "normalization factor must have magnitude 1 due to previous left-orthonormalization"

    rho_right = mpo0.to_matrix()
    # density matrices must match
    assert np.linalg.norm(rho_left - c_right*rho_right) < 1e-10, \
        "density matrices after left- and right-orthonormalization must match"

    # check right-orthonormalization
    for i in range(mpo0.nsites):
        s = mpo0.a[i].shape
        assert s[1] == d and s[2] == d
        q = mpo0.a[i].transpose((3, 1, 2, 0)).reshape((s[1]*s[2]*s[3], s[0]))
        assert np.allclose(q.conj().T @ q, np.identity(s[0]), rtol=1e-12), \
            "MPO tensor is not an isometry after right-orthonormalization"


def test_mpo_identity():

    rng = np.random.default_rng()

    # dimensions
    d = 3
    nsites = 6

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=d)

    # construct MPO representation of identity
    idop = ptn.MPO.identity(qsite, nsites)

    assert np.array_equal(idop.to_matrix(), np.identity(d**nsites)), \
        "MPO must represent the identity map"


def test_mpo_from_opgraph():

    rng = np.random.default_rng()

    # lattice size
    nsites = 5

    # physical quantum numbers
    qsite = np.array([-1, 0, 2, 0])

    # generate a symbolic operator graph
    graph = ptn.opgraph.OpGraph(
        [ptn.opgraph.OpGraphNode( 0, [      ], [ 2,  5],  0),
         ptn.opgraph.OpGraphNode( 1, [ 2    ], [ 1    ],  1),
         ptn.opgraph.OpGraphNode( 2, [ 5    ], [ 4    ],  0),
         ptn.opgraph.OpGraphNode( 3, [ 1,  4], [ 7,  0], -1),
         ptn.opgraph.OpGraphNode( 4, [ 7    ], [ 3    ],  0),
         ptn.opgraph.OpGraphNode( 5, [ 0    ], [ 6,  8],  1),
         ptn.opgraph.OpGraphNode( 6, [ 3,  6], [10    ], -1),
         ptn.opgraph.OpGraphNode( 7, [ 8    ], [ 9    ],  0),
         ptn.opgraph.OpGraphNode( 8, [10,  9], [      ],  1)],
        [ptn.opgraph.OpGraphEdge( 2, [ 0,  1], [( -2, -0.6)]),
         ptn.opgraph.OpGraphEdge( 5, [ 0,  2], [( -5,  1.3)]),
         ptn.opgraph.OpGraphEdge( 1, [ 1,  3], [( -1,  0.4)]),
         ptn.opgraph.OpGraphEdge( 4, [ 2,  3], [( -4, -1.2)]),
         ptn.opgraph.OpGraphEdge( 7, [ 3,  4], [( -7,  0.7)]),
         ptn.opgraph.OpGraphEdge( 0, [ 3,  5], [(  0,  0.5)]),
         ptn.opgraph.OpGraphEdge( 3, [ 4,  6], [( -3, -1.6)]),
         ptn.opgraph.OpGraphEdge( 6, [ 5,  6], [( -6,  0.8)]),
         ptn.opgraph.OpGraphEdge( 8, [ 5,  7], [( -8, -0.3)]),
         ptn.opgraph.OpGraphEdge(10, [ 6,  8], [(-10,  0.9)]),
         ptn.opgraph.OpGraphEdge( 9, [ 7,  8], [( -9, -0.2)])],
        [0, 8])
    assert graph.is_consistent()

    # random local operators
    opmap = { opid: ptn.crandn(2 * (len(qsite),), rng) for opid in range(-10, 1) }
    # enforce sparsity pattern according to quantum numbers
    for edge in graph.edges.values():
        qbonds_loc = [graph.nodes[nid].qnum for nid in edge.nids]
        mask = ptn.qnumber_outer_sum([[qbonds_loc[0]], qsite, -qsite, [-qbonds_loc[1]]])[0, :, :, 0]
        for opid, _ in edge.opics:
            opmap[opid] = np.where(mask == 0, opmap[opid], 0)

    # convert graph to an MPO
    mpo = ptn.MPO.from_opgraph(qsite, graph, opmap)
    assert mpo.bond_dims == [1, 2, 1, 2, 2, 1]

    # compare matrix representations
    assert np.allclose(mpo.to_matrix(), graph.to_matrix(opmap))

    # reference operator representation
    opstringids = [
        [(-2, -0.6), (-1,  0.4), (-7,  0.7), (-3, -1.6), (-10,  0.9)],
        [(-5,  1.3), (-4, -1.2), (-7,  0.7), (-3, -1.6), (-10,  0.9)],
        [(-2, -0.6), (-1,  0.4), ( 0,  0.5), (-6,  0.8), (-10,  0.9)],
        [(-2, -0.6), (-1,  0.4), ( 0,  0.5), (-8, -0.3), ( -9, -0.2)],
        [(-5,  1.3), (-4, -1.2), ( 0,  0.5), (-6,  0.8), (-10,  0.9)],
        [(-5,  1.3), (-4, -1.2), ( 0,  0.5), (-8, -0.3), ( -9, -0.2)]]
    opref = np.zeros(2 * (len(qsite)**nsites,), dtype=complex)
    for opstringid in opstringids:
        op = np.identity(1)
        for opid, coeff in opstringid:
            op = np.kron(op, coeff * opmap[opid])
        opref += op
    # compare
    assert np.allclose(mpo.to_matrix(), opref)


def test_mpo_to_matrix():

    rng = np.random.default_rng()

    # create a random matrix product operator
    qsite = rng.integers(-2, 3, size=3)
    qbonds = [rng.integers(-2, 3, size=bi) for bi in [1, 11, 16, 23, 19, 9, 1]]
    mpo = ptn.MPO(qsite, qbonds, fill="random", rng=rng)
    # rescale to reach norm of order 1
    for i in range(mpo.nsites):
        mpo.a[i] *= 5
    # matrix representations
    mat_dense  = mpo.to_matrix(sparse_format=False)
    mat_sparse = mpo.to_matrix(sparse_format=True)
    # compare
    assert np.allclose(mat_dense, mat_sparse.toarray(), rtol=1e-13), \
        "dense and sparse matrix representations of an MPO must agree"


def test_mpo_add():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=3)

    # create random matrix product operators
    qbonds0 = [rng.integers(-2, 3, size=bi) for bi in [1, 11, 15, 23, 18, 9, 1]]
    qbonds1 = [rng.integers(-2, 3, size=bi) for bi in [1, 7, 23, 11, 17, 13, 1]]
    # leading and trailing (dummy) virtual bond quantum numbers must agree
    qbonds1[ 0] = qbonds0[ 0].copy()
    qbonds1[-1] = qbonds0[-1].copy()
    op0 = ptn.MPO(qsite, qbonds0, fill="random", rng=rng)
    op1 = ptn.MPO(qsite, qbonds1, fill="random", rng=rng)

    # MPO addition
    op = op0 + op1

    # reference calculation
    op_ref = op0.to_matrix() + op1.to_matrix()

    # compare
    assert np.allclose(op.to_matrix(), op_ref, rtol=1e-12), \
        "addition of two MPOs must agree with matrix representation"


def test_mpo_add_singlesite():

    # separate test for a single site since implementation is a special case

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=5)

    # create random matrix product operators acting on a single site
    # leading and trailing (dummy) virtual bond quantum numbers
    qbonds = [np.array([-1]), np.array([-2])]
    op0 = ptn.MPO(qsite, qbonds, fill="random", rng=rng)
    op1 = ptn.MPO(qsite, qbonds, fill="random", rng=rng)

    # MPO addition
    op = op0 + op1

    # reference calculation
    op_ref = op0.to_matrix() + op1.to_matrix()

    # compare
    assert np.allclose(op.to_matrix(), op_ref, rtol=1e-12), \
        "addition of two MPOs must agree with matrix representation"


def test_mpo_sub():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=3)

    # create random matrix product operators
    qbonds0 = [rng.integers(-2, 3, size=bi) for bi in [1, 11, 15, 23, 18, 9, 1]]
    qbonds1 = [rng.integers(-2, 3, size=bi) for bi in [1, 7, 23, 11, 17, 13, 1]]
    # leading and trailing (dummy) virtual bond quantum numbers must agree
    qbonds1[ 0] = qbonds0[ 0].copy()
    qbonds1[-1] = qbonds0[-1].copy()
    op0 = ptn.MPO(qsite, qbonds0, fill="random", rng=rng)
    op1 = ptn.MPO(qsite, qbonds1, fill="random", rng=rng)

    # MPO subtraction
    op = op0 - op1

    # reference calculation
    op_ref = op0.to_matrix() - op1.to_matrix()

    # compare
    assert np.allclose(op.to_matrix(), op_ref, rtol=1e-12), \
        "subtraction of two MPOs must agree with matrix representation"


def test_mpo_sub_singlesite():

    # separate test for a single site since implementation is a special case

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=5)

    # create random matrix product operators acting on a single site
    # leading and trailing (dummy) virtual bond quantum numbers
    qbonds = [np.array([-1]), np.array([-2])]
    op0 = ptn.MPO(qsite, qbonds, fill="random", rng=rng)
    op1 = ptn.MPO(qsite, qbonds, fill="random", rng=rng)

    # MPO subtraction
    op = op0 - op1

    # reference calculation
    op_ref = op0.to_matrix() - op1.to_matrix()

    # compare
    assert np.allclose(op.to_matrix(), op_ref, rtol=1e-12), \
        "subtraction of two MPOs must agree with matrix representation"


def test_mpo_multiply():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=3)

    # create random matrix product operators
    op0 = ptn.MPO(qsite, [rng.integers(-2, 3, size=bi)
                          for bi in [1, 10, 13, 24, 17, 9, 1]],
                  fill="random", rng=rng)
    op1 = ptn.MPO(qsite, [rng.integers(-2, 3, size=bi)
                          for bi in [1, 8, 17, 11, 23, 13, 1]],
                  fill="random", rng=rng)

    # MPO multiplication (composition)
    op = op0 @ op1

    # reference calculation
    op_ref = op0.to_matrix() @ op1.to_matrix()

    # compare
    assert np.allclose(op.to_matrix(), op_ref, rtol=1e-12), \
        "composition of two MPOs must agree with matrix representation"
