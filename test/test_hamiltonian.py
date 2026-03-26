import numpy as np
from scipy import sparse
from scipy.stats import unitary_group
import pytenet as ptn


def test_ising():

    # Hamiltonian parameters
    J =  5.0/11
    h = -2.0/7
    g = 13.0/8
    # number of lattice sites
    nsites = 7

    # construct MPO
    h_mpo = ptn.ising_mpo(nsites, J, h, g)
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()

    # reference Hamiltonian
    h_ref = construct_ising_1d_hamiltonian(nsites, J, h, g)

    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference Hamiltonian must match"


def test_heisenberg_xxz():

    # Hamiltonian parameters
    J = 14.0/25
    D = 13.0/8
    h =  2.0/7
    # number of lattice sites
    nsites = 7

    # construct MPO
    h_mpo = ptn.heisenberg_xxz_mpo(nsites, J, D, h)
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()

    # reference Hamiltonian
    h_ref = construct_heisenberg_xxz_1d_hamiltonian(nsites, J, D, h)

    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference Hamiltonian must match"


def test_heisenberg_xxz_spin1():

    # Hamiltonian parameters
    J =  1.2
    D = -0.9
    h =  1.0/7
    # number of lattice sites
    nsites = 6

    # construct MPO
    h_mpo = ptn.heisenberg_xxz_spin1_mpo(nsites, J, D, h)
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()

    # reference Hamiltonian
    h_ref = construct_heisenberg_xxz_spin1_1d_hamiltonian(nsites, J, D, h)

    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference Hamiltonian must match"


def test_bose_hubbard():

    # physical dimension per site (maximal occupancy is d - 1)
    d = 4
    # number of lattice sites
    nsites = 5
    # Hamiltonian parameters
    t  = 0.7
    u  = 3.2
    mu = 1.3

    # construct MPO
    h_mpo = ptn.bose_hubbard_mpo(d, nsites, t, u, mu)
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()

    # reference Hamiltonian
    h_ref = construct_bose_hubbard_1d_hamiltonian(nsites, d, t, u, mu)

    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference Hamiltonian must match"


def test_fermi_hubbard():

    # number of lattice sites
    nsites = 5
    # Hamiltonian parameters
    t  = 1.2
    u  = 2.7
    mu = 0.3

    # construct MPO
    h_mpo = ptn.fermi_hubbard_mpo(nsites, t, u, mu)
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()

    # reference Hamiltonian
    h_ref = construct_fermi_hubbard_1d_hamiltonian(nsites, t, u, mu)

    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference Hamiltonian must match"


def test_linear_fermionic_operator():

    rng = np.random.default_rng()

    # number of lattice sites
    nsites = 6
    # coefficients
    coeff = ptn.crandn(nsites, rng)

    for ftype in ("c", "a"):
        # construct the MPO
        h_mpo = ptn.linear_fermionic_mpo(coeff, ftype)
        assert h_mpo.bond_dims == [1] + (nsites - 1)*[2] + [1], \
            "virtual bond dimensions must match theoretical prediction"
        # matrix representation, for comparison with reference
        h_mat = h_mpo.to_matrix()

        # reference operator
        clist, alist, _ = construct_fermi_operators(nsites)
        h_ref = sum(coeff[i] * (clist[i] if ftype == "c" else alist[i]) for i in range(nsites))

        # compare
        assert np.allclose(h_mat, h_ref.todense()), \
            "matrix representation of MPO and reference operator must match"


def test_linear_spin_fermionic_operator():

    rng = np.random.default_rng()

    # number of lattice sites
    nsites = 5
    # coefficients
    coeff = ptn.crandn(nsites, rng)

    for ftype in ("c", "a"):
        for sigma in (1, -1):
            # construct the MPO
            h_mpo = ptn.linear_spin_fermionic_mpo(coeff, ftype, sigma)
            assert h_mpo.bond_dims == [1] + (nsites - 1)*[2] + [1], \
                "virtual bond dimensions must match theoretical prediction"
            # matrix representation, for comparison with reference
            h_mat = h_mpo.to_matrix()

            # reference operator
            clist, alist, _ = construct_fermi_operators(2*nsites)
            offset = (0 if sigma == 1 else 1)
            h_ref = sum(coeff[i] * (clist[2*i+offset] if ftype == "c" else alist[2*i+offset])
                         for i in range(nsites))

            # compare
            assert np.allclose(h_mat, h_ref.todense()), \
                "matrix representation of MPO and reference operator must match"


def test_quadratic_fermionic_operator():

    rng = np.random.default_rng()

    # number of lattice sites
    nsites = 6
    # coefficients
    coeffc = ptn.crandn(nsites, rng)
    coeffa = ptn.crandn(nsites, rng)

    # construct the MPO
    h_mpo = ptn.quadratic_fermionic_mpo(coeffc, coeffa)
    assert h_mpo.bond_dims == [1] + (nsites - 1)*[4] + [1], \
        "virtual bond dimensions must match theoretical prediction"
    # matrix representation, for comparison with reference
    h_mat = h_mpo.to_matrix()
    # reference operator
    clist, alist, _ = construct_fermi_operators(nsites)
    h_ref = sum(coeffc[i] * clist[i] for i in range(nsites)) \
          @ sum(coeffa[i] * alist[i] for i in range(nsites))
    # compare
    assert np.allclose(h_mat, h_ref.todense()), \
        "matrix representation of MPO and reference operator must match"


def test_quadratic_spin_fermionic_operator():

    rng = np.random.default_rng()

    # number of lattice sites
    nsites = 5
    # coefficients
    coeffc = ptn.crandn(nsites, rng)
    coeffa = ptn.crandn(nsites, rng)

    for sigma in (1, -1):
        # construct the MPO
        h_mpo = ptn.quadratic_spin_fermionic_mpo(coeffc, coeffa, sigma)
        assert h_mpo.bond_dims == [1] + (nsites - 1)*[4] + [1], \
            "virtual bond dimensions must match theoretical prediction"
        # matrix representation, for comparison with reference
        h_mat = h_mpo.to_matrix()

        # reference operator
        clist, alist, _ = construct_fermi_operators(2*nsites)
        offset = (0 if sigma == 1 else 1)
        h_ref = sum(coeffc[i] * clist[2*i+offset] for i in range(nsites)) \
              @ sum(coeffa[i] * alist[2*i+offset] for i in range(nsites))

        # compare
        assert np.allclose(h_mat, h_ref.todense()), \
            "matrix representation of MPO and reference operator must match"


def test_molecular_hamiltonian_construction():

    rng = np.random.default_rng()

    # number of fermionic modes
    nsites = 7
    # Hamiltonian parameters
    tkin = ptn.crandn(2 * (nsites,), rng)
    vint = ptn.crandn(4 * (nsites,), rng)

    # reference Hamiltonian
    h_ref = construct_molecular_hamiltonian(tkin, vint)

    for opt in (True, False):
        h_mpo = ptn.molecular_hamiltonian_mpo(tkin, vint, opt)

        # theoretically predicted virtual bond dimensions
        b_theo = []
        for i in range(nsites + 1):
            nl = i
            nr = nsites - i
            n = min(nl, nr)
            # identity chains
            if opt:
                b1 = 2 if 1 < i < nsites - 1 else 1
            else:
                # slightly sub-optimal
                b1 = 2 if 1 <= i <= nsites - 1 else 1
            # a^{\dagger}_i and a_i chains, reaching (almost) from one boundary to the other
            if opt:
                b2 = 2 * min(nl**2 * (nl - 1) // 2, nr) + 2 * min(nl, nr**2 * (nr - 1) // 2)
            else:
                # slightly sub-optimal
                b2 = 2 * ((nl if i < nsites - 1 else 0) + (nr if i > 1 else 0))
            # a^{\dagger}_i a^{\dagger}_j (for i < j),
            # a_i a_j (for i > j)
            # and a^{\dagger}_i a_j chains,
            # extending from boundary to center
            b3 = 2 * n * (n - 1) // 2 + n**2
            b_theo.append(b1 + b2 + b3)
        assert h_mpo.bond_dims == b_theo

        # compare matrix representations
        assert np.allclose(h_mpo.to_matrix(), h_ref.todense()), \
            "matrix representation of MPO and reference Hamiltonian must match"


def test_molecular_hamiltonian_orbital_rotation():

    rng = np.random.default_rng()

    # number of fermionic modes
    nsites = 6
    # Hamiltonian coefficients
    tkin = ptn.crandn(2 * (nsites,), rng)
    vint = ptn.crandn(4 * (nsites,), rng)

    for i in range(nsites - 1):

        h_mpo = ptn.molecular_hamiltonian_mpo(tkin, vint, optimize=False)

        # random rotation matrix for two orbitals
        u2 = unitary_group.rvs(2, random_state=rng)

        # extend to overall orbital rotation matrix
        u = np.identity(nsites, dtype=u2.dtype)
        u[i:i+2, i:i+2] = u2

        # apply transposed single-orbital rotation matrix to Hamiltonian coefficients
        tkin_rotorb = np.einsum(u, (2, 0), u.conj(), (3, 1), tkin, (2, 3), (0, 1))
        vint_rotorb = np.einsum(u, (4, 0), u, (5, 1), u.conj(), (6, 2), u.conj(), (7, 3),
                                vint, (4, 5, 6, 7), (0, 1, 2, 3))

        # rotated reference MPO
        h_mpo_rotorb = ptn.molecular_hamiltonian_mpo(tkin_rotorb, vint_rotorb, optimize=False)

        # copy transformed MPO tensors at sites `i` and `i + 1`
        h_mpo.a[i    ] = np.copy(h_mpo_rotorb.a[i    ])
        h_mpo.a[i + 1] = np.copy(h_mpo_rotorb.a[i + 1])
        # apply left and right gauge transformations
        v_l, v_r = ptn.molecular_hamiltonian_orbital_gauge_transform(h_mpo, u2, i)
        h_mpo.a[i    ] = np.einsum(v_l, (2, 4), h_mpo.a[i    ], (0, 1, 4, 3), (0, 1, 2, 3))
        h_mpo.a[i + 1] = np.einsum(v_r, (3, 4), h_mpo.a[i + 1], (0, 1, 2, 4), (0, 1, 2, 3))

        # compare matrix representations
        assert np.allclose(h_mpo.to_matrix(), h_mpo_rotorb.to_matrix()), \
            "matrix representation of MPO after orbital rotation and reference Hamiltonian must match"


def test_spin_molecular_hamiltonian_construction():

    rng = np.random.default_rng()

    # number of spin-endowed lattice sites
    nsites = 4
    # Hamiltonian parameters
    tkin = ptn.crandn(2 * (nsites,), rng)
    vint = ptn.crandn(4 * (nsites,), rng)

    # reference Hamiltonian
    h_ref = construct_spin_molecular_hamiltonian(tkin, vint)

    for opt in (True, False):
        h_mpo = ptn.spin_molecular_hamiltonian_mpo(tkin, vint, opt)

        # compare matrix representations
        assert np.allclose(h_mpo.to_matrix(), h_ref.todense()), \
            "matrix representation of MPO and reference Hamiltonian must match"


def construct_ising_1d_hamiltonian(nsites: int, J: float, h: float, g: float):
    """
    Construct the Ising Hamiltonian `sum J Z Z + h Z + g X`
    on a one-dimensional lattice as sparse matrix.
    """
    # Pauli-X and Z matrices
    sigma_x = sparse.csr_matrix([[0., 1.], [1.,  0.]])
    sigma_z = sparse.csr_matrix([[1., 0.], [0., -1.]])
    # interaction terms and external field
    hint = sparse.kron(sigma_z, sigma_z)
    hamiltonian = \
        sum(J * sparse.kron(sparse.identity(2**j),
                sparse.kron(hint,
                            sparse.identity(2**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(h*sigma_z + g*sigma_x,
                        sparse.identity(2**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_heisenberg_xxz_1d_hamiltonian(nsites: int, J: float, D: float, h: float):
    """
    Construct the XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a one-dimensional lattice as sparse matrix.
    """
    # spin operators
    sup = np.array([[0.,  1.], [0.,  0. ]])
    sdn = np.array([[0.,  0.], [1.,  0. ]])
    sz  = np.array([[0.5, 0.], [0., -0.5]])
    # interaction terms and external field
    hint = J * 0.5 * (sparse.kron(sup, sdn) + sparse.kron(sdn, sup)) + D * sparse.kron(sz, sz)
    hamiltonian = \
        sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(hint,
                        sparse.identity(2**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(-h * sz,
                        sparse.identity(2**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_heisenberg_xxz_spin1_1d_hamiltonian(nsites: int, J: float, D: float, h: float):
    """
    Construct the spin-1 XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a one-dimensional lattice as sparse matrix.
    """
    # spin operators
    sq2 = np.sqrt(2.)
    sup = np.array([[0.,  sq2, 0.], [0.,  0.,  sq2], [0.,  0.,  0.]])
    sdn = np.array([[0.,  0.,  0.], [sq2, 0.,  0. ], [0.,  sq2, 0.]])
    sz  = np.array([[1.,  0.,  0.], [0.,  0.,  0. ], [0.,  0., -1.]])
    # interaction terms
    hint = J * 0.5 * (sparse.kron(sup, sdn) + sparse.kron(sdn, sup)) + D * sparse.kron(sz, sz)
    hamiltonian = \
        sum(sparse.kron(sparse.identity(3**j),
            sparse.kron(hint,
                        sparse.identity(3**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(3**j),
            sparse.kron(-h * sz,
                        sparse.identity(3**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_bose_hubbard_1d_hamiltonian(nsites: int, d: int, t: float, u: float, mu: float):
    """
    Construct the Bose-Hubbard Hamiltonian
    with nearest-neighbor hopping on a one-dimensional lattice as a sparse matrix.
    """
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    # kinetic hopping terms, interaction terms and external field
    tkin = -t * (sparse.kron(b_dag, b_ann) + sparse.kron(b_ann, b_dag))
    hint = 0.5 * u * (numop @ (numop - np.identity(d))) - mu * numop
    hamiltonian = \
        sum(sparse.kron(sparse.identity(d**j),
            sparse.kron(tkin,
                        sparse.identity(d**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(d**j),
            sparse.kron(hint,
                        sparse.identity(d**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_fermi_operators(nmodes: int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `nmodes` modes (or sites),
    based on Jordan-Wigner transformation.
    """
    id2 = sparse.identity(2)
    z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    u = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    for i in range(nmodes):
        c = sparse.identity(1)
        for j in range(nmodes):
            if j < i:
                c = sparse.kron(c, z)
            elif j == i:
                c = sparse.kron(c, u)
            else:
                c = sparse.kron(c, id2)
        c = sparse.csr_matrix(c)
        c.eliminate_zeros()
        clist.append(c)
    # corresponding annihilation operators
    alist = [sparse.csr_matrix(c.conj().T) for c in clist]
    # corresponding number operators
    nlist = []
    for i in range(nmodes):
        f = 1 << (nmodes - i - 1)
        data = [1. if (n & f == f) else 0. for n in range(2**nmodes)]
        nlist.append(sparse.dia_matrix((data, 0), 2*(2**nmodes,)))
    return clist, alist, nlist


def construct_fermi_hubbard_1d_hamiltonian(nsites: int, t: float, u: float, mu: float):
    """
    Construct the Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping on a one-dimensional lattice as sparse matrix.
    """
    clist, alist, nlist = construct_fermi_operators(2*nsites)
    # kinetic hopping terms and
    # interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn)
    hamiltonian = sum(-t * (clist[j] @ alist[j+2] + clist[j+2] @ alist[j])
                      for j in range(2*nsites - 2)) \
                + sum((u * (nlist[j]   - 0.5*sparse.identity(4**nsites)) \
                         @ (nlist[j+1] - 0.5*sparse.identity(4**nsites)) \
                       - mu * (nlist[j] + nlist[j+1]))
                      for j in range(0, 2*nsites, 2))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_molecular_hamiltonian(tkin, vint):
    """
    Construct the molecular Hamiltonian as a sparse matrix.
    """
    nmodes = tkin.shape[0]
    assert tkin.shape == (nmodes, nmodes)
    assert vint.shape == (nmodes, nmodes, nmodes, nmodes)

    clist, alist, _ = construct_fermi_operators(nmodes)

    # kinetic hopping and interaction terms
    hamiltonian = \
        sum(tkin[i, j] * (clist[i] @ alist[j])
            for i in range(nmodes)
            for j in range(nmodes)) + \
        sum(0.5 * vint[i, j, k, l] * (clist[i] @ clist[j] @ alist[l] @ alist[k])
            for i in range(nmodes)
            for j in range(nmodes)
            for k in range(nmodes)
            for l in range(nmodes))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_spin_molecular_hamiltonian(tkin, vint):
    """
    Construct a molecular Hamiltonian for a spin orbital basis as sparse matrix.
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)

    nsites = tkin.shape[0]
    assert tkin.shape == 2 * (nsites,)
    assert vint.shape == 4 * (nsites,)

    # enlarge the single- and two-particle electron overlap integral tensors
    # from an orbital basis without spin to a spin orbital basis

    # single-particle integrals
    tkin_spin = np.kron(tkin, np.identity(2))

    # two-particle integrals
    tmp = np.zeros((2*nsites, nsites, 2*nsites, nsites), dtype=vint.dtype)
    for i in range(nsites):
        for j in range(nsites):
            tmp[:, i, :, j] = np.kron(vint[:, i, :, j], np.identity(2))
    vint_spin = np.zeros((2*nsites, 2*nsites, 2*nsites, 2*nsites), dtype=vint.dtype)
    for i in range(2*nsites):
        for j in range(2*nsites):
            vint_spin[i, :, j, :] = np.kron(tmp[i, :, j, :], np.identity(2))

    return construct_molecular_hamiltonian(tkin_spin, vint_spin)
