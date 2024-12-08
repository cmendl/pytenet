import unittest
import numpy as np
from scipy import sparse
from scipy.stats import unitary_group
import pytenet as ptn


class TestHamiltonian(unittest.TestCase):

    def test_ising(self):

        # Hamiltonian parameters
        J =  5.0/11
        h = -2.0/7
        g = 13.0/8
        # number of lattice sites
        L = 7

        # construct MPO
        mpoH = ptn.ising_mpo(L, J, h, g)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_ising_hamiltonian(L, J, h, g)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_heisenberg_xxz(self):

        # Hamiltonian parameters
        J = 14.0/25
        D = 13.0/8
        h =  2.0/7
        # number of lattice sites
        L = 7

        # construct MPO
        mpoH = ptn.heisenberg_xxz_mpo(L, J, D, h)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_heisenberg_xxz_hamiltonian(L, J, D, h)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_heisenberg_xxz_spin1(self):

        # Hamiltonian parameters
        J =  1.2
        D = -0.9
        h =  1.0/7
        # number of lattice sites
        L = 6

        # construct MPO
        mpoH = ptn.heisenberg_xxz_spin1_mpo(L, J, D, h)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_heisenberg_xxz_spin1_hamiltonian(L, J, D, h)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_bose_hubbard(self):

        # physical dimension per site (maximal occupancy is d - 1)
        d = 4
        # number of lattice sites
        L = 5
        # Hamiltonian parameters
        t  = 0.7
        U  = 3.2
        mu = 1.3

        # construct MPO
        mpoH = ptn.bose_hubbard_mpo(d, L, t, U, mu)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_bose_hubbard_hamiltonian(d, L, t, U, mu)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_fermi_hubbard(self):

        # number of lattice sites
        L = 5
        # Hamiltonian parameters
        t  = 1.2
        U  = 2.7
        mu = 0.3

        # construct MPO
        mpoH = ptn.fermi_hubbard_mpo(L, t, U, mu)
        # matrix representation, for comparison with reference
        H = mpoH.as_matrix()

        # reference Hamiltonian
        Href = construct_fermi_hubbard_hamiltonian(L, t, U, mu)

        # compare
        self.assertTrue(np.allclose(H, Href.todense()),
            msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_linear_fermionic(self):

        rng = np.random.default_rng()

        # number of lattice sites
        L = 6
        # coefficients
        coeff = ptn.crandn(L, rng)

        for ftype in ('c', 'a'):
            # construct the MPO
            mpo = ptn.linear_fermionic_mpo(coeff, ftype)
            self.assertEqual(mpo.bond_dims, [1] + (L - 1)*[2] + [1],
                msg='virtual bond dimensions must match theoretical prediction')
            # matrix representation, for comparison with reference
            op = mpo.as_matrix()
            # reference operator
            clist, alist = generate_fermi_operators(L)
            op_ref = sum(coeff[i] * (clist[i] if ftype == 'c' else alist[i]) for i in range(L))
            # compare
            self.assertTrue(np.allclose(op, op_ref.todense()),
                msg='matrix representation of MPO and reference operator must match')


    def test_quadratic_fermionic(self):

        rng = np.random.default_rng()

        # number of lattice sites
        L = 6
        # coefficients
        coeffc = ptn.crandn(L, rng)
        coeffa = ptn.crandn(L, rng)

        # construct the MPO
        mpo = ptn.quadratic_fermionic_mpo(coeffc, coeffa)
        self.assertEqual(mpo.bond_dims, [1] + (L - 1)*[4] + [1],
            msg='virtual bond dimensions must match theoretical prediction')
        # matrix representation, for comparison with reference
        op = mpo.as_matrix()
        # reference operator
        clist, alist = generate_fermi_operators(L)
        op_ref = sum(coeffc[i] * clist[i] for i in range(L)) @ sum(coeffa[i] * alist[i] for i in range(L))
        # compare
        self.assertTrue(np.allclose(op, op_ref.todense()),
            msg='matrix representation of MPO and reference operator must match')


    def test_molecular_hamiltonian_construction(self):

        rng = np.random.default_rng()

        # number of fermionic modes
        L = 7
        # Hamiltonian parameters
        tkin = ptn.crandn(2 * (L,), rng)
        vint = ptn.crandn(4 * (L,), rng)

        # reference Hamiltonian
        Href = construct_molecular_hamiltonian(tkin, vint)

        for opt in (True, False):
            mpoH = ptn.molecular_hamiltonian_mpo(tkin, vint, opt)

            # theoretically predicted virtual bond dimensions
            D_theo = []
            for i in range(L + 1):
                nl = i
                nr = L - i
                n = min(nl, nr)
                # identity chains
                if opt:
                    D1 = 2 if 1 < i < L - 1 else 1
                else:
                    # slightly sub-optimal
                    D1 = 2 if 1 <= i <= L - 1 else 1
                # a^{\dagger}_i and a_i chains, reaching (almost) from one boundary to the other
                if opt:
                    D2 = 2 * min(nl**2 * (nl - 1) // 2, nr) + 2 * min(nl, nr**2 * (nr - 1) // 2)
                else:
                    # slightly sub-optimal
                    D2 = 2 * ((nl if i < L - 1 else 0) + (nr if i > 1 else 0))
                # a^{\dagger}_i a^{\dagger}_j (for i < j), a_i a_j (for i > j) and a^{\dagger}_i a_j chains, extending from boundary to center
                D3 = 2 * n * (n - 1) // 2 + n**2
                D_theo.append(D1 + D2 + D3)
            self.assertEqual(mpoH.bond_dims, D_theo)

            # compare matrix representations
            self.assertTrue(np.allclose(mpoH.as_matrix(), Href.todense()),
                msg='matrix representation of MPO and reference Hamiltonian must match')


    def test_molecular_hamiltonian_orbital_rotation(self):

        rng = np.random.default_rng()

        # number of fermionic modes
        L = 6
        # Hamiltonian coefficients
        tkin = ptn.crandn(2 * (L,), rng)
        vint = ptn.crandn(4 * (L,), rng)

        for i in range(L - 1):

            mpoH = ptn.molecular_hamiltonian_mpo(tkin, vint, optimize=False)

            # random rotation matrix for two orbitals
            u2 = unitary_group.rvs(2, random_state=rng)

            # extend to overall orbital rotation matrix
            u = np.identity(L, dtype=u2.dtype)
            u[i:i+2, i:i+2] = u2

            # apply transposed single-orbital rotation matrix to Hamiltonian coefficients
            tkin_rotorb = np.einsum(u, (2, 0), u.conj(), (3, 1), tkin, (2, 3), (0, 1))
            vint_rotorb = np.einsum(u, (4, 0), u, (5, 1), u.conj(), (6, 2), u.conj(), (7, 3), vint, (4, 5, 6, 7), (0, 1, 2, 3))

            # rotated reference MPO
            mpoH_rotorb = ptn.molecular_hamiltonian_mpo(tkin_rotorb, vint_rotorb, optimize=False)

            # copy transformed MPO tensors at sites `i` and `i + 1`
            mpoH.A[i    ] = np.copy(mpoH_rotorb.A[i    ])
            mpoH.A[i + 1] = np.copy(mpoH_rotorb.A[i + 1])
            # apply left and right gauge transformations
            v_l, v_r = ptn.molecular_hamiltonian_orbital_gauge_transform(mpoH, u2, i)
            mpoH.A[i    ] = np.einsum(v_l, (2, 4), mpoH.A[i    ], (0, 1, 4, 3), (0, 1, 2, 3))
            mpoH.A[i + 1] = np.einsum(v_r, (3, 4), mpoH.A[i + 1], (0, 1, 2, 4), (0, 1, 2, 3))

            # compare matrix representations
            self.assertTrue(np.allclose(mpoH.as_matrix(), mpoH_rotorb.as_matrix()),
                msg='matrix representation of MPO after orbital rotation and reference Hamiltonian must match')


    def test_spin_molecular_hamiltonian_construction(self):

        rng = np.random.default_rng()

        # number of spin-endowed lattice sites
        L = 4
        # Hamiltonian parameters
        tkin = ptn.crandn(2 * (L,), rng)
        vint = ptn.crandn(4 * (L,), rng)

        # reference Hamiltonian
        Href = construct_spin_molecular_hamiltonian(tkin, vint)

        for opt in (True, False):
            mpoH = ptn.spin_molecular_hamiltonian_mpo(tkin, vint, opt)

            # compare matrix representations
            self.assertTrue(np.allclose(mpoH.as_matrix(), Href.todense()),
                msg='matrix representation of MPO and reference Hamiltonian must match')


def construct_ising_hamiltonian(L: int, J: float, h: float, g: float):
    """
    Construct Ising Hamiltonian `sum J sz sz + h sz + g sx`
    on a 1D lattice as sparse matrix.
    """
    # Pauli-X and Z matrices
    sigma_x = sparse.csr_matrix([[0., 1.], [1.,  0.]])
    sigma_z = sparse.csr_matrix([[1., 0.], [0., -1.]])
    H = sparse.csr_matrix((2**L, 2**L), dtype=float)
    # interaction terms
    hint = sparse.kron(sigma_z, sigma_z)
    for j in range(L - 1):
        H += J * sparse.kron(sparse.identity(2**j),
                 sparse.kron(hint,
                             sparse.identity(2**(L-j-2))))
    # external field
    for j in range(L):
        H += sparse.kron(sparse.identity(2**j),
             sparse.kron(h*sigma_z + g*sigma_x,
                         sparse.identity(2**(L-j-1))))
    return H


def construct_heisenberg_xxz_hamiltonian(L: int, J: float, D: float, h: float):
    """
    Construct XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a 1D lattice as sparse matrix.
    """
    # spin operators
    Sup = np.array([[0.,  1.], [0.,  0. ]])
    Sdn = np.array([[0.,  0.], [1.,  0. ]])
    Sz  = np.array([[0.5, 0.], [0., -0.5]])
    H = sparse.csr_matrix((2**L, 2**L), dtype=float)
    # interaction terms
    hint = J * 0.5 * (sparse.kron(Sup, Sdn) + sparse.kron(Sdn, Sup)) + D * sparse.kron(Sz, Sz)
    for j in range(L - 1):
        H += sparse.kron(sparse.identity(2**j),
             sparse.kron(hint,
                         sparse.identity(2**(L-j-2))))
    # external field
    for j in range(L):
        H -= sparse.kron(sparse.identity(2**j),
             sparse.kron(h*Sz,
                         sparse.identity(2**(L-j-1))))
    return H


def construct_heisenberg_xxz_spin1_hamiltonian(L: int, J: float, D: float, h: float):
    """
    Construct spin-1 XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a 1D lattice as sparse matrix.
    """
    # spin operators
    sq2 = np.sqrt(2.)
    Sup = np.array([[0.,  sq2, 0.], [0.,  0.,  sq2], [0.,  0.,  0.]])
    Sdn = np.array([[0.,  0.,  0.], [sq2, 0.,  0. ], [0.,  sq2, 0.]])
    Sz  = np.array([[1.,  0.,  0.], [0.,  0.,  0. ], [0.,  0., -1.]])
    H = sparse.csr_matrix((3**L, 3**L), dtype=float)
    # interaction terms
    hint = J * 0.5 * (sparse.kron(Sup, Sdn) + sparse.kron(Sdn, Sup)) + D * sparse.kron(Sz, Sz)
    for j in range(L - 1):
        H += sparse.kron(sparse.identity(3**j),
             sparse.kron(hint,
                         sparse.identity(3**(L-j-2))))
    # external field
    for j in range(L):
        H -= sparse.kron(sparse.identity(3**j),
             sparse.kron(h*Sz,
                         sparse.identity(3**(L-j-1))))
    return H


def construct_bose_hubbard_hamiltonian(d: int, L: int, t: float, U: float, mu: float):
    """
    Construct Bose-Hubbard Hamiltonian
    with nearest-neighbor hopping on a 1D lattice as sparse matrix.
    """
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    H = sparse.csr_matrix((d**L, d**L), dtype=float)
    # interaction terms
    hint = -t * (sparse.kron(b_dag, b_ann) + sparse.kron(b_ann, b_dag))
    for j in range(L - 1):
        H += sparse.kron(sparse.identity(d**j),
             sparse.kron(hint,
                         sparse.identity(d**(L-j-2))))
    # external field
    for j in range(L):
        H += sparse.kron(sparse.identity(d**j),
             sparse.kron(0.5*U*(numop @ (numop - np.identity(d))) - mu*numop,
                         sparse.identity(d**(L-j-1))))
    return H


def generate_fermi_operators(L: int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `L` sites (or modes).
    """
    I = sparse.identity(2)
    Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    for i in range(L):
        c = sparse.identity(1)
        for j in range(L):
            if j < i:
                c = sparse.kron(c, I)
            elif j == i:
                c = sparse.kron(c, U)
            else:
                c = sparse.kron(c, Z)
        clist.append(c)
    # corresponding annihilation operators
    alist = [c.conj().T for c in clist]
    return (clist, alist)


def construct_fermi_hubbard_hamiltonian(L: int, t: float, U: float, mu: float):
    """
    Construct Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping on a 1D lattice as sparse matrix.
    """
    clist, alist = generate_fermi_operators(2*L)
    nlist = [c @ a for c, a in zip(clist, alist)]
    H = sparse.csr_matrix((4**L, 4**L), dtype=float)
    # kinetic hopping terms
    for j in range(2*L - 2):
        H -= t * (clist[j] @ alist[j+2] + clist[j+2] @ alist[j])
    # interaction U (n_up-1/2) (n_dn-1/2) and number operator - mu (n_up + n_dn)
    for j in range(0, 2*L, 2):
        H += (U * (nlist[j] - 0.5*sparse.identity(4**L)) @ (nlist[j+1] - 0.5*sparse.identity(4**L))
              - mu * (nlist[j] + nlist[j+1]))
    H.eliminate_zeros()
    return H


def construct_molecular_hamiltonian(tkin, vint):
    """
    Construct a molecular Hamiltonian as sparse matrix.
    """
    L = tkin.shape[0]

    complex_hamiltonian = np.iscomplexobj(tkin) or np.iscomplexobj(vint)
    H = sparse.csr_matrix((2**L, 2**L), dtype=(complex if complex_hamiltonian else float))

    clist, alist = generate_fermi_operators(L)

    # kinetic hopping terms
    for i in range(L):
        for j in range(L):
            H += tkin[i, j] * (clist[i] @ alist[j])
    # interaction terms
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for l in range(L):
                    H += 0.5*vint[i, j, k, l] * (clist[i] @ clist[j] @ alist[l] @ alist[k])
    H.eliminate_zeros()
    return H


def construct_spin_molecular_hamiltonian(tkin, vint):
    """
    Construct a molecular Hamiltonian for a spin orbital basis as sparse matrix.
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)

    L = tkin.shape[0]
    assert tkin.shape == (L, L)
    assert vint.shape == (L, L, L, L)

    # enlarge the single- and two-particle electron overlap integral tensors
    # from an orbital basis without spin to a spin orbital basis

    # single-particle integrals
    tkin_spin = np.kron(tkin, np.eye(2))

    # two-particle integrals
    tmp = np.zeros((2*L, L, 2*L, L), dtype=vint.dtype)
    for i in range(L):
        for j in range(L):
            tmp[:, i, :, j] = np.kron(vint[:, i, :, j], np.eye(2))
    vint_spin = np.zeros((2*L, 2*L, 2*L, 2*L), dtype=vint.dtype)
    for i in range(2*L):
        for j in range(2*L):
            vint_spin[i, :, j, :] = np.kron(tmp[i, :, j, :], np.eye(2))

    return construct_molecular_hamiltonian(tkin_spin, vint_spin)


if __name__ == '__main__':
    unittest.main()
