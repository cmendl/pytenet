import numpy as np
from scipy.stats import unitary_group
from fermi_operators import construct_fermi_operators
import pytenet as ptn


def test_molecular_hamiltonian_mpo():

    rng = np.random.default_rng()

    # number of fermionic modes
    for nsites in range(3, 8):

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


def test_molecular_hamiltonian_orbital_gauge_transform():

    rng = np.random.default_rng()

    # number of fermionic modes
    for nsites in range(4, 8):

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
            h_mpo.a[i    ] = np.einsum(v_l, (0, 4), h_mpo.a[i    ], (4, 1, 2, 3), (0, 1, 2, 3))
            h_mpo.a[i + 1] = np.einsum(v_r, (3, 4), h_mpo.a[i + 1], (0, 1, 2, 4), (0, 1, 2, 3))

            # compare matrix representations
            assert np.allclose(h_mpo.to_matrix(), h_mpo_rotorb.to_matrix()), \
                "matrix representation of MPO after orbital rotation and " \
                "reference Hamiltonian must match"


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
