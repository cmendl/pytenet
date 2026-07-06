import numpy as np
from fermi_operators import construct_fermi_operators
import pytenet as ptn


def test_linear_fermionic_mpo():

    rng = np.random.default_rng()

    # number of lattice sites
    for nsites in range(2, 8):
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


def test_linear_spin_fermionic_mpo():

    rng = np.random.default_rng()

    # number of spin-endowed lattice sites
    for nsites in range(2, 6):
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
