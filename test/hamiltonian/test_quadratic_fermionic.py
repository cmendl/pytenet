import time
import autoray as ar
from autoray import numpy as np
from fermi_operators import construct_fermi_operators
import pytenet as ptn


def test_quadratic_fermionic_mpo():

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # number of lattice sites
            for nsites in range(2, 8):
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
                coeffc_np = ar.to_numpy(coeffc)
                coeffa_np = ar.to_numpy(coeffa)
                h_ref = sum(coeffc_np[i] * clist[i] for i in range(nsites)) \
                      @ sum(coeffa_np[i] * alist[i] for i in range(nsites))
                # compare
                assert np.allclose(h_mat, np.asarray(h_ref.toarray())), \
                    "matrix representation of MPO and reference operator must match"


def test_quadratic_spin_fermionic_mpo():

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # number of spin-endowed lattice sites
            for nsites in range(2, 6):
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
                    coeffc_np = ar.to_numpy(coeffc)
                    coeffa_np = ar.to_numpy(coeffa)
                    h_ref = sum(coeffc_np[i] * clist[2*i+offset] for i in range(nsites)) \
                          @ sum(coeffa_np[i] * alist[2*i+offset] for i in range(nsites))

                    # compare
                    assert np.allclose(h_mat, np.asarray(h_ref.toarray())), \
                        "matrix representation of MPO and reference operator must match"
