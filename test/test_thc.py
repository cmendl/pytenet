import time
import autoray as ar
from autoray import numpy as np
import torch
from scipy import sparse
from hamiltonian.test_spin_molecular import construct_spin_molecular_hamiltonian
import pytenet as ptn


def test_thc_spin_molecular_hamiltonian():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # number of spin-endowed lattice sites
            nsites = 4
            # THC rank
            thc_rank = 7

            for use_complex in (False, True):
                h_thc = _generate_random_thc_hamiltonian(nsites, thc_rank, use_complex, rng)
                if backend == "torch" and torch.cuda.is_available():
                    h_thc.to_device("gpu")

                # reference Hamiltonian
                vint_thc = np.einsum(
                    h_thc.thc_kernel, (4, 5),
                    h_thc.thc_transform, (0, 4),
                    h_thc.thc_transform, (1, 4),
                    h_thc.thc_transform, (2, 5),
                    h_thc.thc_transform, (3, 5),
                    (0, 1, 2, 3))
                # switch to different ordering convention for interaction term
                h_ref = construct_spin_molecular_hamiltonian(
                    h_thc.tkin + 0.5*np.trace(vint_thc, axis1=1, axis2=2),
                    np.transpose(vint_thc, (0, 2, 1, 3)))

                # compare matrix representations
                h_thc_dns = h_thc.to_matrix(sparse_format=False)
                assert np.allclose(h_thc_dns, np.asarray(h_ref.toarray(), like=h_thc_dns)), \
                    "matrix representation of THC and reference Hamiltonian must match"
                assert sparse.linalg.norm(h_thc.to_matrix(sparse_format=True) - h_ref) < 1e-12, \
                    "matrix representation of THC and reference Hamiltonian must match"


def test_apply_thc_spin_molecular_hamiltonian():

    torch.set_default_dtype(torch.float64)

    for backend in ["numpy", "torch"]:
        with ar.backend_like(backend):
            rng = ar.do("random.default_rng", int(time.time()))

            # number of spin-endowed lattice sites
            nsites = 5
            # THC rank
            thc_rank = 8

            for use_complex in (False, True):
                h_thc = _generate_random_thc_hamiltonian(nsites, thc_rank, use_complex, rng)
                if backend == "torch" and torch.cuda.is_available():
                    h_thc.to_device("gpu")

                # create a random matrix product state
                # physical particle number and spin quantum numbers (encoded as single integer)
                qn = [0,  1,  1,  2]
                qs = [0, -1,  1,  0]
                qsite = [ptn.encode_quantum_number_pair(q[0], q[1]) for q in zip(qn, qs)]
                b = [1, 15, 22, 33, 13, 1]
                # ensure that the MPS does not represent a zero vector
                while True:
                    qbonds = [[ptn.encode_quantum_number_pair(
                                ptn.random_qnumbers(-1, 2, size=1, rng=rng)[0],
                                ptn.random_qnumbers(-1, 2, size=1, rng=rng)[0])
                        for _ in range(bi)]
                        for bi in b]
                    psi = ptn.MPS(qsite, qbonds,
                                  fill="random" if use_complex else "random real",
                                  rng=rng)
                    if ptn.mps_norm(psi) > 0:
                        break
                # rescale to achieve norm of order 1
                for i in range(psi.nsites):
                    psi.a[i] *= 5
                if backend == "torch" and torch.cuda.is_available():
                    psi.to_device("gpu")

                h_psi_ref = h_thc.to_matrix(sparse_format=True) @ ar.to_numpy(psi.to_vector())

                for tol in (0, 1e-3):
                    h_psi = ptn.apply_thc_spin_molecular_hamiltonian(h_thc, psi, tol)

                    # compare H |psi> vectors
                    ctol = (1e-13 if tol == 0 else 0.005)
                    h_psi_vec = h_psi.to_vector()
                    assert np.allclose(h_psi_vec, np.asarray(h_psi_ref, like=h_psi_vec),
                                       atol=ctol, rtol=ctol)


def _generate_random_thc_hamiltonian(nsites: int, thc_rank: int, use_complex: bool, rng):
    """
    Generate a spin molecular Hamiltonian using the tensor hypercontraction
    representation with random coefficients.
    """
    if use_complex:
        # kinetic coefficients can be complex in general
        tkin = 0.4 * ptn.crandn((nsites, nsites), rng)
    else:
        tkin = 0.4 * rng.normal(size=(nsites, nsites))
    tkin = 0.5 * (tkin + tkin.conj().T)

    thc_kernel = rng.normal(size=(thc_rank, thc_rank))
    thc_kernel = 0.5 * (thc_kernel + thc_kernel.T)
    thc_transform = 0.4 * rng.normal(size=(nsites, thc_rank))

    return ptn.THCSpinMolecularHamiltonian(tkin, thc_kernel, thc_transform)
