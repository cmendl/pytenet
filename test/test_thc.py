import numpy as np
from scipy import sparse
from test_hamiltonian import construct_spin_molecular_hamiltonian
import pytenet as ptn


def test_thc_spin_molecular_hamiltonian():

    rng = np.random.default_rng()

    # number of spin-endowed lattice sites
    nsites = 4
    # THC rank
    thc_rank = 7

    h_thc = _generate_random_thc_hamiltonian(nsites, thc_rank, rng)

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
        vint_thc.transpose((0, 2, 1, 3)))

    # compare matrix representations
    assert np.allclose(h_thc.to_matrix(sparse_format=False), h_ref.todense()), \
        "matrix representation of THC and reference Hamiltonian must match"
    assert sparse.linalg.norm(h_thc.to_matrix(sparse_format=True) - h_ref) < 1e-12, \
        "matrix representation of THC and reference Hamiltonian must match"


def test_apply_thc_spin_molecular_hamiltonian():

    rng = np.random.default_rng()

    # number of spin-endowed lattice sites
    nsites = 5
    # THC rank
    thc_rank = 8

    h_thc = _generate_random_thc_hamiltonian(nsites, thc_rank, rng)

    # create a random matrix product state
    # physical particle number and spin quantum numbers (encoded as single integer)
    qn = [0,  1,  1,  2]
    qs = [0, -1,  1,  0]
    qsite = [ptn.encode_quantum_number_pair(q[0], q[1]) for q in zip(qn, qs)]
    b = [1, 15, 22, 33, 13, 1]
    # ensure that the MPS does not represent a zero vector
    while True:
        qbonds = [[ptn.encode_quantum_number_pair(rng.integers(-1, 2), rng.integers(-1, 2))
               for _ in range(bi)]
               for bi in b]
        psi = ptn.MPS(qsite, qbonds, fill="random", rng=rng)
        if ptn.mps_norm(psi) > 0:
            break
    # rescale to achieve norm of order 1
    for i in range(psi.nsites):
        psi.a[i] *= 5

    h_psi_ref = h_thc.to_matrix(sparse_format=True) @ psi.to_vector()

    for tol in (0, 1e-3):
        h_psi = ptn.apply_thc_spin_molecular_hamiltonian(h_thc, psi, tol)

        # compare H |psi> vectors
        ctol = (1e-13 if tol == 0 else 0.005)
        assert np.allclose(h_psi.to_vector(), h_psi_ref, atol=ctol, rtol=ctol)


def _generate_random_thc_hamiltonian(nsites: int, thc_rank: int, rng: np.random.Generator):
    """
    Generate a spin molecular Hamiltonian using the tensor hypercontraction
    representation with random coefficients.
    """
    # kinetic coefficients can be complex in general
    tkin = 0.4 * ptn.crandn((nsites, nsites), rng)
    tkin = 0.5 * (tkin + tkin.conj().T)

    thc_kernel = rng.normal(size=(thc_rank, thc_rank))
    thc_kernel = 0.5 * (thc_kernel + thc_kernel.T)
    thc_transform = 0.4 * rng.normal(size=(nsites, thc_rank))

    return ptn.THCSpinMolecularHamiltonian(tkin, thc_kernel, thc_transform)
