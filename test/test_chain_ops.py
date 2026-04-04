import numpy as np
import pytenet as ptn


def test_mps_norm():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=4)
    # number of lattice sites
    nsites = 6

    # create a random matrix product state
    psi = ptn.MPS.construct_random(
        nsites, qsite, qnum_sector=1, max_vdim=15, rng=rng)
    # rescale to achieve norm of order 1
    for i in range(nsites):
        psi.a[i] *= 5

    # calculate the norm of psi using the MPS representation
    nrm = ptn.mps_norm(psi)

    # reference value
    nrm_ref = np.linalg.norm(psi.to_vector())

    # compare
    assert abs(nrm - nrm_ref) / max(abs(nrm_ref), 1e-12) < 1e-12, \
        "matrix product state norm must match reference value"


def test_mpo_average():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-1, 2, size=3)

    # create random matrix product state
    qbonds_psi = [rng.integers(-1, 2, size=bi) for bi in [1, 7, 26, 19, 25, 8, 1]]
    psi = ptn.MPS(qsite, qbonds_psi, fill="random", rng=rng)
    # rescale to achieve norm of order 1
    for i in range(psi.nsites):
        psi.a[i] *= 5

    # create random matrix product operator
    qbonds_op = [rng.integers(-1, 2, size=bi) for bi in [1, 5, 16, 14, 17, 4, 1]]
    # set leading and trailing bond quantum numbers to zero to avoid zero average
    qbonds_op[ 0] = np.array([0])
    qbonds_op[-1] = np.array([0])
    op = ptn.MPO(qsite, qbonds_op, fill="random", rng=rng)
    # rescale to achieve norm of order 1
    for i in range(op.nsites):
        op.a[i] *= 5

    # calculate average (expectation value) <psi | op | psi>
    avr = ptn.mpo_average(psi, op)

    # reference value based on full Fock space representation
    x = psi.to_vector()
    avr_ref = np.vdot(x, op.to_matrix() @ x)

    # compare
    assert abs(avr - avr_ref) / max(abs(avr_ref), 1e-12) < 1e-12, \
        "operator average must match reference value"


def test_mpo_density_average():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = rng.integers(-1, 2, size=4)

    # represent a density matrix as a random matrix product operator
    # (Hermitian property not relevant here)
    qbonds_rho = [rng.integers(-1, 2, size=bi) for bi in [1, 3, 7, 6, 11, 5, 1]]
    # leading and trailing (dummy) virtual bond quantum numbers
    # must be the same to avoid zero average
    qbonds_rho[-1] = qbonds_rho[0]
    rho = ptn.MPO(qsite, qbonds_rho, fill="random", rng=rng)
    # rescale to achieve norm of order 1
    for i in range(rho.nsites):
        rho.a[i] *= 5

    # create a random matrix product operator
    qbonds_op = [rng.integers(-1, 2, size=bi) for bi in [1, 5, 16, 14, 17, 4, 1]]
    # set leading and trailing bond quantum numbers to zero to avoid zero average
    qbonds_op[ 0] = np.array([0])
    qbonds_op[-1] = np.array([0])
    op = ptn.MPO(qsite, qbonds_op, fill="random", rng=rng)
    # rescale to achieve norm of order 1
    for i in range(op.nsites):
        op.a[i] *= 5

    # calculate average (expectation value) tr[op rho]
    avr = ptn.mpo_density_average(rho, op)

    # reference value based on full Fock space representation
    avr_ref = np.trace(op.to_matrix() @ rho.to_matrix())

    # compare
    assert abs(avr - avr_ref) / max(abs(avr_ref), 1e-12) < 1e-12, \
        "operator average must match reference value"


def test_apply_mpo():

    rng = np.random.default_rng()

    # physical quantum numbers
    qsite = [0, -1, 1]

    # nsites = 6

    # create a random matrix product state
    psi = ptn.MPS(qsite, [rng.integers(-1, 2, size=bi) for bi in [1, 9, 25, 31, 23, 8, 1]],
                  fill="random", rng=rng)
    # rescale to achieve norm of order 1
    for i in range(psi.nsites):
        psi.a[i] *= 5

    # create a random matrix product operator
    op = ptn.MPO(qsite, [rng.integers(-1, 2, size=bi) for bi in [1, 5, 16, 43, 35, 7, 1]],
                 fill="random", rng=rng)
    # rescale to achieve norm of order 1
    for i in range(op.nsites):
        op.a[i] *= 5

    op_psi = ptn.apply_mpo(op, psi)

    # reference
    op_psi_ref = op.to_matrix() @ psi.to_vector()

    # compare
    assert np.allclose(op_psi.to_vector(), op_psi_ref, rtol=1e-12, atol=1e-12)
