"""
Apply the TDVP time evolution algorithm to a general matrix product operator
by casting it into MPS form.
"""

import copy
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pytenet as ptn
from tangent_space import tangent_space_projector


def entropy(lmbda: np.ndarray):
    """
    Compute the Neumann entropy given the eigenvalues of the density matrix.
    """
    lmbda = lmbda[lmbda > 0]
    return -np.dot(lmbda, np.log2(lmbda))


def schmidt_coefficients_wavefunction(d: int, L: int, psi: np.ndarray):
    """
    Compute the Schmidt coefficients (singular values)
    of a wavefunction for symmetric left-right partitioning.
    """
    return np.linalg.svd(psi.reshape((d**(L//2), d**(L//2))), full_matrices=0, compute_uv=False)


def schmidt_coefficients_operator(d: int, L: int, op: np.ndarray):
    """
    Compute the Schmidt coefficients (singular values)
    of an operator for symmetric left-right partitioning.
    """
    op = op.reshape((d**(L//2), d**(L//2), d**(L//2), d**(L//2))).transpose((0, 2, 1, 3)).reshape(d**L, d**L)
    return np.linalg.svd(op, full_matrices=0, compute_uv=False)


def main():

    # physical local Hilbert space dimension
    d = 2

    # number of lattice sites
    L = 6

    # construct matrix product operator representation of Heisenberg Hamiltonian
    J  =  1.0
    DH =  1.2
    h  = -0.2
    mpoH = ptn.heisenberg_xxz_mpo(L, J, DH, h)
    mpoH.zero_qnumbers()
    # realize commutator [H, .] as matrix product operator
    mpoHcomm = heisenberg_xxz_comm_mpo(L, J, DH, h)
    mpoHcomm.zero_qnumbers()
    print('2-norm of [H, .] operator:', np.linalg.norm(mpoHcomm.as_matrix(), 2))

    mpsH = cast_to_mps(mpoH)
    print('mpsH.bond_dims:', mpsH.bond_dims)
    print('ptn.norm(mpsH):', ptn.norm(mpsH))

    print('[H, .] applied to H as vector (should be zero):',
          np.linalg.norm(mpoHcomm.as_matrix() @ mpsH.as_vector()))

    # initial MPO with random entries (not necessarily Hermitian)
    Dmax = 40
    D = np.minimum(np.minimum(d**(2*np.arange(L + 1)), d**(2*(L - np.arange(L + 1)))), Dmax)
    print('D:', D)
    rng = np.random.default_rng(42)
    op = ptn.MPO(mpoH.qd, [np.zeros(Di, dtype=int) for Di in D], fill='random', rng=rng)
    # effectively clamp virtual bond dimension
    for i in range(L):
        op.A[i][:, :, 2:, :] = 0
        op.A[i][:, :, :, 2:] = 0

    op.orthonormalize(mode='right')
    op.orthonormalize(mode='left')

    # matrix representation
    op_mat = op.as_matrix()

    # cast into MPS form
    psi = cast_to_mps(op)
    print('norm of psi:', np.linalg.norm(psi.as_vector()))
    print('psi.bond_dims:', psi.bond_dims)

    # check: commutator
    comm_ref = op_mat @ mpoH.as_matrix() - mpoH.as_matrix() @ op_mat
    comm = (mpoHcomm.as_matrix() @ psi.as_vector()).reshape((2*L)*[d]).transpose(
            np.concatenate((2*np.arange(L, dtype=int), 2*np.arange(L, dtype=int)+1))).reshape((d**L, d**L))
    print('commutator reference check error:', np.linalg.norm(comm - comm_ref))

    # initial average energy (should be conserved)
    en_avr_0 = np.trace(mpoH.as_matrix() @ op_mat)
    en_avr_0_alt = ptn.vdot(mpsH, psi)
    print('en_avr_0:', en_avr_0)
    print('abs(en_avr_0_alt - en_avr_0):', abs(en_avr_0_alt - en_avr_0))

    # exact Schmidt coefficients (singular values) of initial state
    sigma_0 = schmidt_coefficients_operator(d, L, op_mat)
    plt.semilogy(np.arange(len(sigma_0)) + 1, sigma_0, '.')
    plt.xlabel('i')
    plt.ylabel(r'$\sigma_i$')
    plt.title('Schmidt coefficients of initial state')
    plt.savefig('evolution_operator_schmidt_0.pdf')
    plt.show()

    print('Schmidt coefficients consistency check:', np.linalg.norm(sigma_0 - schmidt_coefficients_wavefunction(d**2, L, psi.as_vector())))

    print('entropy of initial state:', entropy((sigma_0 / np.linalg.norm(sigma_0))**2))

    # mixed real and imaginary time evolution
    t = 0.1 + 0.5j

    # reference calculation: exp(t H) op exp(-t H)
    op_t_ref = expm(t*mpoH.as_matrix()) @ op_mat @ expm(-t*mpoH.as_matrix())

    # Frobenius norm not preserved by mixed real and imaginary time evolution
    print('Frobenius norm of initial op:', np.linalg.norm(op_mat, 'fro'))
    print('Frobenius norm of op(t):     ', np.linalg.norm(op_t_ref, 'fro'))

    # energy should be conserved
    en_avr_t_ref = np.trace(mpoH.as_matrix() @ op_t_ref)
    print('en_avr_t_ref:', en_avr_t_ref)
    print('abs(en_avr_t_ref - en_avr_0):', abs(en_avr_t_ref - en_avr_0))

    # exact Schmidt coefficients (singular values) of time-evolved state
    sigma_t = schmidt_coefficients_operator(d, L, op_t_ref)
    plt.semilogy(np.arange(len(sigma_t)) + 1, sigma_t, '.')
    plt.xlabel('i')
    plt.ylabel(r'$\sigma_i$')
    plt.title(f'Schmidt coefficients of time-evolved state (t = {-1j*t:g})\n(based on exact time evolution)')
    plt.savefig('evolution_operator_schmidt_t.pdf')
    plt.show()

    S = entropy((sigma_t / np.linalg.norm(sigma_t))**2)
    print('entropy of time-evolved state:', S)

    P = tangent_space_projector(psi)
    print('np.linalg.norm(P @ mpsH.as_vector() - mpsH.as_vector()) (in general non-zero):',
          np.linalg.norm(P @ mpsH.as_vector() - mpsH.as_vector()))

    # number of time steps
    numsteps = 2**(np.arange(5))
    err_op = np.zeros(len(numsteps))
    # relative energy error
    err_en = np.zeros(len(numsteps))

    for i, n in enumerate(numsteps):
        print('n:', n)

        # time step
        dt = t / n

        psi_t = copy.deepcopy(psi)
        ptn.tdvp_singlesite(mpoHcomm, psi_t, dt, n, numiter_lanczos=20)
        op_t = cast_to_mpo(psi_t, op.qd)

        err_op[i] = np.linalg.norm(op_t.as_matrix() - op_t_ref, ord=1)

        en_avr_t = np.trace(mpoH.as_matrix() @ op_t.as_matrix())
        # relative energy error
        err_en[i] = abs(en_avr_t - en_avr_0) / abs(en_avr_0)

    dtinv = numsteps / abs(t)
    plt.loglog(dtinv, err_op, '.-')
    # show quadratic scaling for comparison
    plt.loglog(dtinv, 1e-2/dtinv**2, '--')
    plt.xlabel('1/dt')
    plt.ylabel(r'$\Vert\mathcal{O}[A](t) - \mathcal{O}_\mathrm{ref}(t)\Vert_1$')
    plt.title(f'TDVP time evolution (applied to operator) rate of convergence for\nHeisenberg XXZ model (J={J:g}, D={DH:g}, h={h:g}), L={L}, t={-1j*t:g}')
    plt.savefig('evolution_operator_convergence.pdf')
    plt.show()

    plt.loglog(dtinv, err_en, '.-')
    # show quadratic scaling for comparison
    plt.loglog(dtinv, 3e-2/dtinv**2, '--')
    plt.xlabel('1/dt')
    plt.ylabel(r'$\frac{\vert\epsilon(t) - \epsilon(0)\vert}{\vert\epsilon(0)\vert}, \quad \epsilon(t) = \mathrm{tr}[H \mathcal{O}[A](t)]$')
    plt.title(f'TDVP time evolution (applied to operator) relative energy error for\nHeisenberg XXZ model (J={J:g}, D={DH:g}, h={h:g}), L={L}, t={-1j*t:g}')
    plt.savefig('evolution_operator_energy_conv.pdf')
    plt.show()


def cast_to_mps(mpo):
    """
    Cast a matrix product operator into MPS form by combining the pair of local
    physical dimensions into one dimension.
    """
    mps = ptn.MPS(ptn.qnumber_flatten([mpo.qd, -mpo.qd]), mpo.qD, fill=0.0)
    for i in range(mpo.nsites):
        s = mpo.A[i].shape
        mps.A[i] = mpo.A[i].reshape((s[0]*s[1], s[2], s[3])).copy()
        assert ptn.is_qsparse(mps.A[i], [mps.qd, mps.qD[i], -mps.qD[i+1]])
    return mps


def cast_to_mpo(mps, qd):
    """
    Cast a matrix product state into MPO form by interpreting the physical
    dimension as Kronecker product of a pair of dimensions.
    """
    assert np.array_equal(mps.qd, ptn.qnumber_flatten([qd, -qd]))

    mpo = ptn.MPO(qd, mps.qD, fill=0.0)
    for i in range(mps.nsites):
        s = mps.A[i].shape
        mpo.A[i] = mps.A[i].reshape((len(qd), len(qd), s[1], s[2])).copy()
        assert ptn.is_qsparse(mpo.A[i], [mpo.qd, -mpo.qd, mpo.qD[i], -mpo.qD[i+1]])
    return mpo


def construct_comm_opchain(opchain):
    """
    Construct operator chain on enlarged physical space to realize commutator.
    """
    # -opchain acting from the left
    oplist = [np.kron(op, np.identity(len(op))) for op in opchain.oplist]
    oplist[0] *= -1
    opcL = ptn.OpChain(oplist, opchain.qD)
    # opchain acting from the right
    oplist = [np.kron(np.identity(len(op)), op.T) for op in opchain.oplist]
    opcR = ptn.OpChain(oplist, opchain.qD)
    return [opcL, opcR]


def heisenberg_xxz_comm_mpo(L, J, D, h):
    """
    Construct commutator with XXZ Heisenberg Hamiltonian
    'sum J X X + J Y Y + D Z Z - h Z' on a 1D lattice as MPO.
    """
    # physical quantum numbers (multiplied by 2)
    qd = np.array([1, -1])
    # spin operators
    Sup = np.array([[0.,  1.], [0.,  0. ]])
    Sdn = np.array([[0.,  0.], [1.,  0. ]])
    Sz  = np.array([[0.5, 0.], [0., -0.5]])
    # local two-site and single-site terms
    lopchains = [ptn.OpChain([0.5*J*Sup, Sdn], [ 2]),
                 ptn.OpChain([0.5*J*Sdn, Sup], [-2]),
                 ptn.OpChain([D*Sz, Sz], [0]), ptn.OpChain([-h*Sz], [])]
    # convert to MPO form, with local terms acting either on first or second physical dimension
    locopchains = []
    for opchain in lopchains:
        locopchains += construct_comm_opchain(opchain)
    return ptn.local_opchains_to_mpo(ptn.qnumber_flatten([qd, -qd]), L, locopchains)


if __name__ == '__main__':
    main()
