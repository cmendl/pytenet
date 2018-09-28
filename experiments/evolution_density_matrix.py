"""
Apply the TDVP time evolution algorithm to a general density matrix
by casting it into MPS form.
"""

from __future__ import print_function
import numpy as np
from scipy.linalg import expm
import copy
import pytenet as ptn


def main():

    # physical dimension
    d = 2

    # number of lattice sites
    L = 6

    # construct matrix product operator representation of Heisenberg Hamiltonian
    J  =  1.0
    DH =  1.2
    h  = -0.2
    mpoH = ptn.heisenberg_XXZ_MPO(L, J, DH, h)
    mpoH.zero_qnumbers()
    # realize commutator [H, .] as matrix product operator
    mpoHcomm = heisenberg_XXZ_comm_MPO(L, J, DH, h)
    mpoHcomm.zero_qnumbers()
    print('norm of [H, .] operator:', np.linalg.norm(mpoHcomm.as_matrix()))

    # initial density matrix as MPO with random entries (not necessarily Hermitian)
    Dmax = 20
    D = np.minimum(np.minimum(d**(2*np.arange(L + 1)), d**(2*(L - np.arange(L + 1)))), Dmax)
    np.random.seed(42)
    rho = ptn.MPO(mpoH.qd, [np.zeros(Di, dtype=int) for Di in D], fill='random')
    # effectively clamp virtual bond dimension
    for i in range(L):
        rho.A[i][:, :, 3:, :] = 0
        rho.A[i][:, :, :, 3:] = 0
    rho.orthonormalize(mode='right')
    rho.orthonormalize(mode='left')

    # matrix representation
    rho_mat = rho.as_matrix()

    # cast density matrix into MPS form
    psi = cast_to_MPS(rho)
    print('norm of psi:', np.linalg.norm(psi.as_vector()))
    print('psi.bond_dims:', psi.bond_dims)

    # check: commutator
    comm_ref = np.dot(mpoH.as_matrix(), rho_mat) - np.dot(rho_mat, mpoH.as_matrix())
    comm = np.dot(mpoHcomm.as_matrix(), psi.as_vector()).reshape((2*L)*[d]).transpose(
            np.concatenate((2*np.arange(L, dtype=int), 2*np.arange(L, dtype=int)+1))).reshape((d**L, d**L))
    print('commutator reference check error: {:g}'.format(np.linalg.norm(comm - comm_ref)))

    # relatively small evolution time since norm of [H, .] operator is large
    t = 0.01 + 0.05j

    # reference calculation: exp(-t H) rho exp(t H)
    rho_t_ref = np.dot(np.dot(expm(-t*mpoH.as_matrix()), rho_mat), expm(t*mpoH.as_matrix()))

    # Frobenius norm not preserved by time evolution
    print('Frobenius norm of initial rho:', np.linalg.norm(rho_mat, 'fro'))
    print('Frobenius norm of rho(t):     ', np.linalg.norm(rho_t_ref, 'fro'))

    # number of steps
    numsteps = 20
    # time step
    dt = t / numsteps

    # run time evolution
    psi_t = copy.deepcopy(psi)
    ptn.integrate_local_singlesite(mpoHcomm, psi_t, dt, numsteps, numiter_lanczos=10)
    rho_t = cast_to_MPO(psi_t, rho.qd)

    # compare with reference (error should be approximately 1e-4)
    err = np.linalg.norm(rho_t.as_matrix() - rho_t_ref)
    print('time evolution error: {:g}'.format(err))


def cast_to_MPS(mpo):
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


def cast_to_MPO(mps, qd):
    """
    Cast a matrix product state into MPO form by interpreting the physical
    dimension as Kronecker product of a pair of dimensions.
    """
    assert not np.any(mps.qd - ptn.qnumber_flatten([qd, -qd]))

    mpo = ptn.MPO(qd, mps.qD, fill=0.0)
    for i in range(mps.nsites):
        s = mps.A[i].shape
        mpo.A[i] = mps.A[i].reshape((len(qd), len(qd), s[1], s[2])).copy()
        assert ptn.is_qsparse(mpo.A[i], [mpo.qd, -mpo.qd, mpo.qD[i], -mpo.qD[i+1]])
    return mpo


def construct_comm_opchain(opchain):
    """Construct operator chain on enlarged physical space to realize commutator."""
    # opchain acting from the left
    oplist = [np.kron(op, np.identity(len(op))) for op in opchain.oplist]
    opcL = ptn.OpChain(oplist, opchain.qD)
    # -opchain acting from the right
    oplist = [np.kron(np.identity(len(op)), op.T) for op in opchain.oplist]
    oplist[0] *= -1
    opcR = ptn.OpChain(oplist, opchain.qD)
    return [opcL, opcR]


def heisenberg_XXZ_comm_MPO(L, J, D, h):
    """Construct commutator with XXZ Heisenberg Hamiltonian
    'sum J X X + J Y Y + D Z Z - h Z' on a 1D lattice as MPO."""
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
    return ptn.local_opchains_to_MPO(ptn.qnumber_flatten([qd, -qd]), L, locopchains)


if __name__ == '__main__':
    main()
