import numpy as np
from .mpo import MPO
from .opchain import OpChain

__all__ = ['ising_MPO', 'heisenberg_XXZ_MPO', 'bose_hubbard_MPO', 'fermi_hubbard_MPO', 'local_opchains_to_MPO']


def ising_MPO(L, J, h, g):
    """Construct transverse-field Ising Hamiltonian
    'sum J sz sz + h sz + g sx' on a 1D lattice as MPO."""
    # set physical quantum numbers to zero
    qd = [0, 0]
    # Pauli matrices
    sigma_x = np.array([[0., 1.], [1.,  0.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])
    # local two-site terms
    lopchains = [OpChain([J*sigma_z, sigma_z], [0]), OpChain([h*sigma_z + g*sigma_x], [])]
    # convert to MPO
    return local_opchains_to_MPO(qd, L, lopchains)


def heisenberg_XXZ_MPO(L, J, D, h):
    """Construct XXZ Heisenberg Hamiltonian
    'sum J X X + J Y Y + D Z Z - h Z' on a 1D lattice as MPO."""
    # physical quantum numbers (multiplied by 2)
    qd = [1, -1]
    # spin operators
    Sup = np.array([[0.,  1.], [0.,  0. ]])
    Sdn = np.array([[0.,  0.], [1.,  0. ]])
    Sz  = np.array([[0.5, 0.], [0., -0.5]])
    # local two-site and single-site terms
    lopchains = [OpChain([0.5*J*Sup, Sdn], [ 2]),
                 OpChain([0.5*J*Sdn, Sup], [-2]),
                 OpChain([D*Sz, Sz], [0]), OpChain([-h*Sz], [])]
    # convert to MPO
    return local_opchains_to_MPO(qd, L, lopchains)


def bose_hubbard_MPO(d, L, t, U, mu):
    """Construct Bose-Hubbard Hamiltonian
    with nearest-neighbor hopping on a 1D lattice as MPO.

    Args:
        d   physical dimension per site
            (allowed local occupancies are 0, 1, ..., d - 1)
        L   number of lattice sites
        t   kinetic hopping parameter
        U   interaction strength
        mu  chemical potential
    """
    # physical quantum numbers (particle number)
    qd = np.arange(d)
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    # local two-site and single-site terms
    lopchains = [OpChain([-t*b_dag, b_ann], [ 1]),
                 OpChain([b_ann, -t*b_dag], [-1]),
                 OpChain([0.5*U*np.dot(numop, numop - np.identity(d)) - mu*numop], [])]
    # convert to MPO
    return local_opchains_to_MPO(qd, L, lopchains)


def fermi_hubbard_MPO(L, t, U, mu):
    """Construct Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping on a 1D lattice as MPO.

    States for each spin and site are |0> and |1>.

    Args:
        L   number of lattice sites
        t   kinetic hopping parameter
        U   Hubbard interaction strength
        mu  chemical potential (mu = 0 corresponds to half-filling)
    """
    # physical particle number and spin quantum numbers (encoded as single integer)
    qN = [0,  1,  1,  2]
    qS = [0, -1,  1,  0]
    qd = [(qn[0] << 16) + qn[1] for qn in zip(qN, qS)]
    id2 = np.identity(2)
    # creation and annihilation operators for a single spin and lattice site
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    F = np.array([[1., 0.], [0., -1.]])
    # local two-site and single-site terms
    lopchains = [
        # spin-up kinetic hopping
        OpChain([-t*np.kron(a_dag, F), np.kron(a_ann, id2)], [( 1 << 16) + 1]),
        OpChain([-t*np.kron(a_ann, F), np.kron(a_dag, id2)], [(-1 << 16) - 1]),
        # spin-down kinetic hopping
        OpChain([np.kron(id2, a_dag), -t*np.kron(F, a_ann)], [( 1 << 16) - 1]),
        OpChain([np.kron(id2, a_ann), -t*np.kron(F, a_dag)], [(-1 << 16) + 1]),
        # interaction U (n_up-1/2) (n_dn-1/2) and number operator - mu (n_up + n_dn)
        OpChain([U*np.diag([0.25, -0.25, -0.25, 0.25])
                 - mu*(np.kron(numop, id2) + np.kron(id2, numop))], [])]
    # convert to MPO
    return local_opchains_to_MPO(qd, L, lopchains)


def local_opchains_to_MPO(qd, L, lopchains):
    """Construct Hamiltonian as MPO based on local operator chains, which are shifted along a 1D lattice."""
    opchains = []
    for lopc in lopchains:
        for i in range(L - lopc.length + 1):
            # add shifted opchain
            opchains.append(OpChain(list(lopc.oplist), lopc.qD, istart=i))
    return MPO.from_opchains(qd, L, opchains)
