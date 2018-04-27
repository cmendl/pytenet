import numpy as np
from mpo import MPO
from opchain import OpChain


def ising_MPO(L, J, h, g):
    """Construct transverse-field Ising Hamiltonian
    'sum J sz sz + h sz + g sx' on a 1D lattice as MPO."""
    # Pauli matrices
    sigma_x = np.array([[0., 1.], [1.,  0.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])
    # local two-site terms
    oplists = [[J*sigma_z, sigma_z], [h*sigma_z + g*sigma_x]]
    # convert to MPO
    return _local_oplists_to_MPO(2, L, oplists)


def heisenberg_XXZ_MPO(L, J, D, h):
    """Construct XXZ Heisenberg Hamiltonian
    'sum J X X + J Y Y + D Z Z - h Z' on a 1D lattice as MPO."""
    # spin operators
    Sup = np.array([[0.,  1.], [0.,  0. ]])
    Sdn = np.array([[0.,  0.], [1.,  0. ]])
    Sz  = np.array([[0.5, 0.], [0., -0.5]])
    # local two-site and single-site terms
    oplists = [[0.5*J*Sup, Sdn], [0.5*J*Sdn, Sup], [D*Sz, Sz], [-h*Sz]]
    # convert to MPO
    return _local_oplists_to_MPO(2, L, oplists)


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
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    # local two-site and single-site terms
    oplists = [[-t*b_dag, b_ann], [b_ann, -t*b_dag],
               [0.5*U*np.dot(numop, numop - np.identity(d)) - mu*numop]]
    # convert to MPO
    return _local_oplists_to_MPO(d, L, oplists)


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
    id2 = np.identity(2)
    # creation and annihilation operators for a single spin and lattice site
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    F = np.array([[1., 0.], [0., -1.]])
    # local two-site and single-site terms
    oplists = [
        # spin-up kinetic hopping
        [-t*np.kron(a_dag, F), np.kron(a_ann, id2)],
        [-t*np.kron(a_ann, F), np.kron(a_dag, id2)],
        # spin-down kinetic hopping
        [np.kron(id2, a_dag), -t*np.kron(F, a_ann)],
        [np.kron(id2, a_ann), -t*np.kron(F, a_dag)],
        # interaction U (n_up-1/2) (n_dn-1/2) and number operator - mu (n_up + n_dn)
        [U*np.diag([0.25, -0.25, -0.25, 0.25])
         - mu*(np.kron(numop, id2) + np.kron(id2, numop))]]
    # convert to MPO
    return _local_oplists_to_MPO(4, L, oplists)


def _local_oplists_to_MPO(d, L, locoplists):
    """Construct Hamiltonian as MPO based on local operator chains, which are shifted along a 1D lattice."""
    opchains = []
    for oplist in locoplists:
        for i in range(L - len(oplist) + 1):
            # add shifted opchain
            opchains.append(OpChain(i, oplist.copy()))
    return MPO(d, L=L, opchains=opchains)
