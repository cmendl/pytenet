import copy
from collections.abc import Sequence, Mapping
import numpy as np
from .mpo import MPO
from .opchain import OpChain
from .opgraph import OpGraph

__all__ = ['ising_mpo', 'heisenberg_xxz_mpo', 'heisenberg_xxz_spin1_mpo', 'bose_hubbard_mpo', 'fermi_hubbard_mpo', 'molecular_hamiltonian_mpo']


def ising_mpo(L: int, J: float, h: float, g: float) -> MPO:
    """
    Construct Ising Hamiltonian `sum J sz sz + h sz + g sx`
    on a 1D lattice as MPO.

    Args:
        L:  number of lattice sites
        J:  interaction parameter
        h:  longitudinal-field strength
        g:  transverse-field strength

    Returns:
        MPO: Ising Hamiltonian
    """
    # set physical quantum numbers to zero
    qd = [0, 0]
    # Pauli matrices
    sigma_x = np.array([[0., 1.], [1.,  0.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])
    # operator map
    opmap = {
        0: np.identity(2),
        1: sigma_z,
        2: h*sigma_z + g*sigma_x }
    # local two-site and single-site terms
    lopchains = [OpChain([1, 1], [0, 0, 0], J,   0),
                 OpChain([2   ], [0, 0   ], 1.0, 0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, 0)


def heisenberg_xxz_mpo(L: int, J: float, D: float, h: float) -> MPO:
    """
    Construct XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a 1D lattice as MPO.

    Args:
        L:  number of lattice sites
        J:  J parameter
        D:  Delta parameter
        h:  field strength

    Returns:
        MPO: XXZ Heisenberg Hamiltonian
    """
    # physical quantum numbers (multiplied by 2)
    qd = [1, -1]
    # spin operators
    Sup = np.array([[0.,  1.], [0.,  0. ]])
    Sdn = np.array([[0.,  0.], [1.,  0. ]])
    Sz  = np.array([[0.5, 0.], [0., -0.5]])
    # operator map
    opmap = {
        -1: Sdn,
         0: np.identity(2),
         1: Sup,
         2: Sz }
    # local two-site and single-site terms
    lopchains = [OpChain([ 1, -1], [0,  2, 0], 0.5*J, 0),
                 OpChain([-1,  1], [0, -2, 0], 0.5*J, 0),
                 OpChain([ 2,  2], [0,  0, 0],   D,   0),
                 OpChain([ 2    ], [0,  0   ],  -h,   0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, 0)


def heisenberg_xxz_spin1_mpo(L: int, J: float, D: float, h: float) -> MPO:
    """
    Construct spin-1 XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a 1D lattice as MPO.

    Args:
        L:  number of lattice sites
        J:  J parameter
        D:  Delta parameter
        h:  field strength

    Returns:
        MPO: spin-1 XXZ Heisenberg Hamiltonian
    """
    # physical quantum numbers
    qd = [1, 0, -1]
    # spin operators
    sq2 = np.sqrt(2.)
    Sup = np.array([[0.,  sq2, 0.], [0.,  0.,  sq2], [0.,  0.,  0.]])
    Sdn = np.array([[0.,  0.,  0.], [sq2, 0.,  0. ], [0.,  sq2, 0.]])
    Sz  = np.array([[1.,  0.,  0.], [0.,  0.,  0. ], [0.,  0., -1.]])
    # operator map
    opmap = {
        -1: Sdn,
         0: np.identity(3),
         1: Sup,
         2: Sz }
    # local two-site and single-site terms
    lopchains = [OpChain([ 1, -1], [0,  1, 0], 0.5*J, 0),
                 OpChain([-1,  1], [0, -1, 0], 0.5*J, 0),
                 OpChain([ 2,  2], [0,  0, 0],   D,   0),
                 OpChain([ 2    ], [0,  0   ],  -h,   0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, 0)


def bose_hubbard_mpo(d: int, L: int, t: float, U: float, mu: float) -> MPO:
    """
    Construct Bose-Hubbard Hamiltonian
    with nearest-neighbor hopping on a 1D lattice as MPO.

    Args:
        d:  physical dimension per site
            (allowed local occupancies are 0, 1, ..., d - 1)
        L:  number of lattice sites
        t:  kinetic hopping parameter
        U:  interaction strength
        mu: chemical potential

    Returns:
        MPO: Bose-Hubbard Hamiltonian
    """
    # physical quantum numbers (particle number)
    qd = np.arange(d)
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    # operator map
    opmap = {
        -1: b_ann,
         0: np.identity(d),
         1: b_dag,
         2: 0.5*U*(numop @ (numop - np.identity(d))) - mu*numop }
    # local two-site and single-site terms
    lopchains = [OpChain([ 1, -1], [0,  1, 0], -t,  0),
                 OpChain([-1,  1], [0, -1, 0], -t,  0),
                 OpChain([ 2    ], [0,  0   ], 1.0, 0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, 0)


def fermi_hubbard_mpo(L: int, t: float, U: float, mu: float) -> MPO:
    """
    Construct Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping on a 1D lattice as MPO.

    States for each spin and site are `|0>` and `|1>`.

    Args:
        L:  number of lattice sites
        t:  kinetic hopping parameter
        U:  Hubbard interaction strength
        mu: chemical potential (mu = 0 corresponds to half-filling)

    Returns:
        MPO: Fermi-Hubbard Hamiltonian
    """
    # physical particle number and spin quantum numbers (encoded as single integer)
    qN = [0,  1,  1,  2]
    qS = [0, -1,  1,  0]
    qd = [(qn[0] << 16) + qn[1] for qn in zip(qN, qS)]
    id2 = np.identity(2)
    # creation and annihilation operators for a single spin and lattice site
    a_dag = np.array([[0., 0.], [1., 0.]])
    a_ann = np.array([[0., 1.], [0., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    F = np.array([[1., 0.], [0., -1.]])
    # operator map
    opmap = {
        0: np.identity(4),
        1: np.kron(a_dag, F),
        2: np.kron(a_ann, F),
        3: np.kron(a_ann, id2),
        4: np.kron(a_dag, id2),
        5: np.kron(id2, a_dag),
        6: np.kron(id2, a_ann),
        7: np.kron(F,   a_ann),
        8: np.kron(F,   a_dag),
        9: U*np.diag([0.25, -0.25, -0.25, 0.25]) - mu*(np.kron(numop, id2) + np.kron(id2, numop)) }
    # local two-site and single-site terms
    lopchains = [
        # spin-up kinetic hopping
        OpChain([1, 3], [0, ( 1 << 16) + 1, 0], -t,  0),
        OpChain([2, 4], [0, (-1 << 16) - 1, 0], -t,  0),
        # spin-down kinetic hopping
        OpChain([5, 7], [0, ( 1 << 16) - 1, 0], -t,  0),
        OpChain([6, 8], [0, (-1 << 16) + 1, 0], -t,  0),
        # interaction U (n_up-1/2) (n_dn-1/2) and number operator - mu (n_up + n_dn)
        OpChain([9   ], [0, 0                ], 1.0, 0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, 0)


def molecular_hamiltonian_mpo(tkin, vint) -> MPO:
    r"""
    Construct a molecular Hamiltonian as MPO,
    using physicists' convention for the interaction term (note ordering of k and \ell):

    .. math::

        H = c + \sum_{i,j} t_{i,j} a^{\dagger}_i a_j + \\frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell} a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    L = tkin.shape[0]
    assert tkin.shape == (L, L)
    assert vint.shape == (L, L, L, L)

    # local operators
    # creation and annihilation operators for a single spin and lattice site
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    Z = np.array([[1., 0.], [0., -1.]])
    # operator map
    opmap = {
        -1: a_ann,
         0: np.identity(2),
         1: a_dag,
         2: numop,
         3: Z }

    opchains = []
    # kinetic hopping terms t_{i,j} a^{\dagger}_i a_j
    for i in range(L):
        for j in range(L):
            if i == j:
                # diagonal hopping term
                opchains.append(OpChain([2], [0, 0], tkin[i, i], i))
            else:
                (a, p), (b, q) = sorted([(i, 1), (j, -1)])
                opchains.append(OpChain([p] + (b - a - 1)*[3] + [q],
                                        [0] + (b - a)*[p] + [0], tkin[i, j], a))
    # interaction terms v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k:
    # can anti-commute fermionic operators such that i < j and l < k;
    # global minus sign from Jordan-Wigner transformation
    gint = -0.5 * (vint - np.transpose(vint, (1, 0, 2, 3)) - np.transpose(vint, (0, 1, 3, 2)) + np.transpose(vint, (1, 0, 3, 2)))
    for i in range(L):
        for j in range(i + 1, L):   # i < j
            for k in range(L):
                for l in range(k):  # l < k
                    (a, p), (b, q), (c, r), (d, s) = sorted([(i, 1), (j, 1), (l, -1), (k, -1)])
                    if a == b:
                        assert b < c
                        if c == d:
                            # two number operators
                            oids  = [2] + (c - b - 1)*[0] + [2]
                            qnums = (c - b + 2)*[0]
                        else:
                            # number operator at the beginning
                            oids  = [2] + (c - b - 1)*[0] + [r] + (d - c - 1)*[3] + [s]
                            qnums = (c - b + 1)*[0] + (d - c)*[r] + [0]
                    elif b == c:
                        # number operator in the middle
                        oids  = [p] + (b - a - 1)*[3] + [2] + (d - c - 1)*[3] + [s]
                        qnums = [0] + (d - a)*[p] + [0]
                    elif c == d:
                        # number operator at the end
                        oids  = [p] + (b - a - 1)*[3] + [q] + (c - b - 1)*[0] + [2]
                        qnums = [0] + (b - a)*[p] + (c - b + 1)*[0]
                    else:
                        # generic case: i, j, k, l pairwise different
                        oids  = [p] + (b - a - 1)*[3] + [q] + (c - b - 1)*[0] + [r] + (d - c - 1)*[3] + [s]
                        qnums = [0] + (b - a)*[p] + (c - b)*[p + q] + (d - c)*[-s] + [0]
                    opchains.append(OpChain(oids, qnums, gint[i, j, k, l], a))
    opgraph = OpGraph.from_opchains(opchains, L, 0)
    # convert to MPO
    H = MPO.from_opgraph([0, 1], opgraph, opmap)
    return H


def _local_opchains_to_mpo(qd: Sequence[int], lopchains: Sequence[OpChain], size: int, opmap: Mapping, oid_identity: int) -> MPO:
    """
    Construct Hamiltonian as MPO based on local operator chains,
    which are shifted along a 1D lattice.

    Args:
        qd: physical quantum numbers at each site
        lopchains: local operator chains
        size: system size, i.e., number of lattice sites
        opmap:

    Returns:
        MPO: Hamiltonian as MPO
    """
    opchains = []
    for lopc in lopchains:
        for i in range(size - lopc.length + 1):
            # add shifted opchain; shallow copy sufficient here
            chain = copy.copy(lopc)
            chain.istart = i
            opchains.append(chain)
    graph = OpGraph.from_opchains(opchains, size, oid_identity)
    return MPO.from_opgraph(qd, graph, opmap)
