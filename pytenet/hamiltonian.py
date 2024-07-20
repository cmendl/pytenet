import copy
from collections.abc import Sequence, Mapping
from enum import IntEnum
import numpy as np
from .mpo import MPO
from .opchain import OpChain
from .autop import AutOpNode, AutOpEdge, AutOp
from .opgraph import OpGraphNode, OpGraphEdge, OpGraph

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
    class OID(IntEnum):
        I = 0,
        Z = 1,
        X = 2,
    opmap = {
        OID.I: np.identity(2),
        OID.Z: sigma_z,
        OID.X: sigma_x }
    # operator state automaton:
    #
    #     ___   __Z__o__Z__   ___
    #    / > \ /  >     >  \ / > \
    #  I|     o             o     |I
    #    \_<_/ \_____>_____/ \_<_/
    #            h Z + g X
    #
    node_term0 = AutOpNode(0, [], [], 0)
    node_term1 = AutOpNode(1, [], [], 0)
    node_z     = AutOpNode(2, [], [], 0)
    autop = AutOp([node_term0, node_term1, node_z], [], [node_term0.nid, node_term1.nid])
    # identities looping around terminal nodes
    autop.add_connect_edge(AutOpEdge(0, [node_term0.nid, node_term0.nid], [(OID.I, 1.)]))
    autop.add_connect_edge(AutOpEdge(1, [node_term1.nid, node_term1.nid], [(OID.I, 1.)]))
    # Z Z terms
    autop.add_connect_edge(AutOpEdge(2, [node_term0.nid,     node_z.nid], [(OID.Z, J )]))
    autop.add_connect_edge(AutOpEdge(3, [node_z.nid,     node_term1.nid], [(OID.Z, 1.)]))
    # h Z + g X terms
    autop.add_connect_edge(AutOpEdge(4, [node_term0.nid, node_term1.nid], [(OID.Z, h )]))
    autop.add_connect_edge(AutOpEdge(5, [node_term0.nid, node_term1.nid], [(OID.X, g )]))
    assert autop.is_consistent()
    # convert to a graph and then to an MPO
    graph = OpGraph.from_automaton(autop, L)
    return MPO.from_opgraph(qd, graph, opmap)


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
    class OID(IntEnum):
        Sd = -1,
        Id =  0,
        Su =  1,
        Sz =  2,
    opmap = {
        OID.Sd: Sdn,
        OID.Id: np.identity(2),
        OID.Su: Sup,
        OID.Sz: Sz }
    # local two-site and single-site terms
    lopchains = [OpChain([OID.Su, OID.Sd], [0,  2, 0], 0.5*J, 0),
                 OpChain([OID.Sd, OID.Su], [0, -2, 0], 0.5*J, 0),
                 OpChain([OID.Sz, OID.Sz], [0,  0, 0],   D,   0),
                 OpChain([OID.Sz        ], [0,  0   ],  -h,   0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, OID.Id)


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
    class OID(IntEnum):
        Sd = -1,
        Id =  0,
        Su =  1,
        Sz =  2,
    opmap = {
        OID.Sd: Sdn,
        OID.Id: np.identity(3),
        OID.Su: Sup,
        OID.Sz: Sz }
    # local two-site and single-site terms
    lopchains = [OpChain([OID.Su, OID.Sd], [0,  1, 0], 0.5*J, 0),
                 OpChain([OID.Sd, OID.Su], [0, -1, 0], 0.5*J, 0),
                 OpChain([OID.Sz, OID.Sz], [0,  0, 0],   D,   0),
                 OpChain([OID.Sz        ], [0,  0   ],  -h,   0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, OID.Id)


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
    class OID(IntEnum):
        B  = -1,
        Id =  0,
        Bd =  1,
        N  =  2,
        NI =  3,
    opmap = {
        OID.B:  b_ann,
        OID.Id: np.identity(d),
        OID.Bd: b_dag,
        OID.N:  numop,
        OID.NI: numop @ (numop - np.identity(d)) / 2 }
    # local two-site and single-site terms
    lopchains = [OpChain([OID.Bd, OID.B ], [0,  1, 0], -t,  0),
                 OpChain([OID.B,  OID.Bd], [0, -1, 0], -t,  0),
                 OpChain([OID.N         ], [0,  0   ], -mu, 0),
                 OpChain([OID.NI        ], [0,  0   ],  U,  0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, OID.Id)


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
    Z = np.array([[1., 0.], [0., -1.]])
    # operator map
    class OID(IntEnum):
        Id =  0,
        CI =  1,
        AI =  2,
        CZ =  3,
        AZ =  4,
        IC =  5,
        IA =  6,
        ZC =  7,
        ZA =  8,
        Nt =  9,
        NI = 10,
    opmap = {
        OID.Id: np.identity(4),
        OID.CI: np.kron(a_dag, id2),
        OID.AI: np.kron(a_ann, id2),
        OID.CZ: np.kron(a_dag, Z  ),
        OID.AZ: np.kron(a_ann, Z  ),
        OID.IC: np.kron(id2, a_dag),
        OID.IA: np.kron(id2, a_ann),
        OID.ZC: np.kron(Z,   a_dag),
        OID.ZA: np.kron(Z,   a_ann),
        OID.Nt: np.kron(numop, id2) + np.kron(id2, numop),  # n_up + n_dn
        OID.NI: np.diag([0.25, -0.25, -0.25, 0.25])         # (n_up - 1/2) (n_dn - 1/2)
    }
    # local two-site and single-site terms
    lopchains = [
        # spin-up kinetic hopping
        OpChain([OID.CZ, OID.AI], [0, ( 1 << 16) + 1, 0], -t,  0),
        OpChain([OID.AZ, OID.CI], [0, (-1 << 16) - 1, 0], -t,  0),
        # spin-down kinetic hopping
        OpChain([OID.IC, OID.ZA], [0, ( 1 << 16) - 1, 0], -t,  0),
        OpChain([OID.IA, OID.ZC], [0, (-1 << 16) + 1, 0], -t,  0),
        # number operator - mu (n_up + n_dn)
        OpChain([OID.Nt],         [0, 0                ], -mu, 0),
        # interaction U (n_up - 1/2) (n_dn - 1/2)
        OpChain([OID.NI],         [0, 0                ],  U,  0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, OID.Id)


def molecular_hamiltonian_mpo(tkin, vint, optimize=True) -> MPO:
    r"""
    Construct a molecular Hamiltonian as MPO,
    using physicists' convention for the interaction term (note ordering of k and \ell):

    .. math::

        H = \sum_{i,j} t_{i,j} a^{\dagger}_i a_j + \\frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell} a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
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
    class OID(IntEnum):
        A = -1,
        I =  0,
        C =  1,
        N =  2,
        Z =  3,
    opmap = {
        OID.A: a_ann,
        OID.I: np.identity(2),
        OID.C: a_dag,
        OID.N: numop,
        OID.Z: Z }

    # interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k:
    # can anti-commute fermionic operators such that i < j and l < k;
    # global minus sign from Jordan-Wigner transformation, since a Z = -a
    gint = -0.5 * (vint - np.transpose(vint, (1, 0, 2, 3)) - np.transpose(vint, (0, 1, 3, 2)) + np.transpose(vint, (1, 0, 3, 2)))

    if optimize:
        # optimize MPO bond dimensions based on bipartite graph theory
        opchains = []
        # kinetic hopping terms \sum_{i,j} t_{i,j} a^{\dagger}_i a_j
        for i in range(L):
            for j in range(L):
                if i == j:
                    # diagonal hopping term
                    opchains.append(OpChain([OID.N], [0, 0], tkin[i, i], i))
                else:
                    (a, p), (b, q) = sorted([(i, OID.C), (j, OID.A)])
                    opchains.append(OpChain([p] + (b - a - 1)*[OID.Z] + [q],
                                            [0] + (b - a)    *[p]     + [0], tkin[i, j], a))
        # interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k
        for i in range(L):
            for j in range(i + 1, L):   # i < j
                for k in range(L):
                    for l in range(k):  # l < k
                        (a, p), (b, q), (c, r), (d, s) = sorted([(i, OID.C), (j, OID.C), (l, OID.A), (k, OID.A)])
                        if a == b:
                            assert b < c
                            if c == d:
                                # two number operators
                                oids  = [OID.N] + (c - b - 1)*[OID.I] + [OID.N]
                                qnums = (c - b + 2)*[0]
                            else:
                                # number operator at the beginning
                                oids  = [OID.N] + (c - b - 1)*[OID.I] + [r] + (d - c - 1)*[OID.Z] + [s]
                                qnums = (c - b + 1)*[0] + (d - c)*[r] + [0]
                        elif b == c:
                            # number operator in the middle
                            oids  = [p] + (b - a - 1)*[OID.Z] + [OID.N] + (d - c - 1)*[OID.Z] + [s]
                            qnums = [0] + (d - a)*[p] + [0]
                        elif c == d:
                            # number operator at the end
                            oids  = [p] + (b - a - 1)*[OID.Z] + [q] + (c - b - 1)*[OID.I] + [OID.N]
                            qnums = [0] + (b - a)*[p] + (c - b + 1)*[0]
                        else:
                            # generic case: i, j, k, l pairwise different
                            oids  = [p] + (b - a - 1)*[OID.Z] + [q] + (c - b - 1)*[OID.I] + [r] + (d - c - 1)*[OID.Z] + [s]
                            qnums = [0] + (b - a)*[p] + (c - b)*[p + q] + (d - c)*[-s] + [0]
                        opchains.append(OpChain(oids, qnums, gint[i, j, k, l], a))
        graph = OpGraph.from_opchains(opchains, L, 0)

    else:
        # explicit construction (typically faster, but does not optimize cases
        # of zero coefficients, and is slightly sub-optimal close to boundary)

        assert L >= 4

        # nodes
        nid_next = 0
        # identity chains from the left and right
        nodes_identity_l = {}
        nodes_identity_r = {}
        for i in range(L - 1):
            nodes_identity_l[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1
        for i in range(2, L + 1):
            nodes_identity_r[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1
        # a^{\dagger}_i operators connected to left terminal
        nodes_a_dag_l = {}
        for i in range(L - 2):
            nodes_a_dag_l[i] = {}
            for j in range(i + 1, L - 1):
                nodes_a_dag_l[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1
        # a_i operators connected to left terminal
        nodes_a_ann_l = {}
        for i in range(L - 2):
            nodes_a_ann_l[i] = {}
            for j in range(i + 1, L - 1):
                nodes_a_ann_l[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
        nodes_a_dag_a_dag_l = {}
        for i in range(L//2 - 1):
            for j in range(i + 1, L//2):
                nodes_a_dag_a_dag_l[i, j] = {}
                for k in range(j + 1, L//2 + 1):
                    nodes_a_dag_a_dag_l[i, j][k] = OpGraphNode(nid_next, [], [], 2)
                    nid_next += 1
        # a_i a_j operators connected to left terminal
        nodes_a_ann_a_ann_l = {}
        for i in range(L//2 - 1):
            for j in range(i + 1, L//2):
                nodes_a_ann_a_ann_l[i, j] = {}
                for k in range(j + 1, L//2 + 1):
                    nodes_a_ann_a_ann_l[i, j][k] = OpGraphNode(nid_next, [], [], -2)
                    nid_next += 1
        # a^{\dagger}_i a_j operators connected to left terminal
        nodes_a_dag_a_ann_l = {}
        for i in range(L//2):
            for j in range(L//2):
                nodes_a_dag_a_ann_l[i, j] = {}
                for k in range(max(i, j) + 1, L//2 + 1):
                    nodes_a_dag_a_ann_l[i, j][k] = OpGraphNode(nid_next, [], [], 0)
                    nid_next += 1
        # a^{\dagger}_i operators connected to right terminal
        nodes_a_dag_r = {}
        for i in range(2, L):
            nodes_a_dag_r[i] = {}
            for j in range(2, i + 1):
                nodes_a_dag_r[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # a_i operators connected to right terminal
        nodes_a_ann_r = {}
        for i in range(2, L):
            nodes_a_ann_r[i] = {}
            for j in range(2, i + 1):
                nodes_a_ann_r[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
        nodes_a_dag_a_dag_r = {}
        for i in range(L//2 + 1, L - 1):
            for j in range(i + 1, L):
                nodes_a_dag_a_dag_r[i, j] = {}
                for k in range(L//2 + 1, i + 1):
                    nodes_a_dag_a_dag_r[i, j][k] = OpGraphNode(nid_next, [], [], -2)
                    nid_next += 1
        # a_i a_j operators connected to right terminal
        nodes_a_ann_a_ann_r = {}
        for i in range(L//2 + 1, L - 1):
            for j in range(i + 1, L):
                nodes_a_ann_a_ann_r[i, j] = {}
                for k in range(L//2 + 1, i + 1):
                    nodes_a_ann_a_ann_r[i, j][k] = OpGraphNode(nid_next, [], [], 2)
                    nid_next += 1
        # a^{\dagger}_i a_j operators connected to right terminal
        nodes_a_dag_a_ann_r = {}
        for i in range(L//2 + 1, L):
            for j in range(L//2 + 1, L):
                nodes_a_dag_a_ann_r[i, j] = {}
                for k in range(L//2 + 1, min(i, j) + 1):
                    nodes_a_dag_a_ann_r[i, j][k] = OpGraphNode(nid_next, [], [], 0)
                    nid_next += 1

        # initialize graph with nodes
        graph = OpGraph(list(nodes_identity_l.values()) +
                        list(nodes_identity_r.values()) +
                        [node for nodes in nodes_a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in nodes_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in nodes_a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in nodes_a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in nodes_a_dag_a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in nodes_a_ann_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in nodes_a_dag_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in nodes_a_dag_a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in nodes_a_ann_a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in nodes_a_dag_a_ann_r.values() for node in nodes.values()],
                        [], [nodes_identity_l[0].nid, nodes_identity_r[L].nid])

        # edges
        eid_next = 0
        # identities connected to left and right terminals
        for i in range(L - 2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_identity_l[i].nid, nodes_identity_l[i + 1].nid], [(OID.I, 1.)]))
            eid_next += 1
        for i in range(2, L):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_identity_r[i].nid, nodes_identity_r[i + 1].nid], [(OID.I, 1.)]))
            eid_next += 1
        # a^{\dagger}_i operators connected to left terminal
        for i in range(L - 2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_identity_l[i].nid, nodes_a_dag_l[i][i + 1].nid], [(OID.C, 1.)]))
            eid_next += 1
            # Z operator from Jordan-Wigner transformation
            for j in range(i + 1, L - 2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_dag_l[i][j].nid, nodes_a_dag_l[i][j + 1].nid], [(OID.Z, 1.)]))
                eid_next += 1
        # a_i operators connected to left terminal
        for i in range(L - 2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_identity_l[i].nid, nodes_a_ann_l[i][i + 1].nid], [(OID.A, 1.)]))
            eid_next += 1
            # Z operator from Jordan-Wigner transformation
            for j in range(i + 1, L - 2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_ann_l[i][j].nid, nodes_a_ann_l[i][j + 1].nid], [(OID.Z, 1.)]))
                eid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
        for i in range(L//2 - 1):
            for j in range(i + 1, L//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_dag_l[i][j].nid, nodes_a_dag_a_dag_l[i, j][j + 1].nid], [(OID.C, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(j + 1, L//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_dag_a_dag_l[i, j][k].nid, nodes_a_dag_a_dag_l[i, j][k + 1].nid], [(OID.I, 1.)]))
                    eid_next += 1
        # a_i a_j operators connected to left terminal
        for i in range(L//2 - 1):
            for j in range(i + 1, L//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_ann_l[i][j].nid, nodes_a_ann_a_ann_l[i, j][j + 1].nid], [(OID.A, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(j + 1, L//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_ann_a_ann_l[i, j][k].nid, nodes_a_ann_a_ann_l[i, j][k + 1].nid], [(OID.I, 1.)]))
                    eid_next += 1
        # a^{\dagger}_i a_j operators connected to left terminal
        for i in range(L//2):
            for j in range(L//2):
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_dag_l[i][j].nid, nodes_a_dag_a_ann_l[i, j][j + 1].nid], [(OID.A, 1.)]))
                    eid_next += 1
                elif i == j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_identity_l[i].nid, nodes_a_dag_a_ann_l[i, j][i + 1].nid], [(OID.N, 1.)]))
                    eid_next += 1
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_ann_l[j][i].nid, nodes_a_dag_a_ann_l[i, j][i + 1].nid], [(OID.C, 1.)]))
                    eid_next += 1
                # identities for transition to next site
                for k in range(max(i, j) + 1, L//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_dag_a_ann_l[i, j][k].nid, nodes_a_dag_a_ann_l[i, j][k + 1].nid], [(OID.I, 1.)]))
                    eid_next += 1
        # a^{\dagger}_i operators connected to right terminal
        for i in range(2, L):
            # Z operator from Jordan-Wigner transformation
            for j in range(2, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_dag_r[i][j].nid, nodes_a_dag_r[i][j + 1].nid], [(OID.Z, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_a_dag_r[i][i].nid, nodes_identity_r[i + 1].nid], [(OID.C, 1.)]))
            eid_next += 1
        # a_i operators connected to right terminal
        for i in range(2, L):
            # Z operator from Jordan-Wigner transformation
            for j in range(2, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_ann_r[i][j].nid, nodes_a_ann_r[i][j + 1].nid], [(OID.Z, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_a_ann_r[i][i].nid, nodes_identity_r[i + 1].nid], [(OID.A, 1.)]))
            eid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
        for i in range(L//2 + 1, L - 1):
            for j in range(i + 1, L):
                # identities for transition to next site
                for k in range(L//2 + 1, i):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_dag_a_dag_r[i, j][k].nid, nodes_a_dag_a_dag_r[i, j][k + 1].nid], [(OID.I, 1.)]))
                    eid_next += 1
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_dag_a_dag_r[i, j][i].nid, nodes_a_dag_r[j][i + 1].nid], [(OID.C, 1.)]))
                eid_next += 1
        # a_i a_j operators connected to right terminal
        for i in range(L//2 + 1, L - 1):
            for j in range(i + 1, L):
                # identities for transition to next site
                for k in range(L//2 + 1, i):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_ann_a_ann_r[i, j][k].nid, nodes_a_ann_a_ann_r[i, j][k + 1].nid], [(OID.I, 1.)]))
                    eid_next += 1
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_ann_a_ann_r[i, j][i].nid, nodes_a_ann_r[j][i + 1].nid], [(OID.A, 1.)]))
                eid_next += 1
        # a^{\dagger}_i a_j operators connected to right terminal
        for i in range(L//2 + 1, L):
            for j in range(L//2 + 1, L):
                # identities for transition to next site
                for k in range(L//2 + 1, min(i, j)):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_dag_a_ann_r[i, j][k].nid, nodes_a_dag_a_ann_r[i, j][k + 1].nid], [(OID.I, 1.)]))
                    eid_next += 1
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_dag_a_ann_r[i, j][i].nid, nodes_a_ann_r[j][i + 1].nid], [(OID.C, 1.)]))
                    eid_next += 1
                elif i == j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_dag_a_ann_r[i, j][i].nid, nodes_identity_r[i + 1].nid], [(OID.N, 1.)]))
                    eid_next += 1
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_dag_a_ann_r[i, j][j].nid, nodes_a_dag_r[i][j + 1].nid], [(OID.A, 1.)]))
                    eid_next += 1
        # diagonal kinetic terms t_{i,i} n_i
        for i in range(L//2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_a_dag_a_ann_l[i, i][i + 1].nid, nodes_identity_r[i + 2].nid], [(OID.I, tkin[i, i])]))
            eid_next += 1
        for i in range(L//2 + 1, L):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_identity_l[i - 1].nid, nodes_a_dag_a_ann_r[i, i][i].nid], [(OID.I, tkin[i, i])]))
            eid_next += 1
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [nodes_identity_l[L//2].nid, nodes_identity_r[L//2 + 1].nid], [(OID.N, tkin[L//2, L//2])]))
        eid_next += 1
        # t_{i,j} a^{\dagger}_i a_j terms, for i < j
        for i in range(L//2):
            for j in range(i + 1, L//2 + 1):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_dag_l[i][j].nid, nodes_identity_r[j + 1].nid], [(OID.A, tkin[i, j])]))
                eid_next += 1
        for j in range(L//2 + 1, L):
            for i in range(L//2, j):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_identity_l[i].nid, nodes_a_ann_r[j][i + 1].nid], [(OID.C, tkin[i, j])]))
                eid_next += 1
        for i in range(L//2):
            for j in range(L//2 + 1, L):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_dag_l[i][L//2].nid, nodes_a_ann_r[j][L//2 + 1].nid], [(OID.Z, tkin[i, j])]))
                eid_next += 1
        # t_{i,j} a^{\dagger}_i a_j terms, for i > j
        for j in range(L//2):
            for i in range(j + 1, L//2 + 1):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_ann_l[j][i].nid, nodes_identity_r[i + 1].nid], [(OID.C, tkin[i, j])]))
                eid_next += 1
        for i in range(L//2 + 1, L):
            for j in range(L//2, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_identity_l[j].nid, nodes_a_dag_r[i][j + 1].nid], [(OID.A, tkin[i, j])]))
                eid_next += 1
        for j in range(L//2):
            for i in range(L//2 + 1, L):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_a_ann_l[j][L//2].nid, nodes_a_dag_r[i][L//2 + 1].nid], [(OID.Z, tkin[i, j])]))
                eid_next += 1
        # g_{i,j,k,j} a^{\dagger}_i n_j a_k terms, for i < j < k
        for i in range(L - 2):
            for j in range(i + 1, L - 1):
                for k in range(j + 1, L):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_dag_l[i][j].nid, nodes_a_ann_r[k][j + 1].nid], [(OID.N, gint[i, j, k, j])]))
                    eid_next += 1
        # g_{i,j,i,l} a_l n_i a^{\dagger}_j terms, for l < i < j
        for l in range(L - 2):
            for i in range(l + 1, L - 1):
                for j in range(i + 1, L):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [nodes_a_ann_l[l][i].nid, nodes_a_dag_r[j][i + 1].nid], [(OID.N, gint[i, j, i, l])]))
                    eid_next += 1
        # g_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k terms, for i < j < l < k
        for i in range(L//2 - 1):
            for j in range(i + 1, L//2):
                for l in range(j + 1, L//2 + 1):
                    for k in range(l + 1, L):
                        graph.add_connect_edge(
                            OpGraphEdge(eid_next, [nodes_a_dag_a_dag_l[i, j][l].nid, nodes_a_ann_r[k][l + 1].nid], [(OID.A, gint[i, j, k, l])]))
                        eid_next += 1
        for l in range(L//2 + 1, L - 1):
            for k in range(l + 1, L):
                for j in range(L//2, l):
                    for i in range(j):
                        graph.add_connect_edge(
                            OpGraphEdge(eid_next, [nodes_a_dag_l[i][j].nid, nodes_a_ann_a_ann_r[l, k][j + 1].nid], [(OID.C, gint[i, j, k, l])]))
                        eid_next += 1
        for i in range(L//2 - 1):
            for j in range(i + 1, L//2):
                for l in range(L//2 + 1, L - 1):
                    for k in range(l + 1, L):
                        graph.add_connect_edge(
                            OpGraphEdge(eid_next, [nodes_a_dag_a_dag_l[i, j][L//2].nid, nodes_a_ann_a_ann_r[l, k][L//2 + 1].nid], [(OID.I, gint[i, j, k, l])]))
                        eid_next += 1
        # g_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k terms, for l < k < i < j
        for l in range(L//2 - 1):
            for k in range(l + 1, L//2):
                for i in range(k + 1, L//2 + 1):
                    for j in range(i + 1, L):
                        graph.add_connect_edge(
                            OpGraphEdge(eid_next, [nodes_a_ann_a_ann_l[l, k][i].nid, nodes_a_dag_r[j][i + 1].nid], [(OID.C, gint[i, j, k, l])]))
                        eid_next += 1
        for i in range(L//2 + 1, L - 1):
            for j in range(i + 1, L):
                for k in range(L//2, i):
                    for l in range(k):
                        graph.add_connect_edge(
                            OpGraphEdge(eid_next, [nodes_a_ann_l[l][k].nid, nodes_a_dag_a_dag_r[i, j][k + 1].nid], [(OID.A, gint[i, j, k, l])]))
                        eid_next += 1
        for l in range(L//2 - 1):
            for k in range(l + 1, L//2):
                for i in range(L//2 + 1, L - 1):
                    for j in range(i + 1, L):
                        graph.add_connect_edge(
                            OpGraphEdge(eid_next, [nodes_a_ann_a_ann_l[l, k][L//2].nid, nodes_a_dag_a_dag_r[i, j][L//2 + 1].nid], [(OID.I, gint[i, j, k, l])]))
                        eid_next += 1
        # g_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k terms, for i, l < j, k
        for i in range(L//2):
            for l in range(L//2):
                for j in range(max(i, l) + 1, L):
                    for k in range(max(i, l) + 1, L):
                        if min(j, k) > L//2:
                            continue
                        if j < k:
                            graph.add_connect_edge(
                                OpGraphEdge(eid_next, [nodes_a_dag_a_ann_l[i, l][j].nid, nodes_a_ann_r[k][j + 1].nid], [(OID.C, gint[i, j, k, l])]))
                            eid_next += 1
                        elif j == k:
                            graph.add_connect_edge(
                                OpGraphEdge(eid_next, [nodes_a_dag_a_ann_l[i, l][j].nid, nodes_identity_r[j + 1].nid], [(OID.N, gint[i, j, k, l])]))
                            eid_next += 1
                        else:  # j > k
                            graph.add_connect_edge(
                                OpGraphEdge(eid_next, [nodes_a_dag_a_ann_l[i, l][k].nid, nodes_a_dag_r[j][k + 1].nid], [(OID.A, gint[i, j, k, l])]))
                            eid_next += 1
        for j in range(L//2 + 1, L):
            for k in range(L//2 + 1, L):
                for i in range(min(j, k)):
                    for l in range(min(j, k)):
                        if max(i, l) < L//2:
                            continue
                        if i < l:
                            graph.add_connect_edge(
                                OpGraphEdge(eid_next, [nodes_a_dag_l[i][l].nid, nodes_a_dag_a_ann_r[j, k][l + 1].nid], [(OID.A, gint[i, j, k, l])]))
                            eid_next += 1
                        elif i == l:
                            graph.add_connect_edge(
                                OpGraphEdge(eid_next, [nodes_identity_l[i].nid, nodes_a_dag_a_ann_r[j, k][i + 1].nid], [(OID.N, gint[i, j, k, l])]))
                            eid_next += 1
                        else:  # i > l
                            graph.add_connect_edge(
                                OpGraphEdge(eid_next, [nodes_a_ann_l[l][i].nid, nodes_a_dag_a_ann_r[j, k][i + 1].nid], [(OID.C, gint[i, j, k, l])]))
                            eid_next += 1
        for i in range(L//2):
            for l in range(L//2):
                for j in range(L//2 + 1, L):
                    for k in range(L//2 + 1, L):
                        graph.add_connect_edge(
                            OpGraphEdge(eid_next, [nodes_a_dag_a_ann_l[i, l][L//2].nid, nodes_a_dag_a_ann_r[j, k][L//2 + 1].nid], [(OID.I, gint[i, j, k, l])]))
                        eid_next += 1

    # skip consistency check for larger L (would take very long)
    if L <= 12:
        assert graph.is_consistent()
    # convert to MPO
    return MPO.from_opgraph([0, 1], graph, opmap)


def _local_opchains_to_mpo(qd: Sequence[int], lopchains: Sequence[OpChain], size: int, opmap: Mapping, oid_identity: int) -> MPO:
    """
    Construct Hamiltonian as MPO based on local operator chains,
    which are shifted along a 1D lattice.

    Args:
        qd: physical quantum numbers at each site
        lopchains: local operator chains
        size: system size, i.e., number of lattice sites
        opmap: local operators as dictionary, using operator IDs as keys
        oid_identity: operator ID for identity map

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
