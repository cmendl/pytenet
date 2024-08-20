import copy
from collections.abc import Sequence, Mapping
from enum import IntEnum
import numpy as np
from .mpo import MPO
from .opchain import OpChain
from .autop import AutOpNode, AutOpEdge, AutOp
from .opgraph import OpGraphNode, OpGraphEdge, OpGraph

__all__ = ['ising_mpo', 'heisenberg_xxz_mpo', 'heisenberg_xxz_spin1_mpo',
           'bose_hubbard_mpo', 'fermi_hubbard_mpo',
           'molecular_hamiltonian_mpo', 'molecular_hamiltonian_orbital_gauge_transform',
           'molecular_spin_hamiltonian_mpo']


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
        I = 0
        Z = 1
        X = 2
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
        Sd = -1
        Id =  0
        Su =  1
        Sz =  2
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
        Sd = -1
        Id =  0
        Su =  1
        Sz =  2
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
        B  = -1
        Id =  0
        Bd =  1
        N  =  2
        NI =  3
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


def _encode_quantum_number_pair(qa: int, qb: int):
    """
    Encode a pair of quantum numbers into a single quantum number.
    """
    return (qa << 16) + qb


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
    qd = [_encode_quantum_number_pair(q[0], q[1]) for q in zip(qN, qS)]
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
        Id =  0
        CI =  1
        AI =  2
        CZ =  3
        AZ =  4
        IC =  5
        IA =  6
        ZC =  7
        ZA =  8
        Nt =  9
        NI = 10
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
        OpChain([OID.CZ, OID.AI], [0, _encode_quantum_number_pair( 1,  1), 0], -t, 0),
        OpChain([OID.AZ, OID.CI], [0, _encode_quantum_number_pair(-1, -1), 0], -t, 0),
        # spin-down kinetic hopping
        OpChain([OID.IC, OID.ZA], [0, _encode_quantum_number_pair( 1, -1), 0], -t, 0),
        OpChain([OID.IA, OID.ZC], [0, _encode_quantum_number_pair(-1,  1), 0], -t, 0),
        # number operator - mu (n_up + n_dn)
        OpChain([OID.Nt], [0, 0], -mu, 0),
        # interaction U (n_up - 1/2) (n_dn - 1/2)
        OpChain([OID.NI], [0, 0],  U,  0)]
    # convert to MPO
    return _local_opchains_to_mpo(qd, lopchains, L, opmap, OID.Id)


class MolecularOID(IntEnum):
    """
    Local operator IDs for a molecular Hamiltonian.
    """
    A = -1
    I =  0
    C =  1
    N =  2
    Z =  3


def _molecular_hamiltonian_generate_operator_map():
    """
    Generate the local operator map for a molecular Hamiltonian.
    """
    # creation and annihilation operators for a single spin and lattice site
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    Z = np.array([[1., 0.], [0., -1.]])
    # operator map
    opmap = {
        MolecularOID.A: a_ann,
        MolecularOID.I: np.identity(2),
        MolecularOID.C: a_dag,
        MolecularOID.N: numop,
        MolecularOID.Z: Z
    }
    return opmap


class MolecularOpGraphNodes:
    """
    Operator graph nodes used for molecular Hamiltonian construction.
    """
    def __init__(self, L):

        self.L = L

        nid_next = 0
        # identity chains from the left and right
        self.identity_l = {}
        self.identity_r = {}
        for i in range(L):
            self.identity_l[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1
        for i in range(1, L + 1):
            self.identity_r[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1
        # a^{\dagger}_i operators connected to left terminal
        self.a_dag_l = {}
        for i in range(L - 2):
            self.a_dag_l[i] = {}
            for j in range(i + 1, L - 1):
                self.a_dag_l[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1
        # a_i operators connected to left terminal
        self.a_ann_l = {}
        for i in range(L - 2):
            self.a_ann_l[i] = {}
            for j in range(i + 1, L - 1):
                self.a_ann_l[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
        self.a_dag_a_dag_l = {}
        for i in range(L//2 - 1):
            for j in range(i + 1, L//2):
                self.a_dag_a_dag_l[i, j] = {}
                for k in range(j + 1, L//2 + 1):
                    self.a_dag_a_dag_l[i, j][k] = OpGraphNode(nid_next, [], [], 2)
                    nid_next += 1
        # a_i a_j operators connected to left terminal
        self.a_ann_a_ann_l = {}
        for i in range(L//2):
            for j in range(i):
                self.a_ann_a_ann_l[i, j] = {}
                for k in range(i + 1, L//2 + 1):
                    self.a_ann_a_ann_l[i, j][k] = OpGraphNode(nid_next, [], [], -2)
                    nid_next += 1
        # a^{\dagger}_i a_j operators connected to left terminal
        self.a_dag_a_ann_l = {}
        for i in range(L//2):
            for j in range(L//2):
                self.a_dag_a_ann_l[i, j] = {}
                for k in range(max(i, j) + 1, L//2 + 1):
                    self.a_dag_a_ann_l[i, j][k] = OpGraphNode(nid_next, [], [], 0)
                    nid_next += 1
        # a^{\dagger}_i operators connected to right terminal
        self.a_dag_r = {}
        for i in range(2, L):
            self.a_dag_r[i] = {}
            for j in range(2, i + 1):
                self.a_dag_r[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # a_i operators connected to right terminal
        self.a_ann_r = {}
        for i in range(2, L):
            self.a_ann_r[i] = {}
            for j in range(2, i + 1):
                self.a_ann_r[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
        self.a_dag_a_dag_r = {}
        for i in range(L//2 + 1, L - 1):
            for j in range(i + 1, L):
                self.a_dag_a_dag_r[i, j] = {}
                for k in range(L//2 + 1, i + 1):
                    self.a_dag_a_dag_r[i, j][k] = OpGraphNode(nid_next, [], [], -2)
                    nid_next += 1
        # a_i a_j operators connected to right terminal
        self.a_ann_a_ann_r = {}
        for i in range(L//2 + 1, L):
            for j in range(L//2 + 1, i):
                self.a_ann_a_ann_r[i, j] = {}
                for k in range(L//2 + 1, j + 1):
                    self.a_ann_a_ann_r[i, j][k] = OpGraphNode(nid_next, [], [], 2)
                    nid_next += 1
        # a^{\dagger}_i a_j operators connected to right terminal
        self.a_dag_a_ann_r = {}
        for i in range(L//2 + 1, L):
            for j in range(L//2 + 1, L):
                self.a_dag_a_ann_r[i, j] = {}
                for k in range(L//2 + 1, min(i, j) + 1):
                    self.a_dag_a_ann_r[i, j][k] = OpGraphNode(nid_next, [], [], 0)
                    nid_next += 1

    def get(self, oplist: Sequence[tuple], connection):
        """
        Retrieve nodes corresponding to a single or pairs of creation and annihilation operators.
        """
        if len(oplist) == 1:
            (i, oid) = oplist[0]
            if oid == MolecularOID.C:
                return self.a_dag_l[i] if connection == "left" else self.a_dag_r[i]
            if oid == MolecularOID.A:
                return self.a_ann_l[i] if connection == "left" else self.a_ann_r[i]
            raise KeyError(f"nodes for OID {oid} and i == {i} do not exist")
        if len(oplist) == 2:
            (i, oid_i), (j, oid_j) = oplist
            if (oid_i, oid_j) == (MolecularOID.C, MolecularOID.C):
                i, j = sorted((i, j))
                return self.a_dag_a_dag_l[i, j] if connection == "left" else self.a_dag_a_dag_r[i, j]
            if (oid_i, oid_j) == (MolecularOID.A, MolecularOID.A):
                i, j = sorted((i, j), reverse=True)
                return self.a_ann_a_ann_l[i, j] if connection == "left" else self.a_ann_a_ann_r[i, j]
            if (oid_i, oid_j) == (MolecularOID.C, MolecularOID.A):
                return self.a_dag_a_ann_l[i, j] if connection == "left" else self.a_dag_a_ann_r[i, j]
            if (oid_i, oid_j) == (MolecularOID.A, MolecularOID.C):
                return self.a_dag_a_ann_l[j, i] if connection == "left" else self.a_dag_a_ann_r[j, i]
            raise KeyError(f"nodes for OIDs ({oid_i}, {oid_j}) and (i, j) == ({i}, {j}) do not exist")
        raise KeyError(f"nodes for operator list of length {len(oplist)} do not exist")

    def generate_graph(self) -> OpGraph:
        """
        Create and initialize an operator graph with corresponding nodes and edges.
        """
        L = self.L

        # initialize graph with nodes
        graph = OpGraph(list(self.identity_l.values()) +
                        list(self.identity_r.values()) +
                        [node for nodes in self.a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_r.values() for node in nodes.values()],
                        [], [self.identity_l[0].nid, self.identity_r[L].nid])
        # edges
        eid_next = 0
        # identities connected to left and right terminals
        for i in range(L - 1):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.identity_l[i + 1].nid], [(MolecularOID.I, 1.)]))
            eid_next += 1
        for i in range(1, L):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_r[i].nid, self.identity_r[i + 1].nid], [(MolecularOID.I, 1.)]))
            eid_next += 1
        # a^{\dagger}_i operators connected to left terminal
        for i in range(L - 2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_dag_l[i][i + 1].nid], [(MolecularOID.C, 1.)]))
            eid_next += 1
            # Z operator from Jordan-Wigner transformation
            for j in range(i + 1, L - 2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_l[i][j].nid, self.a_dag_l[i][j + 1].nid], [(MolecularOID.Z, 1.)]))
                eid_next += 1
        # a_i operators connected to left terminal
        for i in range(L - 2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_ann_l[i][i + 1].nid], [(MolecularOID.A, 1.)]))
            eid_next += 1
            # Z operator from Jordan-Wigner transformation
            for j in range(i + 1, L - 2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_l[i][j].nid, self.a_ann_l[i][j + 1].nid], [(MolecularOID.Z, 1.)]))
                eid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
        for i in range(L//2 - 1):
            for j in range(i + 1, L//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_l[i][j].nid, self.a_dag_a_dag_l[i, j][j + 1].nid], [(MolecularOID.C, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(j + 1, L//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_l[i, j][k].nid, self.a_dag_a_dag_l[i, j][k + 1].nid], [(MolecularOID.I, 1.)]))
                    eid_next += 1
        # a_i a_j operators connected to left terminal
        for i in range(L//2):
            for j in range(i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_l[j][i].nid, self.a_ann_a_ann_l[i, j][i + 1].nid], [(MolecularOID.A, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(i + 1, L//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_l[i, j][k].nid, self.a_ann_a_ann_l[i, j][k + 1].nid], [(MolecularOID.I, 1.)]))
                    eid_next += 1
        # a^{\dagger}_i a_j operators connected to left terminal
        for i in range(L//2):
            for j in range(L//2):
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_l[i][j].nid, self.a_dag_a_ann_l[i, j][j + 1].nid], [(MolecularOID.A, 1.)]))
                    eid_next += 1
                elif i == j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_dag_a_ann_l[i, j][i + 1].nid], [(MolecularOID.N, 1.)]))
                    eid_next += 1
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_l[j][i].nid, self.a_dag_a_ann_l[i, j][i + 1].nid], [(MolecularOID.C, 1.)]))
                    eid_next += 1
                # identities for transition to next site
                for k in range(max(i, j) + 1, L//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_l[i, j][k].nid, self.a_dag_a_ann_l[i, j][k + 1].nid], [(MolecularOID.I, 1.)]))
                    eid_next += 1
        # a^{\dagger}_i operators connected to right terminal
        for i in range(2, L):
            # Z operator from Jordan-Wigner transformation
            for j in range(2, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_r[i][j].nid, self.a_dag_r[i][j + 1].nid], [(MolecularOID.Z, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_dag_r[i][i].nid, self.identity_r[i + 1].nid], [(MolecularOID.C, 1.)]))
            eid_next += 1
        # a_i operators connected to right terminal
        for i in range(2, L):
            # Z operator from Jordan-Wigner transformation
            for j in range(2, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_r[i][j].nid, self.a_ann_r[i][j + 1].nid], [(MolecularOID.Z, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_ann_r[i][i].nid, self.identity_r[i + 1].nid], [(MolecularOID.A, 1.)]))
            eid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
        for i in range(L//2 + 1, L - 1):
            for j in range(i + 1, L):
                # identities for transition to next site
                for k in range(L//2 + 1, i):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, j][k].nid, self.a_dag_a_dag_r[i, j][k + 1].nid], [(MolecularOID.I, 1.)]))
                    eid_next += 1
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, j][i].nid, self.a_dag_r[j][i + 1].nid], [(MolecularOID.C, 1.)]))
                eid_next += 1
        # a_i a_j operators connected to right terminal
        for i in range(L//2 + 1, L):
            for j in range(L//2 + 1, i):
                # identities for transition to next site
                for k in range(L//2 + 1, j):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, j][k].nid, self.a_ann_a_ann_r[i, j][k + 1].nid], [(MolecularOID.I, 1.)]))
                    eid_next += 1
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, j][j].nid, self.a_ann_r[i][j + 1].nid], [(MolecularOID.A, 1.)]))
                eid_next += 1
        # a^{\dagger}_i a_j operators connected to right terminal
        for i in range(L//2 + 1, L):
            for j in range(L//2 + 1, L):
                # identities for transition to next site
                for k in range(L//2 + 1, min(i, j)):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, j][k].nid, self.a_dag_a_ann_r[i, j][k + 1].nid], [(MolecularOID.I, 1.)]))
                    eid_next += 1
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, j][i].nid, self.a_ann_r[j][i + 1].nid], [(MolecularOID.C, 1.)]))
                    eid_next += 1
                elif i == j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, j][i].nid, self.identity_r[i + 1].nid], [(MolecularOID.N, 1.)]))
                    eid_next += 1
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, j][j].nid, self.a_dag_r[i][j + 1].nid], [(MolecularOID.A, 1.)]))
                    eid_next += 1

        return graph

    def copy_nids(self, target):
        """
        Transfer and store the node IDs in the target object.
        """
        L = self.L
        # identity chains from the left and right
        target.nids_identity_l = {}
        target.nids_identity_r = {}
        for i in range(L):
            target.nids_identity_l[i] = self.identity_l[i].nid
        for i in range(1, L + 1):
            target.nids_identity_r[i] = self.identity_r[i].nid
        # a^{\dagger}_i operators connected to left terminal
        target.nids_a_dag_l = {}
        for i in range(L - 2):
            target.nids_a_dag_l[i] = {}
            for j in range(i + 1, L - 1):
                target.nids_a_dag_l[i][j] = self.a_dag_l[i][j].nid
        # a_i operators connected to left terminal
        target.nids_a_ann_l = {}
        for i in range(L - 2):
            target.nids_a_ann_l[i] = {}
            for j in range(i + 1, L - 1):
                target.nids_a_ann_l[i][j] = self.a_ann_l[i][j].nid
        # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
        target.nids_a_dag_a_dag_l = {}
        for i in range(L//2 - 1):
            for j in range(i + 1, L//2):
                target.nids_a_dag_a_dag_l[i, j] = {}
                for k in range(j + 1, L//2 + 1):
                    target.nids_a_dag_a_dag_l[i, j][k] = self.a_dag_a_dag_l[i, j][k].nid
        # a_i a_j operators connected to left terminal
        target.nids_a_ann_a_ann_l = {}
        for i in range(L//2):
            for j in range(i):
                target.nids_a_ann_a_ann_l[i, j] = {}
                for k in range(i + 1, L//2 + 1):
                    target.nids_a_ann_a_ann_l[i, j][k] = self.a_ann_a_ann_l[i, j][k].nid
        # a^{\dagger}_i a_j operators connected to left terminal
        target.nids_a_dag_a_ann_l = {}
        for i in range(L//2):
            for j in range(L//2):
                target.nids_a_dag_a_ann_l[i, j] = {}
                for k in range(max(i, j) + 1, L//2 + 1):
                    target.nids_a_dag_a_ann_l[i, j][k] = self.a_dag_a_ann_l[i, j][k].nid
        # a^{\dagger}_i operators connected to right terminal
        target.nids_a_dag_r = {}
        for i in range(2, L):
            target.nids_a_dag_r[i] = {}
            for j in range(2, i + 1):
                target.nids_a_dag_r[i][j] = self.a_dag_r[i][j].nid
        # a_i operators connected to right terminal
        target.nids_a_ann_r = {}
        for i in range(2, L):
            target.nids_a_ann_r[i] = {}
            for j in range(2, i + 1):
                target.nids_a_ann_r[i][j] = self.a_ann_r[i][j].nid
        # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
        target.nids_a_dag_a_dag_r = {}
        for i in range(L//2 + 1, L - 1):
            for j in range(i + 1, L):
                target.nids_a_dag_a_dag_r[i, j] = {}
                for k in range(L//2 + 1, i + 1):
                    target.nids_a_dag_a_dag_r[i, j][k] = self.a_dag_a_dag_r[i, j][k].nid
        # a_i a_j operators connected to right terminal
        target.nids_a_ann_a_ann_r = {}
        for i in range(L//2 + 1, L):
            for j in range(L//2 + 1, i):
                target.nids_a_ann_a_ann_r[i, j] = {}
                for k in range(L//2 + 1, j + 1):
                    target.nids_a_ann_a_ann_r[i, j][k] = self.a_ann_a_ann_r[i, j][k].nid
        # a^{\dagger}_i a_j operators connected to right terminal
        target.nids_a_dag_a_ann_r = {}
        for i in range(L//2 + 1, L):
            for j in range(L//2 + 1, L):
                target.nids_a_dag_a_ann_r[i, j] = {}
                for k in range(L//2 + 1, min(i, j) + 1):
                    target.nids_a_dag_a_ann_r[i, j][k] = self.a_dag_a_ann_r[i, j][k].nid


def _molecular_hamiltonian_graph_add_term(graph: OpGraph, nodes: MolecularOpGraphNodes, oplist: Sequence[tuple], coeff: float):
    """
    Add an operator term (operator string of creation and annihilation operators)
    to the operator graph describing a molecular Hamiltonian.
    """
    eid_next = max(graph.edges.keys()) + 1

    L = nodes.L

    # sort by site (orbital) index
    oplist = sorted(oplist)

    if len(oplist) == 2:
        (i, oid_i), (j, oid_j) = oplist
        if i == j:
            # expecting number operator
            assert (oid_i, oid_j) == (MolecularOID.A, MolecularOID.C)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid], [(MolecularOID.N, coeff)]))
        else:
            assert i < j
            if j <= L//2:
                nodes_l = nodes.get([(i, oid_i)], "left")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes.identity_r[j + 1].nid], [(oid_j, coeff)]))
            elif i >= L//2:
                nodes_r = nodes.get([(j, oid_j)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes_r[i + 1].nid], [(oid_i, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid_i)], "left")
                nodes_r = nodes.get([(j, oid_j)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[L//2].nid, nodes_r[L//2 + 1].nid], [(MolecularOID.Z, coeff)]))
    elif len(oplist) == 4:
        (i, oid_i), (j, oid_j), (k, oid_k), (l, oid_l) = oplist
        if j == k:
            # expecting number operator
            assert (oid_j, oid_k) == (MolecularOID.A, MolecularOID.C)
            nodes_l = nodes.get([(i, oid_i)], "left")
            nodes_r = nodes.get([(l, oid_l)], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid], [(MolecularOID.N, coeff)]))
        elif k <= L//2:
            nodes_l = nodes.get([(i, oid_i), (j, oid_j)], "left")
            if k == l:
                # expecting number operator
                assert (oid_k, oid_l) == (MolecularOID.A, MolecularOID.C)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes.identity_r[k + 1].nid], [(MolecularOID.N, coeff)]))
            else:
                nodes_r = nodes.get([(l, oid_l)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes_r[k + 1].nid], [(oid_k, coeff)]))
        elif j >= L//2:
            nodes_r = nodes.get([(k, oid_k), (l, oid_l)], "right")
            if i == j:
                # expecting number operator
                assert (oid_i, oid_j) == (MolecularOID.A, MolecularOID.C)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[j].nid, nodes_r[j + 1].nid], [(MolecularOID.N, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid_i)], "left")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid], [(oid_j, coeff)]))
        else:
            nodes_l = nodes.get([(i, oid_i), (j, oid_j)], "left")
            nodes_r = nodes.get([(k, oid_k), (l, oid_l)], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[L//2].nid, nodes_r[L//2 + 1].nid], [(MolecularOID.I, coeff)]))
    else:
        raise NotImplementedError


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

    # interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k:
    # can anti-commute fermionic operators such that i < j and k < l
    gint = 0.5 * (vint - np.transpose(vint, (1, 0, 2, 3)) - np.transpose(vint, (0, 1, 3, 2)) + np.transpose(vint, (1, 0, 3, 2)))

    if optimize:
        # optimize MPO bond dimensions based on bipartite graph theory
        opchains = []
        # kinetic hopping terms \sum_{i,j} t_{i,j} a^{\dagger}_i a_j
        for i in range(L):
            for j in range(L):
                if i == j:
                    # diagonal hopping term
                    opchains.append(OpChain([MolecularOID.N], [0, 0], tkin[i, i], i))
                else:
                    (a, p), (b, q) = sorted([(i, MolecularOID.C), (j, MolecularOID.A)])
                    opchains.append(OpChain([p] + (b - a - 1)*[MolecularOID.Z] + [q],
                                            [0] + (b - a)*[int(p)] + [0], tkin[i, j], a))
        # interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k
        for i in range(L):
            for j in range(i + 1, L):  # i < j
                for k in range(L):
                    for l in range(k + 1, L):  # k < l
                        (a, p), (b, q), (c, r), (d, s) = sorted([(i, MolecularOID.C), (j, MolecularOID.C), (l, MolecularOID.A), (k, MolecularOID.A)])
                        if a == b:
                            assert b < c
                            if c == d:
                                # two number operators
                                oids  = [MolecularOID.N] + (c - b - 1)*[MolecularOID.I] + [MolecularOID.N]
                                qnums = (c - b + 2)*[0]
                            else:
                                # number operator at the beginning
                                oids  = [MolecularOID.N] + (c - b - 1)*[MolecularOID.I] + [r] + (d - c - 1)*[MolecularOID.Z] + [s]
                                qnums = (c - b + 1)*[0] + (d - c)*[int(r)] + [0]
                        elif b == c:
                            # number operator in the middle
                            oids  = [p] + (b - a - 1)*[MolecularOID.Z] + [MolecularOID.N] + (d - c - 1)*[MolecularOID.Z] + [s]
                            qnums = [0] + (d - a)*[int(p)] + [0]
                        elif c == d:
                            # number operator at the end
                            oids  = [p] + (b - a - 1)*[MolecularOID.Z] + [q] + (c - b - 1)*[MolecularOID.I] + [MolecularOID.N]
                            qnums = [0] + (b - a)*[int(p)] + (c - b + 1)*[0]
                        else:
                            # generic case: i, j, k, l pairwise different
                            oids  = [p] + (b - a - 1)*[MolecularOID.Z] + [q] + (c - b - 1)*[MolecularOID.I] + [r] + (d - c - 1)*[MolecularOID.Z] + [s]
                            qnums = [0] + (b - a)*[int(p)] + (c - b)*[int(p) + int(q)] + (d - c)*[-int(s)] + [0]
                        opchains.append(OpChain(oids, qnums, gint[i, j, k, l], a))
        graph = OpGraph.from_opchains(opchains, L, 0)

    else:
        # explicit construction (typically faster, but does not optimize cases
        # of zero coefficients, and is slightly sub-optimal close to boundary)
        assert L >= 4
        nodes = MolecularOpGraphNodes(L)
        graph = nodes.generate_graph()
        # kinetic hopping terms \sum_{i,j} t_{i,j} a^{\dagger}_i a_j
        for i in range(L):
            for j in range(L):
                _molecular_hamiltonian_graph_add_term(
                    graph, nodes, [(i, MolecularOID.C), (j, MolecularOID.A)], tkin[i, j])
        # interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k
        for i in range(L):
            for j in range(i + 1, L):  # i < j
                for k in range(L):
                    for l in range(k + 1, L):  # k < l
                        oplist = [(i, MolecularOID.C),
                                  (j, MolecularOID.C),
                                  (l, MolecularOID.A),
                                  (k, MolecularOID.A)]
                        _molecular_hamiltonian_graph_add_term(
                            graph, nodes, oplist, gint[i, j, k, l])

    # skip consistency check for larger L (would take very long)
    if L <= 12:
        assert graph.is_consistent()
    opmap = _molecular_hamiltonian_generate_operator_map()
    # convert to MPO
    mpo = MPO.from_opgraph([0, 1], graph, opmap, compute_nid_map=(not optimize))
    # store node information in MPO, to identify virtual bonds by creation and annihilation operators
    if not optimize:
        nodes.copy_nids(mpo)

    return mpo


def molecular_hamiltonian_orbital_gauge_transform(h: MPO, u, i: int):
    """
    Generate the left and right gauge transformation matrices corresponding to
    the single-orbital rotation matrix `u` applied to orbitals `i` and `i + 1`.
    """
    u = np.asarray(u)
    assert u.shape == (2, 2)
    assert np.allclose(u.conj().T @ u, np.identity(2))
    assert 0 <= i < h.nsites - 1
    # left gauge transformation matrix
    v_l = np.identity(h.bond_dims[i], dtype=u.dtype)
    # a^{\dagger}_i operators connected to right terminal
    if i in h.nids_a_dag_r:
        if i in h.nids_a_dag_r[i]:
            _, j0 = h.nid_map[h.nids_a_dag_r[i    ][i]]
            _, j1 = h.nid_map[h.nids_a_dag_r[i + 1][i]]
            v_l[j0, j0] = u[0, 0]
            v_l[j0, j1] = u[0, 1]
            v_l[j1, j0] = u[1, 0]
            v_l[j1, j1] = u[1, 1]
    # a_i operators connected to right terminal
    if i in h.nids_a_ann_r:
        if i in h.nids_a_ann_r[i]:
            _, j0 = h.nid_map[h.nids_a_ann_r[i    ][i]]
            _, j1 = h.nid_map[h.nids_a_ann_r[i + 1][i]]
            v_l[j0, j0] = u[0, 0].conj()
            v_l[j0, j1] = u[0, 1].conj()
            v_l[j1, j0] = u[1, 0].conj()
            v_l[j1, j1] = u[1, 1].conj()
    # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
    for k in range(i):
        if (k, i) in h.nids_a_dag_a_dag_r:
            if i in h.nids_a_dag_a_dag_r[k, i]:
                _, j0 = h.nid_map[h.nids_a_dag_a_dag_r[k, i    ][i]]
                _, j1 = h.nid_map[h.nids_a_dag_a_dag_r[k, i + 1][i]]
                v_l[j0, j0] = u[0, 0]
                v_l[j0, j1] = u[0, 1]
                v_l[j1, j0] = u[1, 0]
                v_l[j1, j1] = u[1, 1]
    for k in range(i + 2, h.nsites):
        if (i, k) in h.nids_a_dag_a_dag_r:
            if i in h.nids_a_dag_a_dag_r[i, k]:
                _, j0 = h.nid_map[h.nids_a_dag_a_dag_r[i,     k][i]]
                _, j1 = h.nid_map[h.nids_a_dag_a_dag_r[i + 1, k][i]]
                v_l[j0, j0] = u[0, 0]
                v_l[j0, j1] = u[0, 1]
                v_l[j1, j0] = u[1, 0]
                v_l[j1, j1] = u[1, 1]
    if (i, i + 1) in h.nids_a_dag_a_dag_r:
        if i in h.nids_a_dag_a_dag_r[i, i + 1]:
            _, j = h.nid_map[h.nids_a_dag_a_dag_r[i, i + 1][i]]
            v_l[j, j] = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    # a_i a_j operators connected to right terminal
    for k in range(i):
        if (i, k) in h.nids_a_ann_a_ann_r:
            if i in h.nids_a_ann_a_ann_r[i, k]:
                _, j0 = h.nid_map[h.nids_a_ann_a_ann_r[i,     k][i]]
                _, j1 = h.nid_map[h.nids_a_ann_a_ann_r[i + 1, k][i]]
                v_l[j0, j0] = u[0, 0].conj()
                v_l[j0, j1] = u[0, 1].conj()
                v_l[j1, j0] = u[1, 0].conj()
                v_l[j1, j1] = u[1, 1].conj()
    for k in range(i + 2, h.nsites):
        if (k, i) in h.nids_a_ann_a_ann_r:
            if i in h.nids_a_ann_a_ann_r[k, i]:
                _, j0 = h.nid_map[h.nids_a_ann_a_ann_r[k, i    ][i]]
                _, j1 = h.nid_map[h.nids_a_ann_a_ann_r[k, i + 1][i]]
                v_l[j0, j0] = u[0, 0].conj()
                v_l[j0, j1] = u[0, 1].conj()
                v_l[j1, j0] = u[1, 0].conj()
                v_l[j1, j1] = u[1, 1].conj()
    if (i + 1, i) in h.nids_a_ann_a_ann_r:
        if i in h.nids_a_ann_a_ann_r[i + 1, i]:
            _, j = h.nid_map[h.nids_a_ann_a_ann_r[i + 1, i][i]]
            v_l[j, j] = (u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]).conj()
    # a^{\dagger}_i a_j operators connected to right terminal
    for k in list(range(i)) + list(range(i + 2, h.nsites)):
        if (i, k) in h.nids_a_dag_a_ann_r:
            if i in h.nids_a_dag_a_ann_r[i, k]:
                _, j0 = h.nid_map[h.nids_a_dag_a_ann_r[i,     k][i]]
                _, j1 = h.nid_map[h.nids_a_dag_a_ann_r[i + 1, k][i]]
                v_l[j0, j0] = u[0, 0]
                v_l[j0, j1] = u[0, 1]
                v_l[j1, j0] = u[1, 0]
                v_l[j1, j1] = u[1, 1]
        if (k, i) in h.nids_a_dag_a_ann_r:
            if i in h.nids_a_dag_a_ann_r[k, i]:
                _, j0 = h.nid_map[h.nids_a_dag_a_ann_r[k, i    ][i]]
                _, j1 = h.nid_map[h.nids_a_dag_a_ann_r[k, i + 1][i]]
                v_l[j0, j0] = u[0, 0].conj()
                v_l[j0, j1] = u[0, 1].conj()
                v_l[j1, j0] = u[1, 0].conj()
                v_l[j1, j1] = u[1, 1].conj()
    if (i, i + 1) in h.nids_a_dag_a_ann_r:
        if i in h.nids_a_dag_a_ann_r[i, i + 1]:
            _, j00 = h.nid_map[h.nids_a_dag_a_ann_r[i,     i    ][i]]
            _, j01 = h.nid_map[h.nids_a_dag_a_ann_r[i,     i + 1][i]]
            _, j10 = h.nid_map[h.nids_a_dag_a_ann_r[i + 1, i    ][i]]
            _, j11 = h.nid_map[h.nids_a_dag_a_ann_r[i + 1, i + 1][i]]
            v_l[j00, j00] = abs(u[0, 0])**2
            v_l[j00, j01] = u[0, 0] * u[0, 1].conj()
            v_l[j00, j10] = u[0, 1] * u[0, 0].conj()
            v_l[j00, j11] = abs(u[0, 1])**2
            v_l[j01, j00] = u[0, 0] * u[1, 0].conj()
            v_l[j01, j01] = u[0, 0] * u[1, 1].conj()
            v_l[j01, j10] = u[0, 1] * u[1, 0].conj()
            v_l[j01, j11] = u[0, 1] * u[1, 1].conj()
            v_l[j10, j00] = u[1, 0] * u[0, 0].conj()
            v_l[j10, j01] = u[1, 0] * u[0, 1].conj()
            v_l[j10, j10] = u[1, 1] * u[0, 0].conj()
            v_l[j10, j11] = u[1, 1] * u[0, 1].conj()
            v_l[j11, j00] = abs(u[1, 0])**2
            v_l[j11, j01] = u[1, 0] * u[1, 1].conj()
            v_l[j11, j10] = u[1, 1] * u[1, 0].conj()
            v_l[j11, j11] = abs(u[1, 1])**2
    assert np.allclose(v_l.conj().T @ v_l, np.identity(v_l.shape[1]))

    # right gauge transformation matrix
    v_r = np.identity(h.bond_dims[i + 2], dtype=u.dtype)
    # a^{\dagger}_i operators connected to left terminal
    if i in h.nids_a_dag_l:
        if i + 2 in h.nids_a_dag_l[i]:
            _, j0 = h.nid_map[h.nids_a_dag_l[i    ][i + 2]]
            _, j1 = h.nid_map[h.nids_a_dag_l[i + 1][i + 2]]
            v_r[j0, j0] = u[0, 0]
            v_r[j0, j1] = u[0, 1]
            v_r[j1, j0] = u[1, 0]
            v_r[j1, j1] = u[1, 1]
    # a_i operators connected to left terminal
    if i in h.nids_a_ann_l:
        if i + 2 in h.nids_a_ann_l[i]:
            _, j0 = h.nid_map[h.nids_a_ann_l[i    ][i + 2]]
            _, j1 = h.nid_map[h.nids_a_ann_l[i + 1][i + 2]]
            v_r[j0, j0] = u[0, 0].conj()
            v_r[j0, j1] = u[0, 1].conj()
            v_r[j1, j0] = u[1, 0].conj()
            v_r[j1, j1] = u[1, 1].conj()
    # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
    for k in range(i):
        if (k, i) in h.nids_a_dag_a_dag_l:
            if i + 2 in h.nids_a_dag_a_dag_l[k, i]:
                _, j0 = h.nid_map[h.nids_a_dag_a_dag_l[k, i    ][i + 2]]
                _, j1 = h.nid_map[h.nids_a_dag_a_dag_l[k, i + 1][i + 2]]
                v_r[j0, j0] = u[0, 0]
                v_r[j0, j1] = u[0, 1]
                v_r[j1, j0] = u[1, 0]
                v_r[j1, j1] = u[1, 1]
    for k in range(i + 2, h.nsites):
        if (i, k) in h.nids_a_dag_a_dag_l:
            if i + 2 in h.nids_a_dag_a_dag_l[i, k]:
                _, j0 = h.nid_map[h.nids_a_dag_a_dag_l[i,     k][i + 2]]
                _, j1 = h.nid_map[h.nids_a_dag_a_dag_l[i + 1, k][i + 2]]
                v_r[j0, j0] = u[0, 0]
                v_r[j0, j1] = u[0, 1]
                v_r[j1, j0] = u[1, 0]
                v_r[j1, j1] = u[1, 1]
    if (i, i + 1) in h.nids_a_dag_a_dag_l:
        if i + 2 in h.nids_a_dag_a_dag_l[i, i + 1]:
            _, j = h.nid_map[h.nids_a_dag_a_dag_l[i, i + 1][i + 2]]
            v_r[j, j] = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    # a_i a_j operators connected to left terminal
    for k in range(i):
        if (i, k) in h.nids_a_ann_a_ann_l:
            if i + 2 in h.nids_a_ann_a_ann_l[i, k]:
                _, j0 = h.nid_map[h.nids_a_ann_a_ann_l[i,     k][i + 2]]
                _, j1 = h.nid_map[h.nids_a_ann_a_ann_l[i + 1, k][i + 2]]
                v_r[j0, j0] = u[0, 0].conj()
                v_r[j0, j1] = u[0, 1].conj()
                v_r[j1, j0] = u[1, 0].conj()
                v_r[j1, j1] = u[1, 1].conj()
    for k in range(i + 2, h.nsites):
        if (k, i) in h.nids_a_ann_a_ann_l:
            if i + 2 in h.nids_a_ann_a_ann_l[k, i]:
                _, j0 = h.nid_map[h.nids_a_ann_a_ann_l[k, i    ][i + 2]]
                _, j1 = h.nid_map[h.nids_a_ann_a_ann_l[k, i + 1][i + 2]]
                v_r[j0, j0] = u[0, 0].conj()
                v_r[j0, j1] = u[0, 1].conj()
                v_r[j1, j0] = u[1, 0].conj()
                v_r[j1, j1] = u[1, 1].conj()
    if (i + 1, i) in h.nids_a_ann_a_ann_l:
        if i + 2 in h.nids_a_ann_a_ann_l[i + 1, i]:
            _, j = h.nid_map[h.nids_a_ann_a_ann_l[i + 1, i][i + 2]]
            v_r[j, j] = (u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]).conj()
    # a^{\dagger}_i a_j operators connected to left terminal
    for k in list(range(i)) + list(range(i + 2, h.nsites)):
        if (i, k) in h.nids_a_dag_a_ann_l:
            if i + 2 in h.nids_a_dag_a_ann_l[i, k]:
                _, j0 = h.nid_map[h.nids_a_dag_a_ann_l[i,     k][i + 2]]
                _, j1 = h.nid_map[h.nids_a_dag_a_ann_l[i + 1, k][i + 2]]
                v_r[j0, j0] = u[0, 0]
                v_r[j0, j1] = u[0, 1]
                v_r[j1, j0] = u[1, 0]
                v_r[j1, j1] = u[1, 1]
        if (k, i) in h.nids_a_dag_a_ann_l:
            if i + 2 in h.nids_a_dag_a_ann_l[k, i]:
                _, j0 = h.nid_map[h.nids_a_dag_a_ann_l[k, i    ][i + 2]]
                _, j1 = h.nid_map[h.nids_a_dag_a_ann_l[k, i + 1][i + 2]]
                v_r[j0, j0] = u[0, 0].conj()
                v_r[j0, j1] = u[0, 1].conj()
                v_r[j1, j0] = u[1, 0].conj()
                v_r[j1, j1] = u[1, 1].conj()
    if (i, i + 1) in h.nids_a_dag_a_ann_l:
        if i + 2 in h.nids_a_dag_a_ann_l[i, i + 1]:
            _, j00 = h.nid_map[h.nids_a_dag_a_ann_l[i,     i    ][i + 2]]
            _, j01 = h.nid_map[h.nids_a_dag_a_ann_l[i,     i + 1][i + 2]]
            _, j10 = h.nid_map[h.nids_a_dag_a_ann_l[i + 1, i    ][i + 2]]
            _, j11 = h.nid_map[h.nids_a_dag_a_ann_l[i + 1, i + 1][i + 2]]
            v_r[j00, j00] = abs(u[0, 0])**2
            v_r[j00, j01] = u[0, 0] * u[0, 1].conj()
            v_r[j00, j10] = u[0, 1] * u[0, 0].conj()
            v_r[j00, j11] = abs(u[0, 1])**2
            v_r[j01, j00] = u[0, 0] * u[1, 0].conj()
            v_r[j01, j01] = u[0, 0] * u[1, 1].conj()
            v_r[j01, j10] = u[0, 1] * u[1, 0].conj()
            v_r[j01, j11] = u[0, 1] * u[1, 1].conj()
            v_r[j10, j00] = u[1, 0] * u[0, 0].conj()
            v_r[j10, j01] = u[1, 0] * u[0, 1].conj()
            v_r[j10, j10] = u[1, 1] * u[0, 0].conj()
            v_r[j10, j11] = u[1, 1] * u[0, 1].conj()
            v_r[j11, j00] = abs(u[1, 0])**2
            v_r[j11, j01] = u[1, 0] * u[1, 1].conj()
            v_r[j11, j10] = u[1, 1] * u[1, 0].conj()
            v_r[j11, j11] = abs(u[1, 1])**2
    assert np.allclose(v_r.conj().T @ v_r, np.identity(v_r.shape[1]))

    return v_l, v_r


def molecular_spin_hamiltonian_mpo(tkin, vint, optimize=True) -> MPO:
    r"""
    Construct a molecular spin Hamiltonian as MPO, taking the electronic spin into account and
    using physicists' convention for the interaction term (note ordering of k and \ell):

    .. math::

        H = \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma} + \\frac{1}{2} \sum_{i,j,k,\ell,\sigma,\tau} v_{i,j,k,\ell} a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{\ell,\tau} a_{k,\sigma}
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    L = tkin.shape[0]
    assert tkin.shape == (L, L)
    assert vint.shape == (L, L, L, L)

    # physical particle number and spin quantum numbers (encoded as single integer)
    qN = [0,  1,  1,  2]
    qS = [0, -1,  1,  0]
    qd = [_encode_quantum_number_pair(q[0], q[1]) for q in zip(qN, qS)]
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
        Id =  0
        IC =  1
        IA =  2
        IN =  3
        CI =  4
        CC =  5
        CA =  6
        CN =  7
        CZ =  8
        AI =  9
        AC = 10
        AA = 11
        AN = 12
        AZ = 13
        NI = 14
        NC = 15
        NA = 16
        NN = 17
        NZ = 18
        ZC = 19
        ZA = 20
        ZN = 21
        ZZ = 22
    opmap = {
        OID.Id: np.identity(4),
        OID.IC: np.kron(id2,   a_dag),
        OID.IA: np.kron(id2,   a_ann),
        OID.IN: np.kron(id2,   numop),
        OID.CI: np.kron(a_dag, id2  ),
        OID.CC: np.kron(a_dag, a_dag),
        OID.CA: np.kron(a_dag, a_ann),
        OID.CN: np.kron(a_dag, numop),
        OID.CZ: np.kron(a_dag, Z    ),
        OID.AI: np.kron(a_ann, id2  ),
        OID.AC: np.kron(a_ann, a_dag),
        OID.AA: np.kron(a_ann, a_ann),
        OID.AN: np.kron(a_ann, numop),
        OID.AZ: np.kron(a_ann, Z    ),
        OID.NI: np.kron(numop, id2  ),
        OID.NC: np.kron(numop, a_dag),
        OID.NA: np.kron(numop, a_ann),
        OID.NN: np.kron(numop, numop),
        OID.NZ: np.kron(numop, Z    ),
        OID.ZC: np.kron(Z,     a_dag),
        OID.ZA: np.kron(Z,     a_ann),
        OID.ZN: np.kron(Z,     numop),
        OID.ZZ: np.kron(Z,     Z    ),
    }

    # interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau,\mu,\nu} v_{i,j,k,l} \delta_{\sigma,\mu} \delta_{\tau,\nu} a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\nu} a_{k,\mu}:
    # can anti-commute fermionic operators such that (i,\sigma) < (j,\tau) and (k,\mu) < (l,\nu)
    gint0 = 0.5 * (vint                             + np.transpose(vint, (1, 0, 3, 2)))
    gint1 = 0.5 * (np.transpose(vint, (1, 0, 2, 3)) + np.transpose(vint, (0, 1, 3, 2)))

    def get_vint_coeff(spatialidx, spinidx):
        valid = False
        coeff = 0
        if (spinidx[0] == spinidx[2]) and (spinidx[1] == spinidx[3]):
            coeff += gint0[spatialidx]
            valid = True
        if (spinidx[0] == spinidx[3]) and (spinidx[1] == spinidx[2]):
            coeff -= gint1[spatialidx]
            valid = True
        return coeff, valid

    if optimize:
        # optimize MPO bond dimensions based on bipartite graph theory

        class OIDSingle(IntEnum):
            A = -1
            I =  0
            C =  1
            N =  2
            Z =  3

        # map a single OID pair to a combined OID
        oid_single_pair_map = {
            (OIDSingle.I, OIDSingle.I): OID.Id,
            (OIDSingle.I, OIDSingle.C): OID.IC,
            (OIDSingle.I, OIDSingle.A): OID.IA,
            (OIDSingle.I, OIDSingle.N): OID.IN,
            (OIDSingle.C, OIDSingle.I): OID.CI,
            (OIDSingle.C, OIDSingle.C): OID.CC,
            (OIDSingle.C, OIDSingle.A): OID.CA,
            (OIDSingle.C, OIDSingle.N): OID.CN,
            (OIDSingle.C, OIDSingle.Z): OID.CZ,
            (OIDSingle.A, OIDSingle.I): OID.AI,
            (OIDSingle.A, OIDSingle.C): OID.AC,
            (OIDSingle.A, OIDSingle.A): OID.AA,
            (OIDSingle.A, OIDSingle.N): OID.AN,
            (OIDSingle.A, OIDSingle.Z): OID.AZ,
            (OIDSingle.N, OIDSingle.I): OID.NI,
            (OIDSingle.N, OIDSingle.C): OID.NC,
            (OIDSingle.N, OIDSingle.A): OID.NA,
            (OIDSingle.N, OIDSingle.N): OID.NN,
            (OIDSingle.N, OIDSingle.Z): OID.NZ,
            (OIDSingle.Z, OIDSingle.C): OID.ZC,
            (OIDSingle.Z, OIDSingle.A): OID.ZA,
            (OIDSingle.Z, OIDSingle.N): OID.ZN,
            (OIDSingle.Z, OIDSingle.Z): OID.ZZ,
        }

        def convert_to_spin_opchain(opchain_single: OpChain):
            assert opchain_single.qnums[ 0] == 0
            assert opchain_single.qnums[-1] == 0
            if opchain_single.istart % 2 == 1:
                opchain_single.oids.insert(0, OIDSingle.I)
                opchain_single.qnums.insert(0, 0)
                opchain_single.istart -= 1
            if opchain_single.length % 2 == 1:
                opchain_single.oids.append(OIDSingle.I)
                opchain_single.qnums.append(0)
            assert opchain_single.length % 2 == 0
            oids = [oid_single_pair_map[pair] for pair in zip(opchain_single.oids[0::2], opchain_single.oids[1::2])]
            qnums = [0]
            qspin = 0
            for i in range(opchain_single.length // 2):
                # determine spin quantum number from particle quantum numbers
                qspin -= (opchain_single.qnums[2*i] - 2*opchain_single.qnums[2*i + 1] + opchain_single.qnums[2*i + 2])
                qnums.append(_encode_quantum_number_pair(opchain_single.qnums[2*(i + 1)], qspin))
            assert qnums[-1] == 0
            return OpChain(oids, qnums, opchain_single.coeff, opchain_single.istart // 2)

        opchains = []
        # kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
        for i in range(2*L):
            for j in range(2*L):
                if (i - j) % 2 != 0:
                    continue
                if i == j:
                    # diagonal hopping term
                    opchains.append(convert_to_spin_opchain(OpChain([OIDSingle.N], [0, 0], tkin[i//2, i//2], i)))
                else:
                    (a, p), (b, q) = sorted([(i, OIDSingle.C), (j, OIDSingle.A)])
                    opchains.append(convert_to_spin_opchain(
                        OpChain([p] + (b - a - 1)*[OIDSingle.Z] + [q],
                                [0] + (b - a)*[int(p)] + [0], tkin[i//2, j//2], a)))
        # interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau} v_{i,j,k,l} a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\tau} a_{k,\sigma}
        for i in range(2*L):
            for j in range(i + 1, 2*L):  # i < j
                for k in range(2*L):
                    for l in range(k + 1, 2*L):  # k < l
                        coeff, valid = get_vint_coeff((i // 2, j // 2, k // 2, l // 2),
                                                      (i %  2, j %  2, k %  2, l %  2))
                        if not valid:
                            continue
                        (a, p), (b, q), (c, r), (d, s) = sorted([(i, OIDSingle.C), (j, OIDSingle.C), (l, OIDSingle.A), (k, OIDSingle.A)])
                        if a == b:
                            assert b < c
                            if c == d:
                                # two number operators
                                oids  = [OIDSingle.N] + (c - b - 1)*[OIDSingle.I] + [OIDSingle.N]
                                qnums = (c - b + 2)*[0]
                            else:
                                # number operator at the beginning
                                oids  = [OIDSingle.N] + (c - b - 1)*[OIDSingle.I] + [r] + (d - c - 1)*[OIDSingle.Z] + [s]
                                qnums = (c - b + 1)*[0] + (d - c)*[int(r)] + [0]
                        elif b == c:
                            # number operator in the middle
                            oids  = [p] + (b - a - 1)*[OIDSingle.Z] + [OIDSingle.N] + (d - c - 1)*[OIDSingle.Z] + [s]
                            qnums = [0] + (d - a)*[int(p)] + [0]
                        elif c == d:
                            # number operator at the end
                            oids  = [p] + (b - a - 1)*[OIDSingle.Z] + [q] + (c - b - 1)*[OIDSingle.I] + [OIDSingle.N]
                            qnums = [0] + (b - a)*[int(p)] + (c - b + 1)*[0]
                        else:
                            # generic case: i, j, k, l pairwise different
                            oids  = [p] + (b - a - 1)*[OIDSingle.Z] + [q] + (c - b - 1)*[OIDSingle.I] + [r] + (d - c - 1)*[OIDSingle.Z] + [s]
                            qnums = [0] + (b - a)*[int(p)] + (c - b)*[int(p) + int(q)] + (d - c)*[-int(s)] + [0]
                        opchains.append(convert_to_spin_opchain(OpChain(oids, qnums, coeff, a)))
        graph = OpGraph.from_opchains(opchains, L, OID.Id)

    else:
        raise NotImplementedError

    # skip consistency check for larger L (would take very long)
    if L <= 10:
        assert graph.is_consistent()
    # convert to MPO
    return MPO.from_opgraph(qd, graph, opmap, compute_nid_map=(not optimize))


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
