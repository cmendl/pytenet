import copy
import itertools
from collections.abc import Sequence, Mapping
from enum import IntEnum
import numpy as np
from .mpo import MPO
from .opchain import OpChain
from .autop import AutOpNode, AutOpEdge, AutOp
from .opgraph import OpGraphNode, OpGraphEdge, OpGraph

__all__ = ['ising_mpo', 'heisenberg_xxz_mpo', 'heisenberg_xxz_spin1_mpo',
           'bose_hubbard_mpo', 'fermi_hubbard_mpo',
           'linear_fermionic_mpo', 'quadratic_fermionic_mpo',
           'molecular_hamiltonian_mpo', 'molecular_hamiltonian_orbital_gauge_transform',
           'spin_molecular_hamiltonian_mpo']


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


def linear_fermionic_mpo(coeff, ftype: str) -> MPO:
    """
    Represent a sum of fermionic creation or annihilation operators of the following form as MPO:

    .. math::

        op = \sum_{i=1}^L coeff_i a^{\dagger}_i \text{ or } op = \sum_{i=1}^L coeff_i a_i
    """
    L = len(coeff)

    use_creation_op = (ftype in ['c', 'create', 'creation'])

    # creation and annihilation operators
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    Z = np.array([[1., 0.], [0., -1.]])
    # operator map
    class OID(IntEnum):
        A = -1
        I =  0
        C =  1
        Z =  2
    opmap = {
        OID.A: a_ann,
        OID.I: np.identity(2),
        OID.C: a_dag,
        OID.Z: Z
    }

    # construct operator graph
    nid_next = 0
    # identity and Z strings from the left and right
    identity_l = {}
    z_string_r = {}
    for i in range(L):
        identity_l[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    for i in range(1, L + 1):
        z_string_r[i] = OpGraphNode(nid_next, [], [], 1 if use_creation_op else -1)
        nid_next += 1
    # initialize graph with nodes
    graph = OpGraph(list(identity_l.values()) +
                    list(z_string_r.values()),
                    [], [identity_l[0].nid, z_string_r[L].nid])
    # edges
    eid_next = 0
    # identities
    for i in range(L - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, identity_l[i + 1].nid], [(OID.I, 1.)]))
        eid_next += 1
    # Z strings
    for i in range(1, L):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [z_string_r[i].nid, z_string_r[i + 1].nid], [(OID.Z, 1.)]))
        eid_next += 1
    # creation or annihilation operators
    for i in range(L):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, z_string_r[i + 1].nid], [(OID.C if use_creation_op else OID.A, coeff[i])]))
        eid_next += 1
    assert graph.is_consistent()

    # convert to MPO
    return MPO.from_opgraph([0, 1], graph, opmap)


def quadratic_fermionic_mpo(coeffc, coeffa) -> MPO:
    """
    Represent a product of sums of fermionic creation or annihilation operators of the following form as MPO:

    .. math::

        op = (\sum_{i=1}^L coeffc_i a^{\dagger}_i) (\sum_{j=1}^L coeffa_j a_j)
    """
    assert len(coeffc) == len(coeffa)
    L = len(coeffc)

    # creation and annihilation operators
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    Z = np.array([[1., 0.], [0., -1.]])
    # operator map
    class OID(IntEnum):
        A = -1
        I =  0
        C =  1
        N =  2
        Z =  3
    opmap = {
        OID.A: a_ann,
        OID.I: np.identity(2),
        OID.C: a_dag,
        OID.N: numop,
        OID.Z: Z
    }

    # construct operator graph
    nid_next = 0
    # identity chains from the left and right
    identity_l = {}
    identity_r = {}
    for i in range(L):
        identity_l[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    for i in range(1, L + 1):
        identity_r[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    # nodes connecting creation and annihilation operators
    ca_nodes = {}
    ac_nodes = {}
    for i in range(1, L):
        ca_nodes[i] = OpGraphNode(nid_next, [], [], 1)
        nid_next += 1
    for i in range(1, L):
        ac_nodes[i] = OpGraphNode(nid_next, [], [], -1)
        nid_next += 1
    # initialize graph with nodes
    graph = OpGraph(list(identity_l.values()) +
                    list(identity_r.values()) +
                    list(ca_nodes.values()) +
                    list(ac_nodes.values()),
                    [], [identity_l[0].nid, identity_r[L].nid])
    # edges
    eid_next = 0
    # identities
    for i in range(L - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, identity_l[i + 1].nid], [(OID.I, 1.)]))
        eid_next += 1
    for i in range(1, L):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_r[i].nid, identity_r[i + 1].nid], [(OID.I, 1.)]))
        eid_next += 1
    # Z strings
    for i in range(1, L - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ca_nodes[i].nid, ca_nodes[i + 1].nid], [(OID.Z, 1.)]))
        eid_next += 1
    for i in range(1, L - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ac_nodes[i].nid, ac_nodes[i + 1].nid], [(OID.Z, 1.)]))
        eid_next += 1
    # number operators
    for i in range(L):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, identity_r[i + 1].nid], [(OID.N, coeffc[i]*coeffa[i])]))
        eid_next += 1
    # creation and annihilation operators
    for i in range(L - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, ca_nodes[i + 1].nid], [(OID.C, coeffc[i])]))
        eid_next += 1
    for i in range(1, L):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ca_nodes[i].nid, identity_r[i + 1].nid], [(OID.A, coeffa[i])]))
        eid_next += 1
    for i in range(L - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, ac_nodes[i + 1].nid], [(OID.A, coeffa[i])]))
        eid_next += 1
    for i in range(1, L):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ac_nodes[i].nid, identity_r[i + 1].nid], [(OID.C, coeffc[i])]))
        eid_next += 1
    assert graph.is_consistent()

    # convert to MPO
    return MPO.from_opgraph([0, 1], graph, opmap)


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
            (i, oid0), (j, oid1) = oplist
            if (oid0, oid1) == (MolecularOID.C, MolecularOID.C):
                i, j = sorted((i, j))
                return self.a_dag_a_dag_l[i, j] if connection == "left" else self.a_dag_a_dag_r[i, j]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.A):
                i, j = sorted((i, j), reverse=True)
                return self.a_ann_a_ann_l[i, j] if connection == "left" else self.a_ann_a_ann_r[i, j]
            if (oid0, oid1) == (MolecularOID.C, MolecularOID.A):
                return self.a_dag_a_ann_l[i, j] if connection == "left" else self.a_dag_a_ann_r[i, j]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.C):
                return self.a_dag_a_ann_l[j, i] if connection == "left" else self.a_dag_a_ann_r[j, i]
            raise KeyError(f"nodes for OIDs ({oid0}, {oid1}) and (i, j) == ({i}, {j}) do not exist")
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
        (i, oid0), (j, oid1) = oplist
        if i == j:
            # expecting number operator
            assert (oid0, oid1) == (MolecularOID.A, MolecularOID.C)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid], [(MolecularOID.N, coeff)]))
        else:
            assert i < j
            if j <= L//2:
                nodes_l = nodes.get([(i, oid0)], "left")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes.identity_r[j + 1].nid], [(oid1, coeff)]))
            elif i >= L//2:
                nodes_r = nodes.get([(j, oid1)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes_r[i + 1].nid], [(oid0, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid0)], "left")
                nodes_r = nodes.get([(j, oid1)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[L//2].nid, nodes_r[L//2 + 1].nid], [(MolecularOID.Z, coeff)]))
    elif len(oplist) == 4:
        (i, oid0), (j, oid1), (k, oid2), (l, oid3) = oplist
        if j == k:
            # expecting number operator
            assert (oid1, oid2) == (MolecularOID.A, MolecularOID.C)
            nodes_l = nodes.get([(i, oid0)], "left")
            nodes_r = nodes.get([(l, oid3)], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid], [(MolecularOID.N, coeff)]))
        elif k <= L//2:
            nodes_l = nodes.get([(i, oid0), (j, oid1)], "left")
            if k == l:
                # expecting number operator
                assert (oid2, oid3) == (MolecularOID.A, MolecularOID.C)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes.identity_r[k + 1].nid], [(MolecularOID.N, coeff)]))
            else:
                nodes_r = nodes.get([(l, oid3)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes_r[k + 1].nid], [(oid2, coeff)]))
        elif j >= L//2:
            nodes_r = nodes.get([(k, oid2), (l, oid3)], "right")
            if i == j:
                # expecting number operator
                assert (oid0, oid1) == (MolecularOID.A, MolecularOID.C)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[j].nid, nodes_r[j + 1].nid], [(MolecularOID.N, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid0)], "left")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid], [(oid1, coeff)]))
        else:
            nodes_l = nodes.get([(i, oid0), (j, oid1)], "left")
            nodes_r = nodes.get([(k, oid2), (l, oid3)], "right")
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


class SpinMolecularOID(IntEnum):
    """
    Local operator IDs for a molecular Hamiltonian using a spin orbital basis.
    """
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


class SpinOperatorConverter:
    """
    Local operator conversion when transitioning from a spatial to a spin orbital basis.
    """

    # map a single OID pair to a combined OID
    oid_single_pair_map = {
        (MolecularOID.I, MolecularOID.I): SpinMolecularOID.Id,
        (MolecularOID.I, MolecularOID.C): SpinMolecularOID.IC,
        (MolecularOID.I, MolecularOID.A): SpinMolecularOID.IA,
        (MolecularOID.I, MolecularOID.N): SpinMolecularOID.IN,
        (MolecularOID.C, MolecularOID.I): SpinMolecularOID.CI,
        (MolecularOID.C, MolecularOID.C): SpinMolecularOID.CC,
        (MolecularOID.C, MolecularOID.A): SpinMolecularOID.CA,
        (MolecularOID.C, MolecularOID.N): SpinMolecularOID.CN,
        (MolecularOID.C, MolecularOID.Z): SpinMolecularOID.CZ,
        (MolecularOID.A, MolecularOID.I): SpinMolecularOID.AI,
        (MolecularOID.A, MolecularOID.C): SpinMolecularOID.AC,
        (MolecularOID.A, MolecularOID.A): SpinMolecularOID.AA,
        (MolecularOID.A, MolecularOID.N): SpinMolecularOID.AN,
        (MolecularOID.A, MolecularOID.Z): SpinMolecularOID.AZ,
        (MolecularOID.N, MolecularOID.I): SpinMolecularOID.NI,
        (MolecularOID.N, MolecularOID.C): SpinMolecularOID.NC,
        (MolecularOID.N, MolecularOID.A): SpinMolecularOID.NA,
        (MolecularOID.N, MolecularOID.N): SpinMolecularOID.NN,
        (MolecularOID.N, MolecularOID.Z): SpinMolecularOID.NZ,
        (MolecularOID.Z, MolecularOID.C): SpinMolecularOID.ZC,
        (MolecularOID.Z, MolecularOID.A): SpinMolecularOID.ZA,
        (MolecularOID.Z, MolecularOID.N): SpinMolecularOID.ZN,
        (MolecularOID.Z, MolecularOID.Z): SpinMolecularOID.ZZ,
    }

    @classmethod
    def to_spin_opchain(cls, opchain_single: OpChain) -> OpChain:
        """
        Convert an operator chain with interleaved spin-up and spin-down
        local operators to an operator chain using a spin orbital basis.
        """
        assert opchain_single.qnums[ 0] == 0
        assert opchain_single.qnums[-1] == 0
        if opchain_single.istart % 2 == 1:
            opchain_single.oids.insert(0, MolecularOID.I)
            opchain_single.qnums.insert(0, 0)
            opchain_single.istart -= 1
        if opchain_single.length % 2 == 1:
            opchain_single.oids.append(MolecularOID.I)
            opchain_single.qnums.append(0)
        assert opchain_single.length % 2 == 0
        oids = [cls.oid_single_pair_map[pair]
                for pair in zip(opchain_single.oids[0::2], opchain_single.oids[1::2])]
        qnums = [0]
        qspin = 0
        for i in range(opchain_single.length // 2):
            # determine spin quantum number from particle quantum numbers
            qspin -= (opchain_single.qnums[2*i] - 2*opchain_single.qnums[2*i + 1] + opchain_single.qnums[2*i + 2])
            qnums.append(_encode_quantum_number_pair(opchain_single.qnums[2*(i + 1)], qspin))
        assert qnums[-1] == 0
        return OpChain(oids, qnums, opchain_single.coeff, opchain_single.istart // 2)

    @classmethod
    def to_spin_operator(cls, oplist: Sequence[tuple], even_parity_left: bool, even_parity_right: bool) -> SpinMolecularOID:
        """
        Convert a list of local creation and annihilation operators of the form [(spin_a, oid_a), ...]
        to the corresponding operator using a spin orbital basis.
        """
        oplist = sorted(oplist)
        if len(oplist) == 1:
            (spin, oid) = oplist[0]
            if spin == 0:
                transfer = MolecularOID.I if even_parity_right else MolecularOID.Z
                return cls.oid_single_pair_map[oid, transfer]
            if spin == 1:
                transfer = MolecularOID.I if even_parity_left else MolecularOID.Z
                return cls.oid_single_pair_map[transfer, oid]
            raise ValueError("'spin' must either be 0 (spin-up) or 1 (spin-down)")
        if len(oplist) == 2:
            (spin_a, oid_a), (spin_b, oid_b) = oplist
            if spin_a == spin_b:
                assert ((oid_a, oid_b) == (MolecularOID.C, MolecularOID.A)
                     or (oid_a, oid_b) == (MolecularOID.A, MolecularOID.C))
                if spin_a == 0:
                    return SpinMolecularOID.NI if even_parity_right else SpinMolecularOID.NZ
                if spin_a == 1:
                    return SpinMolecularOID.IN if even_parity_left else SpinMolecularOID.ZN
                raise ValueError("'spin' must either be 0 (spin-up) or 1 (spin-down)")
            assert (spin_a, spin_b) == (0, 1)  # must be sorted
            return cls.oid_single_pair_map[oid_a, oid_b]
        if len(oplist) == 3:
            # note: oplist is sorted
            (spin_a, oid_a), (spin_b, oid_b), (spin_c, oid_c) = oplist
            if (spin_a, spin_b, spin_c) == (0, 0, 1):
                assert ((oid_a, oid_b) == (MolecularOID.C, MolecularOID.A)
                     or (oid_a, oid_b) == (MolecularOID.A, MolecularOID.C))
                return cls.oid_single_pair_map[MolecularOID.N, oid_c]
            if (spin_a, spin_b, spin_c) == (0, 1, 1):
                assert ((oid_b, oid_c) == (MolecularOID.C, MolecularOID.A)
                     or (oid_b, oid_c) == (MolecularOID.A, MolecularOID.C))
                return cls.oid_single_pair_map[oid_a, MolecularOID.N]
            raise ValueError("'spin' must either be 0 (spin-up) or 1 (spin-down)")
        if len(oplist) == 4:
            # note: oplist is sorted
            (spin_a, oid_a), (spin_b, oid_b), (spin_c, oid_c), (spin_d, oid_d) = oplist
            assert (spin_a, spin_b, spin_c, spin_d) == (0, 0, 1, 1)
            assert ((oid_a, oid_b) == (MolecularOID.C, MolecularOID.A)
                 or (oid_a, oid_b) == (MolecularOID.A, MolecularOID.C))
            assert ((oid_c, oid_d) == (MolecularOID.C, MolecularOID.A)
                 or (oid_c, oid_d) == (MolecularOID.A, MolecularOID.C))
            return SpinMolecularOID.NN
        raise ValueError(f"'oplist' of length {len(oplist)} is invalid")


def _spin_molecular_hamiltonian_generate_operator_map():
    """
    Generate the local operator map for a molecular Hamiltonian using a spin orbital basis.
    """
    opmap_single = _molecular_hamiltonian_generate_operator_map()
    return { oid : np.kron(opmap_single[pair[0]], opmap_single[pair[1]])
             for pair, oid in SpinOperatorConverter.oid_single_pair_map.items() }


class SpinMolecularOpGraphNodes:
    """
    Operator graph nodes used for molecular Hamiltonian construction, assuming a spin orbital basis.
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
        # a^{\dagger}_{i,\sigma} operators connected to left terminal
        self.a_dag_l = {}
        for i, sigma in itertools.product(range(L - 1), (0, 1)):
            self.a_dag_l[i, sigma] = {}
            for j in range(i + 1, L):
                self.a_dag_l[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], _encode_quantum_number_pair( 1, [1, -1][sigma]))
                nid_next += 1
        # a_{i,\sigma} operators connected to left terminal
        self.a_ann_l = {}
        for i, sigma in itertools.product(range(L - 1), (0, 1)):
            self.a_ann_l[i, sigma] = {}
            for j in range(i + 1, L):
                self.a_ann_l[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], _encode_quantum_number_pair(-1, [-1, 1][sigma]))
                nid_next += 1
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to left terminal
        self.a_dag_a_dag_l = {}
        for i, sigma in itertools.product(range(L//2), (0, 1)):
            for j, tau in itertools.product(range(i, L//2), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                self.a_dag_a_dag_l[i, sigma, j, tau] = {}
                for k in range(j + 1, L//2 + 1):
                    self.a_dag_a_dag_l[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], _encode_quantum_number_pair( 2, [1, -1][sigma] + [1, -1][tau]))
                    nid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to left terminal
        self.a_ann_a_ann_l = {}
        for i, sigma in itertools.product(range(L//2), (0, 1)):
            for j, tau in itertools.product(range(i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                self.a_ann_a_ann_l[i, sigma, j, tau] = {}
                for k in range(i + 1, L//2 + 1):
                    self.a_ann_a_ann_l[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], _encode_quantum_number_pair(-2, [-1, 1][sigma] + [-1, 1][tau]))
                    nid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to left terminal
        self.a_dag_a_ann_l = {}
        for i, sigma in itertools.product(range(L//2), (0, 1)):
            for j, tau in itertools.product(range(L//2), (0, 1)):
                self.a_dag_a_ann_l[i, sigma, j, tau] = {}
                for k in range(max(i, j) + 1, L//2 + 1):
                    self.a_dag_a_ann_l[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], _encode_quantum_number_pair(0, [1, -1][sigma] + [-1, 1][tau]))
                    nid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to right terminal
        self.a_dag_r = {}
        for i, sigma in itertools.product(range(1, L), (0, 1)):
            self.a_dag_r[i, sigma] = {}
            for j in range(1, i + 1):
                self.a_dag_r[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], _encode_quantum_number_pair(-1, [-1, 1][sigma]))
                nid_next += 1
        # a_{i,\sigma} operators connected to right terminal
        self.a_ann_r = {}
        for i, sigma in itertools.product(range(1, L), (0, 1)):
            self.a_ann_r[i, sigma] = {}
            for j in range(1, i + 1):
                self.a_ann_r[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], _encode_quantum_number_pair( 1, [1, -1][sigma]))
                nid_next += 1
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to right terminal
        self.a_dag_a_dag_r = {}
        for i, sigma in itertools.product(range(L//2 + 1, L), (0, 1)):
            for j, tau in itertools.product(range(i, L), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                self.a_dag_a_dag_r[i, sigma, j, tau] = {}
                for k in range(L//2 + 1, i + 1):
                    self.a_dag_a_dag_r[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], _encode_quantum_number_pair(-2, [-1, 1][sigma] + [-1, 1][tau]))
                    nid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to right terminal
        self.a_ann_a_ann_r = {}
        for i, sigma in itertools.product(range(L//2 + 1, L), (0, 1)):
            for j, tau in itertools.product(range(L//2 + 1, i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                self.a_ann_a_ann_r[i, sigma, j, tau] = {}
                for k in range(L//2 + 1, j + 1):
                    self.a_ann_a_ann_r[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], _encode_quantum_number_pair( 2, [1, -1][sigma] + [1, -1][tau]))
                    nid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to right terminal
        self.a_dag_a_ann_r = {}
        for i, sigma in itertools.product(range(L//2 + 1, L), (0, 1)):
            for j, tau in itertools.product(range(L//2 + 1, L), (0, 1)):
                self.a_dag_a_ann_r[i, sigma, j, tau] = {}
                for k in range(L//2 + 1, min(i, j) + 1):
                    self.a_dag_a_ann_r[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], _encode_quantum_number_pair( 0, [-1, 1][sigma] + [1, -1][tau]))
                    nid_next += 1

    def get(self, oplist: Sequence[tuple], connection):
        """
        Retrieve nodes corresponding to a single or pairs of creation and annihilation operators.
        """
        if len(oplist) == 1:
            (i, sigma, oid) = oplist[0]
            if oid == MolecularOID.C:
                return self.a_dag_l[i, sigma] if connection == "left" else self.a_dag_r[i, sigma]
            if oid == MolecularOID.A:
                return self.a_ann_l[i, sigma] if connection == "left" else self.a_ann_r[i, sigma]
            raise KeyError(f"nodes for OID {oid} and (i, sigma) == ({i}, {sigma}) do not exist")
        if len(oplist) == 2:
            (i, sigma, oid0), (j, tau, oid1) = oplist
            if (oid0, oid1) == (MolecularOID.C, MolecularOID.C):
                (i, sigma), (j, tau) = sorted(((i, sigma), (j, tau)))
                return self.a_dag_a_dag_l[i, sigma, j, tau] if connection == "left" else self.a_dag_a_dag_r[i, sigma, j, tau]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.A):
                (i, sigma), (j, tau) = sorted(((i, sigma), (j, tau)), reverse=True)
                return self.a_ann_a_ann_l[i, sigma, j, tau] if connection == "left" else self.a_ann_a_ann_r[i, sigma, j, tau]
            if (oid0, oid1) == (MolecularOID.C, MolecularOID.A):
                return self.a_dag_a_ann_l[i, sigma, j, tau] if connection == "left" else self.a_dag_a_ann_r[i, sigma, j, tau]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.C):
                return self.a_dag_a_ann_l[j, tau, i, sigma] if connection == "left" else self.a_dag_a_ann_r[j, tau, i, sigma]
            raise KeyError(f"nodes for OIDs ({oid0}, {oid1}) and (i, sigma, j, tau) == ({i}, {sigma}, {j}, {tau}) do not exist")
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
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.identity_l[i + 1].nid], [(SpinMolecularOID.Id, 1.)]))
            eid_next += 1
        for i in range(1, L):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_r[i].nid, self.identity_r[i + 1].nid], [(SpinMolecularOID.Id, 1.)]))
            eid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to left terminal
        for i, sigma in itertools.product(range(L - 1), (0, 1)):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_dag_l[i, sigma][i + 1].nid], [([SpinMolecularOID.CZ, SpinMolecularOID.IC][sigma], 1.)]))
            eid_next += 1
            # Z operators from Jordan-Wigner transformation
            for j in range(i + 1, L - 1):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_l[i, sigma][j].nid, self.a_dag_l[i, sigma][j + 1].nid], [(SpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
        # a_{i,\sigma} operators connected to left terminal
        for i, sigma in itertools.product(range(L - 1), (0, 1)):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_ann_l[i, sigma][i + 1].nid], [([SpinMolecularOID.AZ, SpinMolecularOID.IA][sigma], 1.)]))
            eid_next += 1
            # Z operators from Jordan-Wigner transformation
            for j in range(i + 1, L - 1):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_l[i, sigma][j].nid, self.a_ann_l[i, sigma][j + 1].nid], [(SpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to left terminal
        for i, sigma in itertools.product(range(L//2), (0, 1)):
            for j, tau in itertools.product(range(i, L//2), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_l[i, sigma][j].nid, self.a_dag_a_dag_l[i, sigma, j, tau][j + 1].nid], [([SpinMolecularOID.CI, SpinMolecularOID.ZC][tau], 1.)]))
                else:
                    assert sigma == 0 and tau == 1
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.identity_l[j].nid, self.a_dag_a_dag_l[i, sigma, j, tau][j + 1].nid], [(SpinMolecularOID.CC, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(j + 1, L//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_l[i, sigma, j, tau][k].nid, self.a_dag_a_dag_l[i, sigma, j, tau][k + 1].nid], [(SpinMolecularOID.Id, 1.)]))
                    eid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to left terminal
        for i, sigma in itertools.product(range(L//2), (0, 1)):
            for j, tau in itertools.product(range(i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                if i > j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_l[j, tau][i].nid, self.a_ann_a_ann_l[i, sigma, j, tau][i + 1].nid], [([SpinMolecularOID.AI, SpinMolecularOID.ZA][sigma], 1.)]))
                else:
                    assert sigma == 1 and tau == 0
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_ann_a_ann_l[i, sigma, j, tau][i + 1].nid], [(SpinMolecularOID.AA, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(i + 1, L//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_l[i, sigma, j, tau][k].nid, self.a_ann_a_ann_l[i, sigma, j, tau][k + 1].nid], [(SpinMolecularOID.Id, 1.)]))
                    eid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to left terminal
        for i, sigma in itertools.product(range(L//2), (0, 1)):
            for j, tau in itertools.product(range(L//2), (0, 1)):
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_l[i, sigma][j].nid, self.a_dag_a_ann_l[i, sigma, j, tau][j + 1].nid], [([SpinMolecularOID.AI, SpinMolecularOID.ZA][tau], 1.)]))
                elif i == j:
                    if sigma < tau:
                        oid = SpinMolecularOID.CA
                    elif sigma == tau:
                        oid = [SpinMolecularOID.NI, SpinMolecularOID.IN][sigma]
                    else:  # sigma > tau
                        oid = SpinMolecularOID.AC
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_dag_a_ann_l[i, sigma, j, tau][i + 1].nid], [(oid, 1.)]))
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_l[j, tau][i].nid, self.a_dag_a_ann_l[i, sigma, j, tau][i + 1].nid], [([SpinMolecularOID.CI, SpinMolecularOID.ZC][sigma], 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(max(i, j) + 1, L//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_l[i, sigma, j, tau][k].nid, self.a_dag_a_ann_l[i, sigma, j, tau][k + 1].nid], [(SpinMolecularOID.Id, 1.)]))
                    eid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to right terminal
        for i, sigma in itertools.product(range(1, L), (0, 1)):
            # Z operators from Jordan-Wigner transformation
            for j in range(1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_r[i, sigma][j].nid, self.a_dag_r[i, sigma][j + 1].nid], [(SpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_dag_r[i, sigma][i].nid, self.identity_r[i + 1].nid], [([SpinMolecularOID.CI, SpinMolecularOID.ZC][sigma], 1.)]))
            eid_next += 1
        # a_{i,\sigma} operators connected to right terminal
        for i, sigma in itertools.product(range(1, L), (0, 1)):
            # Z operators from Jordan-Wigner transformation
            for j in range(1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_r[i, sigma][j].nid, self.a_ann_r[i, sigma][j + 1].nid], [(SpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_ann_r[i, sigma][i].nid, self.identity_r[i + 1].nid], [([SpinMolecularOID.AI, SpinMolecularOID.ZA][sigma], 1.)]))
            eid_next += 1
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to right terminal
        for i, sigma in itertools.product(range(L//2 + 1, L), (0, 1)):
            for j, tau in itertools.product(range(i, L), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                # identities for transition to next site
                for k in range(L//2 + 1, i):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, sigma, j, tau][k].nid, self.a_dag_a_dag_r[i, sigma, j, tau][k + 1].nid], [(SpinMolecularOID.Id, 1.)]))
                    eid_next += 1
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, sigma, j, tau][i].nid, self.a_dag_r[j, tau][i + 1].nid], [([SpinMolecularOID.CZ, SpinMolecularOID.IC][sigma], 1.)]))
                else:
                    assert sigma == 0 and tau == 1
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, sigma, j, tau][i].nid, self.identity_r[i + 1].nid], [(SpinMolecularOID.CC, 1.)]))
                eid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to right terminal
        for i, sigma in itertools.product(range(L//2 + 1, L), (0, 1)):
            for j, tau in itertools.product(range(L//2 + 1, i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                # identities for transition to next site
                for k in range(L//2 + 1, j):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, sigma, j, tau][k].nid, self.a_ann_a_ann_r[i, sigma, j, tau][k + 1].nid], [(SpinMolecularOID.Id, 1.)]))
                    eid_next += 1
                if i > j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, sigma, j, tau][j].nid, self.a_ann_r[i, sigma][j + 1].nid], [([SpinMolecularOID.AZ, SpinMolecularOID.IA][tau], 1.)]))
                else:
                    assert sigma == 1 and tau == 0
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, sigma, j, tau][j].nid, self.identity_r[j + 1].nid], [(SpinMolecularOID.AA, 1.)]))
                eid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to right terminal
        for i, sigma in itertools.product(range(L//2 + 1, L), (0, 1)):
            for j, tau in itertools.product(range(L//2 + 1, L), (0, 1)):
                # identities for transition to next site
                for k in range(L//2 + 1, min(i, j)):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, sigma, j, tau][k].nid, self.a_dag_a_ann_r[i, sigma, j, tau][k + 1].nid], [(SpinMolecularOID.Id, 1.)]))
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, sigma, j, tau][i].nid, self.a_ann_r[j, tau][i + 1].nid], [([SpinMolecularOID.CZ, SpinMolecularOID.IC][sigma], 1.)]))
                elif i == j:
                    if sigma < tau:
                        oid = SpinMolecularOID.CA
                    elif sigma == tau:
                        oid = [SpinMolecularOID.NI, SpinMolecularOID.IN][sigma]
                    else:  # sigma > tau
                        oid = SpinMolecularOID.AC
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, sigma, j, tau][i].nid, self.identity_r[i + 1].nid], [(oid, 1.)]))
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, sigma, j, tau][j].nid, self.a_dag_r[i, sigma][j + 1].nid], [([SpinMolecularOID.AZ, SpinMolecularOID.IA][tau], 1.)]))
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
        # a^{\dagger}_{i,\sigma} operators connected to left terminal
        target.nids_a_dag_l = {}
        for i, sigma in itertools.product(range(L - 1), (0, 1)):
            target.nids_a_dag_l[i, sigma] = {}
            for j in range(i + 1, L):
                target.nids_a_dag_l[i, sigma][j] = self.a_dag_l[i, sigma][j].nid
        # a_{i,\sigma} operators connected to left terminal
        target.nids_a_ann_l = {}
        for i, sigma in itertools.product(range(L - 1), (0, 1)):
            target.nids_a_ann_l[i, sigma] = {}
            for j in range(i + 1, L):
                target.nids_a_ann_l[i, sigma][j] = self.a_ann_l[i, sigma][j].nid
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to left terminal
        target.nids_a_dag_a_dag_l = {}
        for i, sigma in itertools.product(range(L//2), (0, 1)):
            for j, tau in itertools.product(range(i, L//2), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                target.nids_a_dag_a_dag_l[i, sigma, j, tau] = {}
                for k in range(j + 1, L//2 + 1):
                    target.nids_a_dag_a_dag_l[i, sigma, j, tau][k] = self.a_dag_a_dag_l[i, sigma, j, tau][k].nid
        # a_{i,\sigma} a_{j,\tau} operators connected to left terminal
        target.nids_a_ann_a_ann_l = {}
        for i, sigma in itertools.product(range(L//2), (0, 1)):
            for j, tau in itertools.product(range(i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                target.nids_a_ann_a_ann_l[i, sigma, j, tau] = {}
                for k in range(i + 1, L//2 + 1):
                    target.nids_a_ann_a_ann_l[i, sigma, j, tau][k] = self.a_ann_a_ann_l[i, sigma, j, tau][k].nid
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to left terminal
        target.nids_a_dag_a_ann_l = {}
        for i, sigma in itertools.product(range(L//2), (0, 1)):
            for j, tau in itertools.product(range(L//2), (0, 1)):
                target.nids_a_dag_a_ann_l[i, sigma, j, tau] = {}
                for k in range(max(i, j) + 1, L//2 + 1):
                    target.nids_a_dag_a_ann_l[i, sigma, j, tau][k] = self.a_dag_a_ann_l[i, sigma, j, tau][k].nid
        # a^{\dagger}_{i,\sigma} operators connected to right terminal
        target.nids_a_dag_r = {}
        for i, sigma in itertools.product(range(1, L), (0, 1)):
            target.nids_a_dag_r[i, sigma] = {}
            for j in range(1, i + 1):
                target.nids_a_dag_r[i, sigma][j] = self.a_dag_r[i, sigma][j].nid
        # a_{i,\sigma} operators connected to right terminal
        target.nids_a_ann_r = {}
        for i, sigma in itertools.product(range(1, L), (0, 1)):
            target.nids_a_ann_r[i, sigma] = {}
            for j in range(1, i + 1):
                target.nids_a_ann_r[i, sigma][j] = self.a_ann_r[i, sigma][j].nid
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to right terminal
        target.nids_a_dag_a_dag_r = {}
        for i, sigma in itertools.product(range(L//2 + 1, L), (0, 1)):
            for j, tau in itertools.product(range(i, L), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                target.nids_a_dag_a_dag_r[i, sigma, j, tau] = {}
                for k in range(L//2 + 1, i + 1):
                    target.nids_a_dag_a_dag_r[i, sigma, j, tau][k] = self.a_dag_a_dag_r[i, sigma, j, tau][k].nid
        # a_{i,\sigma} a_{j,\tau} operators connected to right terminal
        target.nids_a_ann_a_ann_r = {}
        for i, sigma in itertools.product(range(L//2 + 1, L), (0, 1)):
            for j, tau in itertools.product(range(L//2 + 1, i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                target.nids_a_ann_a_ann_r[i, sigma, j, tau] = {}
                for k in range(L//2 + 1, j + 1):
                    target.nids_a_ann_a_ann_r[i, sigma, j, tau][k] = self.a_ann_a_ann_r[i, sigma, j, tau][k].nid
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to right terminal
        target.nids_a_dag_a_ann_r = {}
        for i, sigma in itertools.product(range(L//2 + 1, L), (0, 1)):
            for j, tau in itertools.product(range(L//2 + 1, L), (0, 1)):
                target.nids_a_dag_a_ann_r[i, sigma, j, tau] = {}
                for k in range(L//2 + 1, min(i, j) + 1):
                    target.nids_a_dag_a_ann_r[i, sigma, j, tau][k] = self.a_dag_a_ann_r[i, sigma, j, tau][k].nid


def _spin_molecular_hamiltonian_graph_add_term(graph: OpGraph, nodes: SpinMolecularOpGraphNodes, oplist: Sequence[tuple], coeff: float):
    """
    Add an operator term (operator string of creation and annihilation operators)
    to the operator graph describing a molecular Hamiltonian, assuming a spin orbital basis.
    """
    eid_next = max(graph.edges.keys()) + 1

    L = nodes.L

    # sort by site (orbital) and spin index
    oplist = sorted(oplist)

    if len(oplist) == 2:
        (i, sigma, oid0), (j, tau, oid1) = oplist
        if i == j:
            spin_oid = SpinOperatorConverter.to_spin_operator([(sigma, oid0), (tau, oid1)], True, True)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid], [(spin_oid, coeff)]))
        else:
            assert i < j
            if j <= L//2:
                nodes_l = nodes.get([oplist[0]], "left")
                spin_oid = SpinOperatorConverter.to_spin_operator([(tau, oid1)], False, True)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes.identity_r[j + 1].nid], [(spin_oid, coeff)]))
            elif i >= L//2:
                nodes_r = nodes.get([oplist[1]], "right")
                spin_oid = SpinOperatorConverter.to_spin_operator([(sigma, oid0)], True, False)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes_r[i + 1].nid], [(spin_oid, coeff)]))
            else:
                nodes_l = nodes.get([oplist[0]], "left")
                nodes_r = nodes.get([oplist[1]], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[L//2].nid, nodes_r[L//2 + 1].nid], [(SpinMolecularOID.ZZ, coeff)]))
    elif len(oplist) == 4:
        (i, sigma, oid0), (j, tau, oid1), (k, mu, oid2), (l, nu, oid3) = oplist
        if i == j == k == l:
            spin_oid = SpinOperatorConverter.to_spin_operator([(sigma, oid0), (tau, oid1), (mu, oid2), (nu, oid3)], True, True)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid], [(spin_oid, coeff)]))
        elif i == j == k:
            nodes_r = nodes.get([oplist[3]], "right")
            spin_oid = SpinOperatorConverter.to_spin_operator([(sigma, oid0), (tau, oid1), (mu, oid2)], True, False)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes_r[i + 1].nid], [(spin_oid, coeff)]))
        elif j == k == l:
            nodes_l = nodes.get([oplist[0]], "left")
            spin_oid = SpinOperatorConverter.to_spin_operator([(tau, oid1), (mu, oid2), (nu, oid3)], False, True)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[j].nid, nodes.identity_r[j + 1].nid], [(spin_oid, coeff)]))
        elif j == k:
            nodes_l = nodes.get([oplist[0]], "left")
            nodes_r = nodes.get([oplist[3]], "right")
            spin_oid = SpinOperatorConverter.to_spin_operator([(tau, oid1), (mu, oid2)], False, False)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid], [(spin_oid, coeff)]))
        elif k <= L//2:
            nodes_l = nodes.get(oplist[:2], "left")
            if k == l:
                spin_oid = SpinOperatorConverter.to_spin_operator([(mu, oid2), (nu, oid3)], True, True)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes.identity_r[k + 1].nid], [(spin_oid, coeff)]))
            else:
                nodes_r = nodes.get([oplist[3]], "right")
                spin_oid = SpinOperatorConverter.to_spin_operator([(mu, oid2)], True, False)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes_r[k + 1].nid], [(spin_oid, coeff)]))
        elif j >= L//2:
            nodes_r = nodes.get(oplist[2:], "right")
            if i == j:
                spin_oid = SpinOperatorConverter.to_spin_operator([(sigma, oid0), (tau, oid1)], True, True)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[j].nid, nodes_r[j + 1].nid], [(spin_oid, coeff)]))
            else:
                nodes_l = nodes.get([oplist[0]], "left")
                spin_oid = SpinOperatorConverter.to_spin_operator([(tau, oid1)], False, True)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid], [(spin_oid, coeff)]))
        else:
            nodes_l = nodes.get(oplist[:2], "left")
            nodes_r = nodes.get(oplist[2:], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[L//2].nid, nodes_r[L//2 + 1].nid], [(SpinMolecularOID.Id, coeff)]))
    else:
        raise NotImplementedError


def spin_molecular_hamiltonian_mpo(tkin, vint, optimize=True) -> MPO:
    r"""
    Construct a molecular Hamiltonian as MPO, assuming a spin orbital basis and
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

    # interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau,\mu,\nu} v_{i,j,k,l} \delta_{\sigma,\mu} \delta_{\tau,\nu} a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\nu} a_{k,\mu}:
    # can anti-commute fermionic operators such that (i,\sigma) < (j,\tau) and (k,\mu) < (l,\nu)
    gint0 = 0.5 * (vint                             + np.transpose(vint, (1, 0, 3, 2)))
    gint1 = 0.5 * (np.transpose(vint, (1, 0, 2, 3)) + np.transpose(vint, (0, 1, 3, 2)))

    def get_vint_coeff(spatial_idx, spin_idx):
        valid = False
        coeff = 0
        if (spin_idx[0] == spin_idx[2]) and (spin_idx[1] == spin_idx[3]):
            coeff += gint0[spatial_idx]
            valid = True
        if (spin_idx[0] == spin_idx[3]) and (spin_idx[1] == spin_idx[2]):
            coeff -= gint1[spatial_idx]
            valid = True
        return coeff, valid

    if optimize:
        # optimize MPO bond dimensions based on bipartite graph theory

        opchains = []
        # kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
        for i in range(2*L):
            for j in range(2*L):
                if (i - j) % 2 != 0:
                    continue
                if i == j:
                    # diagonal hopping term
                    opchains.append(SpinOperatorConverter.to_spin_opchain(
                        OpChain([MolecularOID.N], [0, 0], tkin[i//2, i//2], i)))
                else:
                    (a, p), (b, q) = sorted([(i, MolecularOID.C), (j, MolecularOID.A)])
                    opchains.append(SpinOperatorConverter.to_spin_opchain(
                        OpChain([p] + (b - a - 1)*[MolecularOID.Z] + [q],
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
                        opchains.append(SpinOperatorConverter.to_spin_opchain(OpChain(oids, qnums, coeff, a)))
        graph = OpGraph.from_opchains(opchains, L, SpinMolecularOID.Id)

    else:
        # explicit construction (typically faster, but does not optimize cases
        # of zero coefficients, and is slightly sub-optimal close to boundary)
        assert L >= 2
        nodes = SpinMolecularOpGraphNodes(L)
        graph = nodes.generate_graph()
        # kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
        for i in range(L):
            for j in range(L):
                for sigma in (0, 1):
                    _spin_molecular_hamiltonian_graph_add_term(
                        graph, nodes, [(i, sigma, MolecularOID.C), (j, sigma, MolecularOID.A)], tkin[i, j])
        # interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau} v_{i,j,k,l} a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\tau} a_{k,\sigma}
        for i, sigma in itertools.product(range(L), (0, 1)):
            for j, tau in itertools.product(range(i, L), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                for k, mu in itertools.product(range(L), (0, 1)):
                    for l, nu in itertools.product(range(k, L), (0, 1)):
                        if (k, mu) >= (l, nu):
                            continue
                        coeff, valid = get_vint_coeff((i, j, k, l), (sigma, tau, mu, nu))
                        if not valid:
                            continue
                        oplist = [(i, sigma, MolecularOID.C),
                                  (j, tau,   MolecularOID.C),
                                  (l, nu,    MolecularOID.A),
                                  (k, mu,    MolecularOID.A)]
                        _spin_molecular_hamiltonian_graph_add_term(
                            graph, nodes, oplist, coeff)

    # skip consistency check for larger L (would take very long)
    if L <= 10:
        assert graph.is_consistent()
    opmap = _spin_molecular_hamiltonian_generate_operator_map()
    # convert to MPO
    mpo = MPO.from_opgraph(qd, graph, opmap, compute_nid_map=(not optimize))
    # store node information in MPO, to identify virtual bonds by creation and annihilation operators
    if not optimize:
        nodes.copy_nids(mpo)

    return mpo


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
