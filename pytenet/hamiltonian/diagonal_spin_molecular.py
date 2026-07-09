"""
Construct a spin molecular Hamiltonian with a diagonal interaction term
(products of two number operators) as a matrix product operator (MPO).
"""

import itertools
from collections.abc import Sequence
from enum import IntEnum
import numpy as np
from ..mpo import MPO
from ..opchain import OpChain
from ..opgraph import OpGraphNode, OpGraphEdge, OpGraph
from ..qnumber import encode_quantum_number_pair

__all__ = ["diagonal_spin_molecular_hamiltonian_mpo"]


class DiagonalSpinMolecularOID(IntEnum):
    """
    Local operator IDs for a molecular Hamiltonian using a spin orbital basis.
    """
    I   =  0
    IC  =  1
    IA  =  2
    CI  =  3
    CZ  =  4
    AI  =  5
    AZ  =  6
    ZC  =  7
    ZA  =  8
    ZZ  =  9
    NN  = 10
    NPN = 11  # n_{\uparrow} + n_{\downarrow}

    @property
    def is_creation_annihil(self) -> bool:
        """
        Whether the operator ID corresponds to a creation or an annihilation operator.
        """
        return 1 <= self.value <= 8


def _diagonal_spin_molecular_hamiltonian_generate_operator_map():
    """
    Generate the local operator map for a spin molecular Hamiltonian
    with a diagonal interaction term.
    """
    id2 = np.identity(2)
    # creation and annihilation operators for a single spin and lattice site
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    z = np.array([[1., 0.], [0., -1.]])
    # operator map
    opmap = {
        DiagonalSpinMolecularOID.I:   np.identity(4),
        DiagonalSpinMolecularOID.IC:  np.kron(id2,   a_dag),
        DiagonalSpinMolecularOID.IA:  np.kron(id2,   a_ann),
        DiagonalSpinMolecularOID.CI:  np.kron(a_dag, id2  ),
        DiagonalSpinMolecularOID.CZ:  np.kron(a_dag, z    ),
        DiagonalSpinMolecularOID.AI:  np.kron(a_ann, id2  ),
        DiagonalSpinMolecularOID.AZ:  np.kron(a_ann, z    ),
        DiagonalSpinMolecularOID.ZC:  np.kron(z,     a_dag),
        DiagonalSpinMolecularOID.ZA:  np.kron(z,     a_ann),
        DiagonalSpinMolecularOID.ZZ:  np.kron(z,     z    ),
        DiagonalSpinMolecularOID.NN:  np.kron(numop, numop),
        DiagonalSpinMolecularOID.NPN: np.kron(id2, numop) + np.kron(numop, id2),
    }
    return opmap


class DiagonalSpinMolecularOpGraphNodes:
    """
    Operator graph nodes used for the diagonal molecular Hamiltonian construction,
    assuming a spin orbital basis.
    """
    def __init__(self, nsites):

        self.nsites = nsites

        nid_next = 0
        # identity chains from the left and right
        self.identity_l = {}
        self.identity_r = {}
        for i in range(nsites):
            self.identity_l[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1
        for i in range(1, nsites + 1):
            self.identity_r[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to left terminal
        self.a_dag_l = {}
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            self.a_dag_l[i, sigma] = {}
            for j in range(i + 1, nsites//2 + 1):
                self.a_dag_l[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair( 1, [1, -1][sigma]))
                nid_next += 1
        # a_{i,\sigma} operators connected to left terminal
        self.a_ann_l = {}
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            self.a_ann_l[i, sigma] = {}
            for j in range(i + 1, nsites//2 + 1):
                self.a_ann_l[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair(-1, [-1, 1][sigma]))
                nid_next += 1
        # n_{i,\uparrow} + n_{i,\downarrow} operators connected to left terminal
        self.numop_l = {}
        for i in range(nsites//2):
            self.numop_l[i] = {}
            for j in range(i + 1, nsites//2 + 1):
                self.numop_l[i][j] = OpGraphNode(nid_next, [], [], 0)
                nid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to right terminal
        self.a_dag_r = {}
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            self.a_dag_r[i, sigma] = {}
            for j in range(nsites//2 + 1, i + 1):
                self.a_dag_r[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair(-1, [-1, 1][sigma]))
                nid_next += 1
        # a_{i,\sigma} operators connected to right terminal
        self.a_ann_r = {}
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            self.a_ann_r[i, sigma] = {}
            for j in range(nsites//2 + 1, i + 1):
                self.a_ann_r[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair( 1, [1, -1][sigma]))
                nid_next += 1
        # n_{i,\uparrow} + n_{i,\downarrow} operators connected to right terminal
        self.numop_r = {}
        for i in range(nsites//2 + 1, nsites):
            self.numop_r[i] = {}
            for j in range(nsites//2 + 1, i + 1):
                self.numop_r[i][j] = OpGraphNode(nid_next, [], [], 0)
                nid_next += 1

    def get(self, i_site: int, oid: DiagonalSpinMolecularOID, connection):
        """
        Retrieve nodes corresponding to a single or pairs of
        creation, annihilation or number operators.
        """
        match oid:
            case DiagonalSpinMolecularOID.IC:
                assert connection == "left"
                return self.a_dag_l[i_site, 1]
            case DiagonalSpinMolecularOID.IA:
                assert connection == "left"
                return self.a_ann_l[i_site, 1]
            case DiagonalSpinMolecularOID.CI:
                assert connection == "right"
                return self.a_dag_r[i_site, 0]
            case DiagonalSpinMolecularOID.CZ:
                assert connection == "left"
                return self.a_dag_l[i_site, 0]
            case DiagonalSpinMolecularOID.AI:
                assert connection == "right"
                return self.a_ann_r[i_site, 0]
            case DiagonalSpinMolecularOID.AZ:
                assert connection == "left"
                return self.a_ann_l[i_site, 0]
            case DiagonalSpinMolecularOID.ZC:
                assert connection == "right"
                return self.a_dag_r[i_site, 1]
            case DiagonalSpinMolecularOID.ZA:
                assert connection == "right"
                return self.a_ann_r[i_site, 1]
            case DiagonalSpinMolecularOID.NPN:
                return self.numop_l[i_site] if connection == "left" else self.numop_r[i_site]
            case _:
                raise KeyError(f"nodes for OID {oid} and site index {i_site} do not exist")

    def generate_graph(self) -> OpGraph:
        """
        Create and initialize an operator graph with corresponding nodes and edges.
        """
        nsites = self.nsites

        # initialize graph with nodes
        graph = OpGraph(list(self.identity_l.values()) +
                        list(self.identity_r.values()) +
                        [node for nodes in self.a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.numop_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.numop_r.values() for node in nodes.values()],
                        [], [self.identity_l[0].nid, self.identity_r[nsites].nid])

        # edges
        eid_next = 0
        # identities connected to left and right terminals
        for i in range(nsites - 1):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.identity_l[i + 1].nid],
                            [(DiagonalSpinMolecularOID.I, 1.)]))
            eid_next += 1
        for i in range(1, nsites):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_r[i].nid, self.identity_r[i + 1].nid],
                            [(DiagonalSpinMolecularOID.I, 1.)]))
            eid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to left terminal
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_dag_l[i, sigma][i + 1].nid],
                            [([DiagonalSpinMolecularOID.CZ,
                               DiagonalSpinMolecularOID.IC][sigma], 1.)]))
            eid_next += 1
            # Z operators from Jordan-Wigner transformation
            for j in range(i + 1, nsites//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_l[i, sigma][j].nid,
                                           self.a_dag_l[i, sigma][j + 1].nid],
                                [(DiagonalSpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
        # a_{i,\sigma} operators connected to left terminal
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_ann_l[i, sigma][i + 1].nid],
                            [([DiagonalSpinMolecularOID.AZ,
                               DiagonalSpinMolecularOID.IA][sigma], 1.)]))
            eid_next += 1
            # Z operators from Jordan-Wigner transformation
            for j in range(i + 1, nsites//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_l[i, sigma][j].nid,
                                           self.a_ann_l[i, sigma][j + 1].nid],
                                [(DiagonalSpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
        # n_{i,\uparrow} + n_{i,\downarrow} operators connected to left terminal
        for i in range(nsites//2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.numop_l[i][i + 1].nid],
                            [(DiagonalSpinMolecularOID.NPN, 1.)]))
            eid_next += 1
            # identities for transition to next site
            for j in range(i + 1, nsites//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.numop_l[i][j].nid,
                                           self.numop_l[i][j + 1].nid],
                                [(DiagonalSpinMolecularOID.I, 1.)]))
                eid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to right terminal
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            # Z operators from Jordan-Wigner transformation
            for j in range(nsites//2 + 1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_r[i, sigma][j].nid,
                                           self.a_dag_r[i, sigma][j + 1].nid],
                                [(DiagonalSpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_dag_r[i, sigma][i].nid, self.identity_r[i + 1].nid],
                            [([DiagonalSpinMolecularOID.CI,
                               DiagonalSpinMolecularOID.ZC][sigma], 1.)]))
            eid_next += 1
        # a_{i,\sigma} operators connected to right terminal
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            # Z operators from Jordan-Wigner transformation
            for j in range(nsites//2 + 1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_r[i, sigma][j].nid,
                                           self.a_ann_r[i, sigma][j + 1].nid],
                                [(DiagonalSpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_ann_r[i, sigma][i].nid, self.identity_r[i + 1].nid],
                            [([DiagonalSpinMolecularOID.AI,
                               DiagonalSpinMolecularOID.ZA][sigma], 1.)]))
            eid_next += 1
        # n_{i,\uparrow} + n_{i,\downarrow} operators connected to right terminal
        for i in range(nsites//2 + 1, nsites):
            # identities for transition to next site
            for j in range(nsites//2 + 1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.numop_r[i][j].nid,
                                           self.numop_r[i][j + 1].nid],
                                [(DiagonalSpinMolecularOID.I, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.numop_r[i][i].nid, self.identity_r[i + 1].nid],
                            [(DiagonalSpinMolecularOID.NPN, 1.)]))
            eid_next += 1

        return graph


def _diagonal_spin_molecular_hamiltonian_graph_add_term(
        graph: OpGraph, nodes: DiagonalSpinMolecularOpGraphNodes,
        oplist: Sequence[tuple], coeff: float):
    """
    Add an operator term (operator string of creation, annihilation and number operators)
    to the operator graph describing a diagonal molecular Hamiltonian.
    """
    eid_next = max(graph.edges.keys()) + 1

    nsites = nodes.nsites

    # sort by site (orbital) index
    oplist = sorted(oplist)

    if len(oplist) == 1:
        (i, oid) = oplist[0]
        # expecting total number operator
        assert oid in (DiagonalSpinMolecularOID.NN, DiagonalSpinMolecularOID.NPN)
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid],
                        [(oid, coeff)]))
    elif len(oplist) == 2:
        (i, oid0), (j, oid1) = oplist
        assert i < j
        if j <= nsites//2:
            nodes_l = nodes.get(i, oid0, "left")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[j].nid, nodes.identity_r[j + 1].nid],
                            [(oid1, coeff)]))
        elif i >= nsites//2:
            nodes_r = nodes.get(j, oid1, "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes_r[i + 1].nid],
                            [(oid0, coeff)]))
        else:
            nodes_l = nodes.get(i, oid0, "left")
            nodes_r = nodes.get(j, oid1, "right")
            if oid0.is_creation_annihil:
                assert oid1.is_creation_annihil
                oid_center = DiagonalSpinMolecularOID.ZZ
            else:
                assert oid0 == DiagonalSpinMolecularOID.NPN
                assert oid1 == DiagonalSpinMolecularOID.NPN
                oid_center = DiagonalSpinMolecularOID.I
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[nsites//2].nid, nodes_r[nsites//2 + 1].nid],
                            [(oid_center, coeff)]))
    else:
        raise NotImplementedError


def diagonal_spin_molecular_hamiltonian_mpo(tkin, vint, optimize=True) -> MPO:
    r"""
    Construct a spin molecular Hamiltonian with a diagonal interaction term as an MPO:

    .. math::

        H = \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
            + \sum_{(i,\sigma) < (j,\tau)} v_{i,j} n_{i,\sigma} n_{j,\tau}
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    nsites = tkin.shape[0]
    assert tkin.shape == (nsites, nsites)
    assert vint.shape == (nsites, nsites)

    # physical particle number and spin quantum numbers (encoded as single integer)
    qn = [0,  1,  1,  2]
    qs = [0, -1,  1,  0]
    qsite = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qn, qs)]

    if optimize:
        # optimize MPO bond dimensions based on bipartite graph theory
        opchains = []
        # kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
        for i in range(nsites):
            # kinetic hopping terms for i > j
            for j in range(i):  # i > j
                for sigma in (0, 1):
                    oid0 = [DiagonalSpinMolecularOID.AZ, DiagonalSpinMolecularOID.IA][sigma]
                    oid1 = [DiagonalSpinMolecularOID.CI, DiagonalSpinMolecularOID.ZC][sigma]
                    opchains.append(OpChain(  [oid0]
                                            + (i - j - 1)*[DiagonalSpinMolecularOID.ZZ]
                                            + [oid1],
                                              [0]
                                            + (i - j)*[encode_quantum_number_pair(-1, 2*sigma - 1)]
                                            + [0], tkin[i, j], j))
            # diagonal term
            opchains.append(OpChain([DiagonalSpinMolecularOID.NPN], [0, 0], tkin[i, i], i))
            # kinetic hopping terms for i < j
            for j in range(i + 1, nsites):  # i < j
                for sigma in (0, 1):
                    oid0 = [DiagonalSpinMolecularOID.CZ, DiagonalSpinMolecularOID.IC][sigma]
                    oid1 = [DiagonalSpinMolecularOID.AI, DiagonalSpinMolecularOID.ZA][sigma]
                    opchains.append(OpChain(  [oid0]
                                            + (j - i - 1)*[DiagonalSpinMolecularOID.ZZ]
                                            + [oid1],
                                              [0]
                                            + (j - i)*[encode_quantum_number_pair(1, 1 - 2*sigma)]
                                            + [0], tkin[i, j], i))
        # interaction terms \sum_{(i,\sigma) < (j,\tau)} v_{i,j} n_{i,\sigma} n_{j,\tau}
        for i in range(nsites):
            # n_{i,\uparrow} n_{i,\downarrow} term
            opchains.append(OpChain([DiagonalSpinMolecularOID.NN],
                                    [0, 0], vint[i, i], i))
            for j in range(i + 1, nsites):  # i < j
                opchains.append(OpChain(
                      [DiagonalSpinMolecularOID.NPN]
                    + (j - i - 1)*[DiagonalSpinMolecularOID.I]
                    + [DiagonalSpinMolecularOID.NPN],
                    (j - i + 2)*[0], vint[i, j], i))
        graph = OpGraph.from_opchains(opchains, nsites, 0)
    else:
        # explicit construction (typically faster, but does not optimize cases
        # of zero coefficients, and is slightly sub-optimal close to boundary)
        assert nsites >= 2
        nodes = DiagonalSpinMolecularOpGraphNodes(nsites)
        graph = nodes.generate_graph()
        # kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
        for i in range(nsites):
            # kinetic hopping terms for i > j
            for j in range(i):  # i > j
                for sigma in (0, 1):
                    oid0 = [DiagonalSpinMolecularOID.AZ, DiagonalSpinMolecularOID.IA][sigma]
                    oid1 = [DiagonalSpinMolecularOID.CI, DiagonalSpinMolecularOID.ZC][sigma]
                    _diagonal_spin_molecular_hamiltonian_graph_add_term(
                        graph, nodes, [(j, oid0), (i, oid1)], tkin[i, j])
            # diagonal term
            _diagonal_spin_molecular_hamiltonian_graph_add_term(
                graph, nodes, [(i, DiagonalSpinMolecularOID.NPN)], tkin[i, i])
            # kinetic hopping terms for i < j
            for j in range(i + 1, nsites):  # i < j
                for sigma in (0, 1):
                    oid0 = [DiagonalSpinMolecularOID.CZ, DiagonalSpinMolecularOID.IC][sigma]
                    oid1 = [DiagonalSpinMolecularOID.AI, DiagonalSpinMolecularOID.ZA][sigma]
                    _diagonal_spin_molecular_hamiltonian_graph_add_term(
                        graph, nodes, [(i, oid0), (j, oid1)], tkin[i, j])
        # interaction terms \sum_{(i,\sigma) < (j,\tau)} v_{i,j} n_{i,\sigma} n_{j,\tau}
        for i in range(nsites):
            # n_{i,\uparrow} n_{i,\downarrow} term
            _diagonal_spin_molecular_hamiltonian_graph_add_term(
                graph, nodes, [(i, DiagonalSpinMolecularOID.NN)], vint[i, i])
            for j in range(i + 1, nsites):  # i < j
                oplist = [(i, DiagonalSpinMolecularOID.NPN),
                          (j, DiagonalSpinMolecularOID.NPN)]
                _diagonal_spin_molecular_hamiltonian_graph_add_term(
                    graph, nodes, oplist, vint[i, j])

    # skip consistency check for larger nsites (would take very long)
    if nsites <= 12:
        assert graph.is_consistent()
    opmap = _diagonal_spin_molecular_hamiltonian_generate_operator_map()
    # convert to MPO
    return MPO.from_opgraph(qsite, graph, opmap, compute_nid_map = not optimize)
