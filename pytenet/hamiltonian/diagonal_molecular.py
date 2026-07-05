"""
Construct a molecular Hamiltonian with a diagonal interaction term
(products of two number operators) as a matrix product operator (MPO).
"""

from collections.abc import Sequence
import numpy as np
from ..mpo import MPO
from ..opchain import OpChain
from ..opgraph import OpGraphNode, OpGraphEdge, OpGraph
from .molecular import MolecularOID, _molecular_hamiltonian_generate_operator_map


__all__ = ["diagonal_molecular_hamiltonian_mpo"]


class DiagonalMolecularOpGraphNodes:
    """
    Operator graph nodes used for the diagonal molecular Hamiltonian construction.
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
        # a^{\dagger}_i operators connected to left terminal
        self.a_dag_l = {}
        for i in range(nsites//2):
            self.a_dag_l[i] = {}
            for j in range(i + 1, nsites//2 + 1):
                self.a_dag_l[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1
        # a_i operators connected to left terminal
        self.a_ann_l = {}
        for i in range(nsites//2):
            self.a_ann_l[i] = {}
            for j in range(i + 1, nsites//2 + 1):
                self.a_ann_l[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # n_i operators connected to left terminal
        self.numop_l = {}
        for i in range(nsites//2):
            self.numop_l[i] = {}
            for j in range(i + 1, nsites//2 + 1):
                self.numop_l[i][j] = OpGraphNode(nid_next, [], [], 0)
                nid_next += 1
        # a^{\dagger}_i operators connected to right terminal
        self.a_dag_r = {}
        for i in range(nsites//2 + 1, nsites):
            self.a_dag_r[i] = {}
            for j in range(nsites//2 + 1, i + 1):
                self.a_dag_r[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # a_i operators connected to right terminal
        self.a_ann_r = {}
        for i in range(nsites//2 + 1, nsites):
            self.a_ann_r[i] = {}
            for j in range(nsites//2 + 1, i + 1):
                self.a_ann_r[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1
        # n_i operators connected to right terminal
        self.numop_r = {}
        for i in range(nsites//2 + 1, nsites):
            self.numop_r[i] = {}
            for j in range(nsites//2 + 1, i + 1):
                self.numop_r[i][j] = OpGraphNode(nid_next, [], [], 0)
                nid_next += 1

    def get(self, i_site: int, oid: MolecularOID, connection):
        """
        Retrieve nodes corresponding to a single or pairs of creation and annihilation operators.
        """
        if oid == MolecularOID.C:
            return self.a_dag_l[i_site] if connection == "left" else self.a_dag_r[i_site]
        if oid == MolecularOID.A:
            return self.a_ann_l[i_site] if connection == "left" else self.a_ann_r[i_site]
        if oid == MolecularOID.N:
            return self.numop_l[i_site] if connection == "left" else self.numop_r[i_site]
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
                            [(MolecularOID.I, 1.)]))
            eid_next += 1
        for i in range(1, nsites):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_r[i].nid, self.identity_r[i + 1].nid],
                            [(MolecularOID.I, 1.)]))
            eid_next += 1
        # a^{\dagger}_i operators connected to left terminal
        for i in range(nsites//2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_dag_l[i][i + 1].nid],
                            [(MolecularOID.C, 1.)]))
            eid_next += 1
            # Z operator from Jordan-Wigner transformation
            for j in range(i + 1, nsites//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_l[i][j].nid, self.a_dag_l[i][j + 1].nid],
                                [(MolecularOID.Z, 1.)]))
                eid_next += 1
        # a_i operators connected to left terminal
        for i in range(nsites//2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_ann_l[i][i + 1].nid],
                            [(MolecularOID.A, 1.)]))
            eid_next += 1
            # Z operator from Jordan-Wigner transformation
            for j in range(i + 1, nsites//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_l[i][j].nid, self.a_ann_l[i][j + 1].nid],
                                [(MolecularOID.Z, 1.)]))
                eid_next += 1
        # n_i operators connected to left terminal
        for i in range(nsites//2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.numop_l[i][i + 1].nid],
                            [(MolecularOID.N, 1.)]))
            eid_next += 1
            # identities for transition to next site
            for j in range(i + 1, nsites//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.numop_l[i][j].nid, self.numop_l[i][j + 1].nid],
                                [(MolecularOID.I, 1.)]))
                eid_next += 1
        # a^{\dagger}_i operators connected to right terminal
        for i in range(nsites//2 + 1, nsites):
            # Z operator from Jordan-Wigner transformation
            for j in range(nsites//2 + 1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_r[i][j].nid, self.a_dag_r[i][j + 1].nid],
                                [(MolecularOID.Z, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_dag_r[i][i].nid, self.identity_r[i + 1].nid],
                            [(MolecularOID.C, 1.)]))
            eid_next += 1
        # a_i operators connected to right terminal
        for i in range(nsites//2 + 1, nsites):
            # Z operator from Jordan-Wigner transformation
            for j in range(nsites//2 + 1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_r[i][j].nid, self.a_ann_r[i][j + 1].nid],
                                [(MolecularOID.Z, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_ann_r[i][i].nid, self.identity_r[i + 1].nid],
                            [(MolecularOID.A, 1.)]))
            eid_next += 1
        # n_i operators connected to right terminal
        for i in range(nsites//2 + 1, nsites):
            # identities for transition to next site
            for j in range(nsites//2 + 1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.numop_r[i][j].nid, self.numop_r[i][j + 1].nid],
                                [(MolecularOID.I, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.numop_r[i][i].nid, self.identity_r[i + 1].nid],
                            [(MolecularOID.N, 1.)]))
            eid_next += 1

        return graph


def _diagonal_molecular_hamiltonian_graph_add_term(
        graph: OpGraph, nodes: DiagonalMolecularOpGraphNodes,
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
        # expecting number operator
        assert oid == MolecularOID.N
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid],
                        [(MolecularOID.N, coeff)]))
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
            if oid0 in (MolecularOID.A, MolecularOID.C):
                assert oid1 in (MolecularOID.A, MolecularOID.C)
                oid_center = MolecularOID.Z
            else:
                assert oid0 == MolecularOID.N
                assert oid1 == MolecularOID.N
                oid_center = MolecularOID.I
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[nsites//2].nid, nodes_r[nsites//2 + 1].nid],
                            [(oid_center, coeff)]))
    else:
        raise NotImplementedError


def diagonal_molecular_hamiltonian_mpo(tkin, vint, optimize=True) -> MPO:
    r"""
    Construct a molecular Hamiltonian with a diagonal interaction term as an MPO:

    .. math::

        H = \sum_{i,j} t_{i,j} a^{\dagger}_i a_j + \frac{1}{2} \sum_{i,j} v_{i,j} n_i n_j
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    nsites = tkin.shape[0]
    assert tkin.shape == (nsites, nsites)
    assert vint.shape == (nsites, nsites)

    # collect the separate number operators into a dedicated term
    uloc = np.diag(tkin) + 0.5 * np.diag(vint)

    # interaction terms \frac{1}{2} \sum_{i,j} v_{i,j} n_i n_j:
    # can swap number operators such that i < j
    gint = np.triu(0.5 * (vint + vint.T), k=1)

    if optimize:
        # optimize MPO bond dimensions based on bipartite graph theory
        opchains = []
        # kinetic hopping terms \sum_{i \neq j} t_{i,j} a^{\dagger}_i a_j
        for i in range(nsites):
            for j in range(nsites):
                if i == j:
                    continue
                (a, p), (b, q) = sorted([(i, MolecularOID.C), (j, MolecularOID.A)])
                opchains.append(OpChain([p] + (b - a - 1)*[MolecularOID.Z] + [q],
                                        [0] + (b - a)*[int(p)] + [0], tkin[i, j], a))
        # on-site number operators \sum_i u_i n_i
        for i in range(nsites):
            opchains.append(OpChain([MolecularOID.N], [0, 0], uloc[i], i))
        # interaction terms \sum_{i < j} v_{i,j} n_i n_j
        for i in range(nsites):
            for j in range(i + 1, nsites):  # i < j
                opchains.append(OpChain(
                    [MolecularOID.N] + (j - i - 1)*[MolecularOID.I] + [MolecularOID.N],
                    (j - i + 2)*[0], gint[i, j], i))
        graph = OpGraph.from_opchains(opchains, nsites, 0)
    else:
        # explicit construction (typically faster, but does not optimize cases
        # of zero coefficients, and is slightly sub-optimal close to boundary)
        assert nsites >= 2
        nodes = DiagonalMolecularOpGraphNodes(nsites)
        graph = nodes.generate_graph()
        # kinetic hopping terms \sum_{i \neq j} t_{i,j} a^{\dagger}_i a_j
        for i in range(nsites):
            for j in range(nsites):
                if i == j:
                    continue
                _diagonal_molecular_hamiltonian_graph_add_term(
                    graph, nodes, [(i, MolecularOID.C), (j, MolecularOID.A)], tkin[i, j])
        # on-site number operators \sum_i u_i n_i
        for i in range(nsites):
            _diagonal_molecular_hamiltonian_graph_add_term(
                graph, nodes, [(i, MolecularOID.N)], uloc[i])
        # interaction terms \sum_{i < j} v_{i,j} n_i n_j
        for i in range(nsites):
            for j in range(i + 1, nsites):  # i < j
                oplist = [(i, MolecularOID.N),
                          (j, MolecularOID.N)]
                _diagonal_molecular_hamiltonian_graph_add_term(
                    graph, nodes, oplist, gint[i, j])

    # skip consistency check for larger nsites (would take very long)
    if nsites <= 12:
        assert graph.is_consistent()
    opmap = _molecular_hamiltonian_generate_operator_map()
    # convert to MPO
    return MPO.from_opgraph([0, 1], graph, opmap, compute_nid_map = not optimize)
