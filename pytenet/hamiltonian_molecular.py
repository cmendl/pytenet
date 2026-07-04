"""
Construct a molecular Hamiltonian as a matrix product operator (MPO).
"""

from collections.abc import Sequence
from enum import IntEnum
import numpy as np
from .mpo import MPO
from .opchain import OpChain
from .opgraph import OpGraphNode, OpGraphEdge, OpGraph

__all__ = ["molecular_hamiltonian_mpo", "molecular_hamiltonian_orbital_gauge_transform"]


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
    z = np.array([[1., 0.], [0., -1.]])
    # operator map
    opmap = {
        MolecularOID.A: a_ann,
        MolecularOID.I: np.identity(2),
        MolecularOID.C: a_dag,
        MolecularOID.N: numop,
        MolecularOID.Z: z
    }
    return opmap


class MolecularOpGraphNodes:
    """
    Operator graph nodes used for molecular Hamiltonian construction.
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
        for i in range(nsites - 2):
            self.a_dag_l[i] = {}
            for j in range(i + 1, nsites - 1):
                self.a_dag_l[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1
        # a_i operators connected to left terminal
        self.a_ann_l = {}
        for i in range(nsites - 2):
            self.a_ann_l[i] = {}
            for j in range(i + 1, nsites - 1):
                self.a_ann_l[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
        self.a_dag_a_dag_l = {}
        for i in range(nsites//2 - 1):
            for j in range(i + 1, nsites//2):
                self.a_dag_a_dag_l[i, j] = {}
                for k in range(j + 1, nsites//2 + 1):
                    self.a_dag_a_dag_l[i, j][k] = OpGraphNode(nid_next, [], [], 2)
                    nid_next += 1
        # a_i a_j operators connected to left terminal
        self.a_ann_a_ann_l = {}
        for i in range(nsites//2):
            for j in range(i):
                self.a_ann_a_ann_l[i, j] = {}
                for k in range(i + 1, nsites//2 + 1):
                    self.a_ann_a_ann_l[i, j][k] = OpGraphNode(nid_next, [], [], -2)
                    nid_next += 1
        # a^{\dagger}_i a_j operators connected to left terminal
        self.a_dag_a_ann_l = {}
        for i in range(nsites//2):
            for j in range(nsites//2):
                self.a_dag_a_ann_l[i, j] = {}
                for k in range(max(i, j) + 1, nsites//2 + 1):
                    self.a_dag_a_ann_l[i, j][k] = OpGraphNode(nid_next, [], [], 0)
                    nid_next += 1
        # a^{\dagger}_i operators connected to right terminal
        self.a_dag_r = {}
        for i in range(2, nsites):
            self.a_dag_r[i] = {}
            for j in range(2, i + 1):
                self.a_dag_r[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # a_i operators connected to right terminal
        self.a_ann_r = {}
        for i in range(2, nsites):
            self.a_ann_r[i] = {}
            for j in range(2, i + 1):
                self.a_ann_r[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
        self.a_dag_a_dag_r = {}
        for i in range(nsites//2 + 1, nsites - 1):
            for j in range(i + 1, nsites):
                self.a_dag_a_dag_r[i, j] = {}
                for k in range(nsites//2 + 1, i + 1):
                    self.a_dag_a_dag_r[i, j][k] = OpGraphNode(nid_next, [], [], -2)
                    nid_next += 1
        # a_i a_j operators connected to right terminal
        self.a_ann_a_ann_r = {}
        for i in range(nsites//2 + 1, nsites):
            for j in range(nsites//2 + 1, i):
                self.a_ann_a_ann_r[i, j] = {}
                for k in range(nsites//2 + 1, j + 1):
                    self.a_ann_a_ann_r[i, j][k] = OpGraphNode(nid_next, [], [], 2)
                    nid_next += 1
        # a^{\dagger}_i a_j operators connected to right terminal
        self.a_dag_a_ann_r = {}
        for i in range(nsites//2 + 1, nsites):
            for j in range(nsites//2 + 1, nsites):
                self.a_dag_a_ann_r[i, j] = {}
                for k in range(nsites//2 + 1, min(i, j) + 1):
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
                return self.a_dag_a_dag_l[i, j] if connection == "left" \
                    else self.a_dag_a_dag_r[i, j]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.A):
                i, j = sorted((i, j), reverse=True)
                return self.a_ann_a_ann_l[i, j] if connection == "left" \
                    else self.a_ann_a_ann_r[i, j]
            if (oid0, oid1) == (MolecularOID.C, MolecularOID.A):
                return self.a_dag_a_ann_l[i, j] if connection == "left" \
                    else self.a_dag_a_ann_r[i, j]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.C):
                return self.a_dag_a_ann_l[j, i] if connection == "left" \
                    else self.a_dag_a_ann_r[j, i]
            raise KeyError(f"nodes for OIDs ({oid0}, {oid1}) and (i, j) == ({i}, {j}) do not exist")
        raise KeyError(f"nodes for operator list of length {len(oplist)} do not exist")

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
                        [node for nodes in self.a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_r.values() for node in nodes.values()],
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
        for i in range(nsites - 2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_dag_l[i][i + 1].nid],
                            [(MolecularOID.C, 1.)]))
            eid_next += 1
            # Z operator from Jordan-Wigner transformation
            for j in range(i + 1, nsites - 2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_l[i][j].nid, self.a_dag_l[i][j + 1].nid],
                                [(MolecularOID.Z, 1.)]))
                eid_next += 1
        # a_i operators connected to left terminal
        for i in range(nsites - 2):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_ann_l[i][i + 1].nid],
                            [(MolecularOID.A, 1.)]))
            eid_next += 1
            # Z operator from Jordan-Wigner transformation
            for j in range(i + 1, nsites - 2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_l[i][j].nid, self.a_ann_l[i][j + 1].nid],
                                [(MolecularOID.Z, 1.)]))
                eid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
        for i in range(nsites//2 - 1):
            for j in range(i + 1, nsites//2):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_l[i][j].nid,
                                           self.a_dag_a_dag_l[i, j][j + 1].nid],
                                [(MolecularOID.C, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(j + 1, nsites//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_l[i, j][k].nid,
                                               self.a_dag_a_dag_l[i, j][k + 1].nid],
                                    [(MolecularOID.I, 1.)]))
                    eid_next += 1
        # a_i a_j operators connected to left terminal
        for i in range(nsites//2):
            for j in range(i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_l[j][i].nid,
                                           self.a_ann_a_ann_l[i, j][i + 1].nid],
                                [(MolecularOID.A, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(i + 1, nsites//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_l[i, j][k].nid,
                                               self.a_ann_a_ann_l[i, j][k + 1].nid],
                                    [(MolecularOID.I, 1.)]))
                    eid_next += 1
        # a^{\dagger}_i a_j operators connected to left terminal
        for i in range(nsites//2):
            for j in range(nsites//2):
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_l[i][j].nid,
                                               self.a_dag_a_ann_l[i, j][j + 1].nid],
                                    [(MolecularOID.A, 1.)]))
                    eid_next += 1
                elif i == j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.identity_l[i].nid,
                                               self.a_dag_a_ann_l[i, j][i + 1].nid],
                                    [(MolecularOID.N, 1.)]))
                    eid_next += 1
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_l[j][i].nid,
                                               self.a_dag_a_ann_l[i, j][i + 1].nid],
                                    [(MolecularOID.C, 1.)]))
                    eid_next += 1
                # identities for transition to next site
                for k in range(max(i, j) + 1, nsites//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_l[i, j][k].nid,
                                               self.a_dag_a_ann_l[i, j][k + 1].nid],
                                    [(MolecularOID.I, 1.)]))
                    eid_next += 1
        # a^{\dagger}_i operators connected to right terminal
        for i in range(2, nsites):
            # Z operator from Jordan-Wigner transformation
            for j in range(2, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_r[i][j].nid, self.a_dag_r[i][j + 1].nid],
                                [(MolecularOID.Z, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_dag_r[i][i].nid, self.identity_r[i + 1].nid],
                            [(MolecularOID.C, 1.)]))
            eid_next += 1
        # a_i operators connected to right terminal
        for i in range(2, nsites):
            # Z operator from Jordan-Wigner transformation
            for j in range(2, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_r[i][j].nid, self.a_ann_r[i][j + 1].nid],
                                [(MolecularOID.Z, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_ann_r[i][i].nid, self.identity_r[i + 1].nid],
                            [(MolecularOID.A, 1.)]))
            eid_next += 1
        # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
        for i in range(nsites//2 + 1, nsites - 1):
            for j in range(i + 1, nsites):
                # identities for transition to next site
                for k in range(nsites//2 + 1, i):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, j][k].nid,
                                               self.a_dag_a_dag_r[i, j][k + 1].nid],
                                    [(MolecularOID.I, 1.)]))
                    eid_next += 1
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, j][i].nid,
                                           self.a_dag_r[j][i + 1].nid],
                                [(MolecularOID.C, 1.)]))
                eid_next += 1
        # a_i a_j operators connected to right terminal
        for i in range(nsites//2 + 1, nsites):
            for j in range(nsites//2 + 1, i):
                # identities for transition to next site
                for k in range(nsites//2 + 1, j):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, j][k].nid,
                                               self.a_ann_a_ann_r[i, j][k + 1].nid],
                                    [(MolecularOID.I, 1.)]))
                    eid_next += 1
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, j][j].nid,
                                           self.a_ann_r[i][j + 1].nid],
                                [(MolecularOID.A, 1.)]))
                eid_next += 1
        # a^{\dagger}_i a_j operators connected to right terminal
        for i in range(nsites//2 + 1, nsites):
            for j in range(nsites//2 + 1, nsites):
                # identities for transition to next site
                for k in range(nsites//2 + 1, min(i, j)):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, j][k].nid,
                                               self.a_dag_a_ann_r[i, j][k + 1].nid],
                                    [(MolecularOID.I, 1.)]))
                    eid_next += 1
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, j][i].nid,
                                               self.a_ann_r[j][i + 1].nid],
                                    [(MolecularOID.C, 1.)]))
                    eid_next += 1
                elif i == j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, j][i].nid,
                                               self.identity_r[i + 1].nid],
                                    [(MolecularOID.N, 1.)]))
                    eid_next += 1
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, j][j].nid,
                                               self.a_dag_r[i][j + 1].nid],
                                    [(MolecularOID.A, 1.)]))
                    eid_next += 1

        return graph

    def copy_nids(self, target):
        """
        Transfer and store the node IDs in the target object.
        """
        nsites = self.nsites
        # identity chains from the left and right
        target.nids_identity_l = {}
        target.nids_identity_r = {}
        for i in range(nsites):
            target.nids_identity_l[i] = self.identity_l[i].nid
        for i in range(1, nsites + 1):
            target.nids_identity_r[i] = self.identity_r[i].nid
        # a^{\dagger}_i operators connected to left terminal
        target.nids_a_dag_l = {}
        for i in range(nsites - 2):
            target.nids_a_dag_l[i] = {}
            for j in range(i + 1, nsites - 1):
                target.nids_a_dag_l[i][j] = self.a_dag_l[i][j].nid
        # a_i operators connected to left terminal
        target.nids_a_ann_l = {}
        for i in range(nsites - 2):
            target.nids_a_ann_l[i] = {}
            for j in range(i + 1, nsites - 1):
                target.nids_a_ann_l[i][j] = self.a_ann_l[i][j].nid
        # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
        target.nids_a_dag_a_dag_l = {}
        for i in range(nsites//2 - 1):
            for j in range(i + 1, nsites//2):
                target.nids_a_dag_a_dag_l[i, j] = {}
                for k in range(j + 1, nsites//2 + 1):
                    target.nids_a_dag_a_dag_l[i, j][k] = self.a_dag_a_dag_l[i, j][k].nid
        # a_i a_j operators connected to left terminal
        target.nids_a_ann_a_ann_l = {}
        for i in range(nsites//2):
            for j in range(i):
                target.nids_a_ann_a_ann_l[i, j] = {}
                for k in range(i + 1, nsites//2 + 1):
                    target.nids_a_ann_a_ann_l[i, j][k] = self.a_ann_a_ann_l[i, j][k].nid
        # a^{\dagger}_i a_j operators connected to left terminal
        target.nids_a_dag_a_ann_l = {}
        for i in range(nsites//2):
            for j in range(nsites//2):
                target.nids_a_dag_a_ann_l[i, j] = {}
                for k in range(max(i, j) + 1, nsites//2 + 1):
                    target.nids_a_dag_a_ann_l[i, j][k] = self.a_dag_a_ann_l[i, j][k].nid
        # a^{\dagger}_i operators connected to right terminal
        target.nids_a_dag_r = {}
        for i in range(2, nsites):
            target.nids_a_dag_r[i] = {}
            for j in range(2, i + 1):
                target.nids_a_dag_r[i][j] = self.a_dag_r[i][j].nid
        # a_i operators connected to right terminal
        target.nids_a_ann_r = {}
        for i in range(2, nsites):
            target.nids_a_ann_r[i] = {}
            for j in range(2, i + 1):
                target.nids_a_ann_r[i][j] = self.a_ann_r[i][j].nid
        # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
        target.nids_a_dag_a_dag_r = {}
        for i in range(nsites//2 + 1, nsites - 1):
            for j in range(i + 1, nsites):
                target.nids_a_dag_a_dag_r[i, j] = {}
                for k in range(nsites//2 + 1, i + 1):
                    target.nids_a_dag_a_dag_r[i, j][k] = self.a_dag_a_dag_r[i, j][k].nid
        # a_i a_j operators connected to right terminal
        target.nids_a_ann_a_ann_r = {}
        for i in range(nsites//2 + 1, nsites):
            for j in range(nsites//2 + 1, i):
                target.nids_a_ann_a_ann_r[i, j] = {}
                for k in range(nsites//2 + 1, j + 1):
                    target.nids_a_ann_a_ann_r[i, j][k] = self.a_ann_a_ann_r[i, j][k].nid
        # a^{\dagger}_i a_j operators connected to right terminal
        target.nids_a_dag_a_ann_r = {}
        for i in range(nsites//2 + 1, nsites):
            for j in range(nsites//2 + 1, nsites):
                target.nids_a_dag_a_ann_r[i, j] = {}
                for k in range(nsites//2 + 1, min(i, j) + 1):
                    target.nids_a_dag_a_ann_r[i, j][k] = self.a_dag_a_ann_r[i, j][k].nid


def _molecular_hamiltonian_graph_add_term(graph: OpGraph, nodes: MolecularOpGraphNodes,
                                          oplist: Sequence[tuple], coeff: float):
    """
    Add an operator term (operator string of creation and annihilation operators)
    to the operator graph describing a molecular Hamiltonian.
    """
    eid_next = max(graph.edges.keys()) + 1

    nsites = nodes.nsites

    # sort by site (orbital) index
    oplist = sorted(oplist)

    if len(oplist) == 2:
        (i, oid0), (j, oid1) = oplist
        if i == j:
            # expecting number operator
            assert (oid0, oid1) == (MolecularOID.A, MolecularOID.C)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid],
                            [(MolecularOID.N, coeff)]))
        else:
            assert i < j
            if j <= nsites//2:
                nodes_l = nodes.get([(i, oid0)], "left")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes.identity_r[j + 1].nid],
                                [(oid1, coeff)]))
            elif i >= nsites//2:
                nodes_r = nodes.get([(j, oid1)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes_r[i + 1].nid],
                                [(oid0, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid0)], "left")
                nodes_r = nodes.get([(j, oid1)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[nsites//2].nid, nodes_r[nsites//2 + 1].nid],
                                [(MolecularOID.Z, coeff)]))
    elif len(oplist) == 4:
        (i, oid0), (j, oid1), (k, oid2), (l, oid3) = oplist
        if j == k:
            # expecting number operator
            assert (oid1, oid2) == (MolecularOID.A, MolecularOID.C)
            nodes_l = nodes.get([(i, oid0)], "left")
            nodes_r = nodes.get([(l, oid3)], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid],
                            [(MolecularOID.N, coeff)]))
        elif k <= nsites//2:
            nodes_l = nodes.get([(i, oid0), (j, oid1)], "left")
            if k == l:
                # expecting number operator
                assert (oid2, oid3) == (MolecularOID.A, MolecularOID.C)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes.identity_r[k + 1].nid],
                                [(MolecularOID.N, coeff)]))
            else:
                nodes_r = nodes.get([(l, oid3)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes_r[k + 1].nid], [(oid2, coeff)]))
        elif j >= nsites//2:
            nodes_r = nodes.get([(k, oid2), (l, oid3)], "right")
            if i == j:
                # expecting number operator
                assert (oid0, oid1) == (MolecularOID.A, MolecularOID.C)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[j].nid, nodes_r[j + 1].nid],
                                [(MolecularOID.N, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid0)], "left")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid], [(oid1, coeff)]))
        else:
            nodes_l = nodes.get([(i, oid0), (j, oid1)], "left")
            nodes_r = nodes.get([(k, oid2), (l, oid3)], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[nsites//2].nid, nodes_r[nsites//2 + 1].nid],
                            [(MolecularOID.I, coeff)]))
    else:
        raise NotImplementedError


def _molecular_hamiltonian_bipartite_graph(tkin, gint) -> OpGraph:
    """
    Construct a molecular Hamiltonian via operator chains optimized based on bipartite graph theory.
    """
    nsites = tkin.shape[0]
    # optimize MPO bond dimensions based on bipartite graph theory
    opchains = []
    # kinetic hopping terms \sum_{i,j} t_{i,j} a^{\dagger}_i a_j
    for i in range(nsites):
        for j in range(nsites):
            if i == j:
                # diagonal hopping term
                opchains.append(OpChain([MolecularOID.N], [0, 0], tkin[i, i], i))
            else:
                (a, p), (b, q) = sorted([(i, MolecularOID.C), (j, MolecularOID.A)])
                opchains.append(OpChain([p] + (b - a - 1)*[MolecularOID.Z] + [q],
                                        [0] + (b - a)*[int(p)] + [0], tkin[i, j], a))
    # interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k
    for i in range(nsites):
        for j in range(i + 1, nsites):  # i < j
            for k in range(nsites):
                for l in range(k + 1, nsites):  # k < l
                    (a, p), (b, q), (c, r), (d, s) = sorted([(i, MolecularOID.C),
                                                             (j, MolecularOID.C),
                                                             (l, MolecularOID.A),
                                                             (k, MolecularOID.A)])
                    if a == b:
                        assert b < c
                        if c == d:
                            # two number operators
                            oids  = (  [MolecularOID.N]
                                     + (c - b - 1)*[MolecularOID.I]
                                     + [MolecularOID.N])
                            qnums = (c - b + 2)*[0]
                        else:
                            # number operator at the beginning
                            oids  = (  [MolecularOID.N]
                                     + (c - b - 1)*[MolecularOID.I]
                                     + [r]
                                     + (d - c - 1)*[MolecularOID.Z]
                                     + [s])
                            qnums = (c - b + 1)*[0] + (d - c)*[int(r)] + [0]
                    elif b == c:
                        # number operator in the middle
                        oids  = (  [p]
                                 + (b - a - 1)*[MolecularOID.Z]
                                 + [MolecularOID.N]
                                 + (d - c - 1)*[MolecularOID.Z]
                                 + [s])
                        qnums = [0] + (d - a)*[int(p)] + [0]
                    elif c == d:
                        # number operator at the end
                        oids  = (  [p]
                                 + (b - a - 1)*[MolecularOID.Z]
                                 + [q]
                                 + (c - b - 1)*[MolecularOID.I]
                                 + [MolecularOID.N])
                        qnums = [0] + (b - a)*[int(p)] + (c - b + 1)*[0]
                    else:
                        # generic case: i, j, k, l pairwise different
                        oids  = (  [p]
                                 + (b - a - 1)*[MolecularOID.Z]
                                 + [q]
                                 + (c - b - 1)*[MolecularOID.I]
                                 + [r]
                                 + (d - c - 1)*[MolecularOID.Z]
                                 + [s])
                        qnums = (  [0]
                                 + (b - a)*[int(p)]
                                 + (c - b)*[int(p) + int(q)]
                                 + (d - c)*[-int(s)]
                                 + [0])
                    opchains.append(OpChain(oids, qnums, gint[i, j, k, l], a))
    return OpGraph.from_opchains(opchains, nsites, 0)


def molecular_hamiltonian_mpo(tkin, vint, optimize=True) -> MPO:
    r"""
    Construct a molecular Hamiltonian as an MPO,
    using physicists' convention for the interaction term (note ordering of k and \ell):

    .. math::

        H = \sum_{i,j} t_{i,j} a^{\dagger}_i a_j
            + \frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell}
              a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    nsites = tkin.shape[0]
    assert tkin.shape == (nsites, nsites)
    assert vint.shape == (nsites, nsites, nsites, nsites)

    # interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k:
    # can anti-commute fermionic operators such that i < j and k < l
    gint = 0.5 * (vint
                  - np.transpose(vint, (1, 0, 2, 3))
                  - np.transpose(vint, (0, 1, 3, 2))
                  + np.transpose(vint, (1, 0, 3, 2)))

    if optimize:
        graph = _molecular_hamiltonian_bipartite_graph(tkin, gint)
    else:
        # explicit construction (typically faster, but does not optimize cases
        # of zero coefficients, and is slightly sub-optimal close to boundary)
        assert nsites >= 4
        nodes = MolecularOpGraphNodes(nsites)
        graph = nodes.generate_graph()
        # kinetic hopping terms \sum_{i,j} t_{i,j} a^{\dagger}_i a_j
        for i in range(nsites):
            for j in range(nsites):
                _molecular_hamiltonian_graph_add_term(
                    graph, nodes, [(i, MolecularOID.C), (j, MolecularOID.A)], tkin[i, j])
        # interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k
        for i in range(nsites):
            for j in range(i + 1, nsites):  # i < j
                for k in range(nsites):
                    for l in range(k + 1, nsites):  # k < l
                        oplist = [(i, MolecularOID.C),
                                  (j, MolecularOID.C),
                                  (l, MolecularOID.A),
                                  (k, MolecularOID.A)]
                        _molecular_hamiltonian_graph_add_term(
                            graph, nodes, oplist, gint[i, j, k, l])

    # skip consistency check for larger nsites (would take very long)
    if nsites <= 12:
        assert graph.is_consistent()
    opmap = _molecular_hamiltonian_generate_operator_map()
    # convert to MPO
    mpo = MPO.from_opgraph([0, 1], graph, opmap, compute_nid_map = not optimize)
    # store node information in MPO, to identify virtual
    # bonds by creation and annihilation operators
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
