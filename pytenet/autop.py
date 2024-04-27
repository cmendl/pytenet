from collections.abc import Sequence
from typing import Callable, Union

__all__ = ['AutOpNode', 'AutOpEdge', 'AutOp']


class AutOpNode:
    """
    Operator state automaton node, corresponding to a virtual bond in an MPO.
    """
    def __init__(self, nid: int, eids_in: Sequence[int], eids_out: Sequence[int], qnum: int):
        assert len(eids_in)  == len(set(eids_in)),  f'incoming edge indices must be pairwise different, received {eids_in}'
        assert len(eids_out) == len(set(eids_out)), f'outgoing edge indices must be pairwise different, received {eids_out}'
        self.nid = nid
        self.eids = (list(eids_in), list(eids_out))
        self.qnum = qnum

    def add_edge_id(self, eid: int, direction: int):
        """
        Add an edge identified by ID 'eid' in the specified direction.
        """
        eids = self.eids[direction]
        assert eid not in eids
        eids.append(eid)


class AutOpEdge:
    """
    Operator state automaton directed edge, representing a weighted sum of local operators
    which are indexed by their IDs.

    An edge can loop from a node to the same node.
    """
    def __init__(self, eid: int, nids: Sequence[int],
                 opics: Union[Sequence[tuple[int, float]], Callable[int, Sequence[tuple[int, float]]]],
                 active: Union[bool, Callable[int, bool]]=True):
        if len(nids) != 2:
            raise ValueError(f'expecting exactly two node IDs per edge, received {len(nids)}')
        self.eid    = eid
        self.nids   = list(nids)
        self.opics  = opics
        self.active = active


class AutOp:
    """
    Operator state automaton, can be converted to an MPO via an operator graph.
    """
    def __init__(self, nodes: Sequence[AutOpNode], edges: Sequence[AutOpEdge], nid_terminal: Sequence[int]):
        # dictionary of nodes
        self.nodes = {}
        for node in nodes:
            self.add_node(node)
        # terminal node IDs
        if len(nid_terminal) != 2:
            raise ValueError(f'expecting two terminal node IDs, received {len(nid_terminal)}')
        if nid_terminal[0] not in self.nodes or nid_terminal[1] not in self.nodes:
            raise ValueError(f'terminal node IDs {nid_terminal} not found')
        self.nid_terminal = list(nid_terminal)
        # dictionary of edges
        self.edges = {}
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node: AutOpNode):
        """
        Add a node to the graph.
        """
        if node.nid in self.nodes:
            raise ValueError(f'node with ID {node.nid} already exists')
        self.nodes[node.nid] = node

    def remove_node(self, nid: int) -> AutOpNode:
        """
        Remove a node from the graph, and return the removed node.
        """
        return self.nodes.pop(nid)

    def add_edge(self, edge: AutOpEdge):
        """
        Add an edge to the graph.
        """
        if edge.eid in self.edges:
            raise ValueError(f'edge with ID {edge.eid} already exists')
        self.edges[edge.eid] = edge

    def add_connect_edge(self, edge: AutOpEdge):
        """
        Add an edge to the graph, and connect nodes referenced by the edge to it.
        """
        self.add_edge(edge)
        # connect nodes back to edge
        for direction in (0, 1):
            if edge.nids[direction] in self.nodes:
                self.nodes[edge.nids[direction]].add_edge_id(edge.eid, 1-direction)

    def remove_edge(self, eid: int) -> AutOpEdge:
        """
        Remove an edge from the graph, and return the removed edge.
        """
        return self.edges.pop(eid)

    def is_consistent(self, verbose: bool = False) -> bool:
        """
        Perform an internal consistency check.
        """
        for k, node in self.nodes.items():
            if k != node.nid:
                if verbose:
                    print(f'Consistency check failed: dictionary key {k} does not match node ID {node.nid}.')
                return False
            for direction in (0, 1):
                for eid in node.eids[direction]:
                    # edge with ID 'eid' must exist
                    if eid not in self.edges:
                        if verbose:
                            print(f'Consistency check failed: edge with ID {eid} '
                                  f'referenced by node {k} does not exist.')
                        return False
                    # edge must refer back to node
                    edge = self.edges[eid]
                    if edge.nids[1-direction] != node.nid:
                        if verbose:
                            print(f'Consistency check failed: edge with ID {eid} '
                                  f'does not refer to node {node.nid}.')
                        return False
        for k, edge in self.edges.items():
            if k != edge.eid:
                if verbose:
                    print(f'Consistency check failed: dictionary key {k} does not match edge ID {edge.eid}.')
                return False
            for direction in (0, 1):
                if edge.nids[direction] not in self.nodes:
                    if verbose:
                        print(f'Consistency check failed: node with ID {edge.nids[direction]} '
                              f'referenced by edge {k} does not exist.')
                    return False
                node = self.nodes[edge.nids[direction]]
                if edge.eid not in node.eids[1-direction]:
                    if verbose:
                        print(f'Consistency check failed: node {node.nid} does not refer to edge {edge.eid}.')
                    return False
        for direction in (0, 1):
            if self.nid_terminal[direction] not in self.nodes:
                if verbose:
                    print(f'Consistency check failed: terminal node ID {self.nid_terminal[direction]} not found.')
                return False
        return True
