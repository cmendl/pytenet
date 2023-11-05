from collections.abc import Sequence, Mapping
from itertools import combinations
import copy
import numpy as np
from .opchain import OpChain
from .optree import OpTreeEdge, OpTreeNode, OpTree

__all__ = ['OpGraphNode', 'OpGraphEdge', 'OpGraph']


class OpGraphNode:
    """
    Operator graph node, corresponding to a virtual bond in an MPO.
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

    def remove_edge_id(self, eid: int, direction: int):
        """
        Remove an edge identified by ID 'eid' in the specified direction.
        """
        self.eids[direction].remove(eid)

    def rename_edge_id(self, eid_cur: int, eid_new: int, direction: int):
        """
        Rename the ID an edge.
        """
        self.remove_edge_id(eid_cur, direction)
        self.add_edge_id(eid_new, direction)

    def flip(self):
        """
        Flip logical input <-> output direction.
        """
        self.eids = tuple(reversed(self.eids))
        return self


class OpGraphEdge:
    """
    Operator graph edge, representing a weighted sum of local operators
    which are indexed by their IDs.
    """
    def __init__(self, eid: int, nids: Sequence[int], opics: Sequence[tuple[int, float]]):
        if len(nids) != 2:
            raise ValueError(f'expecting exactly two node IDs per edge, received {len(nids)}')
        self.eid   = eid
        self.nids  = list(nids)
        self.opics = []
        for i, c in opics:
            # ensure that each index is unique
            for k in range(len(self.opics)):
                if self.opics[k][0] == i:
                    j, d = self.opics.pop(k)
                    assert i == j
                    # re-insert tuple with added coefficients
                    self.opics.append((i, c + d))
                    break
            else:
                # index 'i' not found so far
                self.opics.append((i, c))
        # sort by index
        self.opics = sorted(self.opics)

    def flip(self):
        """
        Flip logical direction.
        """
        self.nids.reverse()
        return self

    def add(self, other):
        """
        Logical addition of the operation represented by another edge.
        """
        assert self.nids == other.nids
        for i, c in other.opics:
            for k in range(len(self.opics)):
                if self.opics[k][0] == i:
                    j, d = self.opics.pop(k)
                    assert i == j
                    # re-insert tuple with added coefficients
                    self.opics.append((i, c + d))
                    break
            else:
                # index 'i' not found so far
                self.opics.append((i, c))
        # sort by index
        self.opics = sorted(self.opics)


class OpGraph:
    """
    Operator graph: internal data structure for generating MPO representations.
    """
    def __init__(self, nodes: Sequence[OpGraphNode], edges: Sequence[OpGraphEdge], nid_terminal: Sequence[int]):
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

    def add_node(self, node: OpGraphNode):
        """
        Add a node to the graph.
        """
        if node.nid in self.nodes:
            raise ValueError(f'node with ID {node.nid} already exists')
        self.nodes[node.nid] = node

    def remove_node(self, nid: int) -> OpGraphNode:
        """
        Remove a node from the graph, and return the removed node.
        """
        return self.nodes.pop(nid)

    def add_edge(self, edge: OpGraphEdge):
        """
        Add an edge to the graph.
        """
        if edge.eid in self.edges:
            raise ValueError(f'edge with ID {edge.eid} already exists')
        self.edges[edge.eid] = edge

    def remove_edge(self, eid: int) -> OpGraphEdge:
        """
        Remove an edge from the graph, and return the removed edge.
        """
        return self.edges.pop(eid)

    def node_depth(self, nid: int, direction: int) -> int:
        """
        Determine the depth of a node (distance to terminal node in specified direction),
        assuming that a corresponding path exists within the graph.
        """
        depth = 0
        node = self.nodes[nid]
        while node.eids[direction]:
            # follow first connection
            edge = self.edges[node.eids[direction][0]]
            node = self.nodes[edge.nids[direction]]
            depth += 1
        return depth

    @property
    def length(self) -> int:
        """
        Length of the graph (distance between terminal nodes).
        """
        return self.node_depth(self.nid_terminal[0], 1)

    def _insert_opchain(self, nid_start: int, nid_end: int, oids: Sequence[int], coeffs: Sequence[float], qnums: Sequence[int], direction: int):
        """
        Insert an operator chain between two nodes
        by generating an alternating sequence of edges and nodes.
        """
        assert nid_start in self.nodes
        assert nid_end in self.nodes
        assert len(oids) == len(coeffs)
        assert len(oids) == len(qnums) + 1
        # next available node and edge ID
        nid_next = max(self.nodes.keys()) + 1
        eid_next = max(self.edges.keys(), default=0) + 1
        node = self.nodes[nid_start]
        for oid, coeff, qnum in zip(oids[:-1], coeffs[:-1], qnums):
            node.add_edge_id(eid_next, direction)
            self.add_edge(OpGraphEdge(eid_next, [nid_next, node.nid] if direction == 0 else [node.nid, nid_next], [(oid, coeff)]))
            node = OpGraphNode(nid_next,
                               [] if direction == 0 else [eid_next],
                               [eid_next] if direction == 0 else [],
                               qnum)
            self.add_node(node)
            nid_next += 1
            eid_next += 1
        # last step
        node.add_edge_id(eid_next, direction)
        self.add_edge(OpGraphEdge(eid_next, [nid_end, node.nid] if direction == 0 else [node.nid, nid_end], [(oids[-1], coeffs[-1])]))
        node = self.nodes[nid_end]
        node.add_edge_id(eid_next, 1-direction)

    @classmethod
    def from_opchains(cls, chains: Sequence[OpChain], length: int, oid_identity: int):
        """
        Construct an operator graph from a list of operator chains.

        Args:
            chains: list of operator chains
            length: overall length of the operator graph
            oid_identity: operator ID for identity map

        Returns:
            OpGraph: the constructed operator graph
        """
        # construct graph with two terminal nodes
        graph = cls([OpGraphNode(0, [], [], 0),
                     OpGraphNode(1, [], [], 0)], [], [0, 1])
        for chain in chains:
            if chain.istart + chain.length > length:
                raise ValueError('extent of operator chain cannot be larger than overall length of operator graph')
            if chain.length == 0:
                continue
            if chain.qnums[0] != 0 or chain.qnums[-1] != 0:
                raise ValueError('expecting quantum number zero at beginning and end of each chain')
            oids = chain.oids
            # include coefficient (arbitrarily) with first operator of the chain
            coeffs = len(chain.oids) * [1.0]
            coeffs[0] = chain.coeff
            qnums = chain.qnums
            if chain.istart > 0:
                # pad identities before the chain
                oids   = chain.istart * [oid_identity] + oids
                coeffs = chain.istart * [1.0] + coeffs
                qnums  = chain.istart * [qnums[0]] + qnums
            if len(oids) < length:
                # pad identities after the chain
                n = length - len(oids)
                oids   = oids   + n * [oid_identity]
                coeffs = coeffs + n * [1.0]
                qnums  = qnums  + n * [qnums[-1]]
            assert len(oids) == length
            graph._insert_opchain(0, 1, oids, coeffs, qnums[1:-1], 1)
        graph.simplify()
        return graph

    def _insert_subtree(self, tree_root: OpTreeNode, nid_root: int, terminal_dist: int, oid_identity: int):
        """
        Insert a subtree into the graph, connecting the leaves with the terminal node of the graph.
        """
        if not isinstance(tree_root, OpTreeNode):
            raise ValueError("'tree_root' must be of type 'OpTreeNode'")
        if terminal_dist < 0:
            raise ValueError("'terminal_dist' cannot be negative")
        # operator graph node
        node = self.nodes[nid_root]
        if node.qnum != tree_root.qnum:
            raise RuntimeError(f'quantum number of operator graph node ({node.qnum}) '
                               f'does not match quantum number of tree node ({tree_root.qnum})')
        if not tree_root.children:
            # arrived at leaf node
            if terminal_dist > 0:
                # insert identity string to end node of graph
                self._insert_opchain(nid_root, self.nid_terminal[1],
                                     terminal_dist * [oid_identity],
                                     terminal_dist * [1.0],
                                     (terminal_dist - 1) * [0], 1)
            else:
                # 'terminal_dist' is zero -> have to be at terminal node
                assert nid_root == self.nid_terminal[1]
            return
        # recursively insert sub-trees
        for edge in tree_root.children:
            assert isinstance(edge, OpTreeEdge)
            # next available node and edge ID
            if terminal_dist > 1:
                nid_next = max(self.nodes.keys()) + 1
            else:
                nid_next = self.nid_terminal[1]
            eid_next = max(self.edges.keys(), default=0) + 1
            node.add_edge_id(eid_next, 1)
            self.add_edge(OpGraphEdge(eid_next, [node.nid, nid_next], [(edge.oid, edge.coeff)]))
            if terminal_dist > 1:
                self.add_node(OpGraphNode(nid_next, [eid_next], [], edge.node.qnum))
            else:
                self.nodes[nid_next].add_edge_id(eid_next, 0)
            self._insert_subtree(edge.node, nid_next, terminal_dist - 1, oid_identity)

    @classmethod
    def from_optrees(cls, trees: Sequence[OpTree], length: int, oid_identity: int):
        """
        Construct an operator graph from a list of operator trees.

        Args:
            trees: list of operator trees
            length: overall length of the operator graph
            oid_identity: operator ID of the identity operation, required for
                          padding identity operations to the terminal nodes of the graph

        Returns:
            OpGraph: the constructed operator graph
        """
        nid_start = 0
        graph = cls([OpGraphNode(nid_start, [], [], 0), OpGraphNode(1, [], [], 0)], [], [0, 1])
        for tree in trees:
            if tree.istart > 0:
                # insert identities between start node and beginning of tree
                nid_root = max(graph.nodes.keys()) + 1
                graph.add_node(OpGraphNode(nid_root, [], [], tree.root.qnum))
                graph._insert_opchain(nid_start, nid_root,
                                      tree.istart * [oid_identity],
                                      tree.istart * [1.0], (tree.istart - 1) * [0], 1)
            else:
                nid_root = nid_start
            graph._insert_subtree(tree.root, nid_root, length - tree.istart, oid_identity)
        graph.simplify()
        return graph

    def merge_edges(self, eid1: int, eid2: int, direction: int):
        """
        Merge paths along edges `eid1` and `eid2` in upstream (0) or downstream (1) direction.
        Edges must originate from same node.
        """
        if direction not in (0, 1):
            raise ValueError(f"'direction' must be 0 or 1, received {direction}")
        edge1 = self.edges[eid1]
        edge2 = self.edges.pop(eid2)
        assert edge1.nids[direction] == edge2.nids[direction], 'to-be merged edges must originate from same node'
        # remove reference to edge2 from base node
        self.nodes[edge2.nids[direction]].remove_edge_id(edge2.eid, 1-direction)
        if edge1.nids[1-direction] == edge2.nids[1-direction]:
            # edges have same upstream node -> add operators
            edge1.add(edge2)
            # remove reference to edge2 from upstream node
            self.nodes[edge2.nids[1-direction]].remove_edge_id(edge2.eid, direction)
            return
        assert edge1.opics == edge2.opics, 'can only merge edges with same logical operators'
        # merge upstream nodes
        node1 = self.nodes[edge1.nids[1-direction]]
        node2 = self.nodes.pop(edge2.nids[1-direction])
        assert len(node1.eids[direction]) == 1, 'to-be merged upstream node can only have one input edge'
        assert len(node2.eids[direction]) == 1, 'to-be merged upstream node can only have one input edge'
        assert node1.qnum == node2.qnum, f'can only merge nodes with same quantum numbers, encountered {node1.qnum} and {node2.qnum}'
        # make former edges from node2 to point to node1
        for eid in node2.eids[1-direction]:
            self.edges[eid].nids[direction] = node1.nid
        # explicit variable assignment to avoid "tuple item assignment" error
        node1_eids = node1.eids[1-direction]
        node1_eids += node2.eids[1-direction]

    def simplify(self):
        """
        Simplify the graph in-place by merging edges representing the same operator.
        """
        changed = True
        while changed:
            changed = False
            for direction in (0, 1):
                while self._simplify_step(direction):
                    changed = True
        # enable chaining
        return self

    def _simplify_step(self, direction: int) -> bool:
        """
        Attempt an in-place graph simplification step by merging two edges in
        the specified direction. Returns true if two edges were merged, and
        false otherwise (in which case the graph was not changed).
        """
        # node IDs at current bond site
        nids0 = [self.nid_terminal[direction]]
        while True:
            for nid in nids0:
                # search for edge pairs which can be merged
                eids = self.nodes[nid].eids[1-direction]
                for eid1, eid2 in combinations(eids, 2):
                    edge1 = self.edges[eid1]
                    edge2 = self.edges[eid2]
                    if edge1.nids[1-direction] == edge2.nids[1-direction]:
                        # edges have same upstream node
                        self.merge_edges(eid1, eid2, direction)
                        return True
                    if edge1.opics != edge2.opics:
                        continue
                    node1 = self.nodes[edge1.nids[1-direction]]
                    node2 = self.nodes[edge2.nids[1-direction]]
                    # to-be merged upstream nodes can only have one input edge
                    if len(node1.eids[direction]) != 1:
                        continue
                    if len(node2.eids[direction]) != 1:
                        continue
                    # can only merge nodes with same quantum numbers
                    if node1.qnum != node2.qnum:
                        continue
                    # actually merge the edges
                    self.merge_edges(eid1, eid2, direction)
                    return True
            # collect node IDs at next bond site
            nids1 = []
            for nid in nids0:
                eids = self.nodes[nid].eids[1-direction]
                for eid in eids:
                    edge = self.edges[eid]
                    assert edge.nids[direction] == nid
                    if edge.nids[1-direction] not in nids1:
                        nids1.append(edge.nids[1-direction])
            if not nids1:   # reached final site
                break
            nids0 = nids1
        # no edges merged
        return False

    def flip(self):
        """
        Flip logical direction of the graph.
        """
        for node in self.nodes.values():
            node.flip()
        for edge in self.edges.values():
            edge.flip()
        self.nid_terminal.reverse()

    def rename_node_id(self, nid_cur: int, nid_new: int):
        """
        Rename node ID `nid_cur` -> `nid_new`.
        """
        if nid_cur not in self.nodes:
            raise ValueError(f"node with ID {nid_cur} does not exist")
        if nid_new in self.nodes:
            raise ValueError(f"node with ID {nid_new} already exists")
        node = self.remove_node(nid_cur)
        assert node.nid == nid_cur
        for direction in (0, 1):
            # update reference to node by edges
            for eid in node.eids[direction]:
                edge = self.edges[eid]
                assert edge.nids[1-direction] == nid_cur
                edge.nids[1-direction] = nid_new
            # update terminal node IDs
            if self.nid_terminal[direction] == nid_cur:
                self.nid_terminal[direction] = nid_new
        node.nid = nid_new
        self.add_node(node)

    def rename_edge_id(self, eid_cur: int, eid_new: int):
        """
        Rename edge ID `eid_cur` -> `eid_new`.
        """
        if eid_cur not in self.edges:
            raise ValueError(f"edge with ID {eid_cur} does not exist")
        if eid_new in self.edges:
            raise ValueError(f"edge with ID {eid_new} already exists")
        edge = self.remove_edge(eid_cur)
        assert edge.eid == eid_cur
        for direction in (0, 1):
            # update reference to edge by nodes
            node = self.nodes[edge.nids[direction]]
            node.rename_edge_id(eid_cur, eid_new, 1-direction)
        edge.eid = eid_new
        self.add_edge(edge)

    def add(self, other):
        """
        Add another graph by merging it into the current one.
        Assuming that the operator IDs are shared between the graphs.
        """
        if not isinstance(other, OpGraph):
            raise ValueError('can only update an operator graph by another operator graph')
        # require a deep copy since the IDs in the 'other' graph might change
        other = copy.deepcopy(other)
        # ensure that node IDs in the two graphs are disjoint
        shared_nids = self.nodes.keys() & other.nodes.keys()
        next_nid = max(max(self.nodes.keys()), max(other.nodes.keys())) + 1
        for nid in shared_nids:
            other.rename_node_id(nid, next_nid)
            next_nid += 1
        # ensure that edge IDs in the two graphs are disjoint
        shared_eids = self.edges.keys() & other.edges.keys()
        next_eid = max(max(self.edges.keys(), default=0), max(other.edges.keys(), default=0)) + 1
        for eid in shared_eids:
            other.rename_edge_id(eid, next_eid)
            next_eid += 1
        # use same identifiers for terminal nodes
        for direction in (0, 1):
            other.rename_node_id(other.nid_terminal[direction], self.nid_terminal[direction])
        # integrate terminal nodes from 'other' graph
        for direction in (0, 1):
            tnode = other.remove_node(other.nid_terminal[direction])
            assert not tnode.eids[direction]
            eids = self.nodes[self.nid_terminal[direction]].eids[1-direction]
            eids += tnode.eids[1-direction]
        # include remaining nodes from other graph
        self.nodes.update(other.nodes)
        # include edges from other graph
        self.edges.update(other.edges)
        # simplify updated graph
        self.simplify()
        # enable chaining
        return self

    def as_matrix(self, opmap: Mapping, direction: int = 1) -> np.ndarray:
        """
        Represent the logical operation of the operator graph as a matrix.
        """
        return _subgraph_as_matrix(self, self.nid_terminal[1-direction], opmap, direction)

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
            if edge.opics != sorted(edge.opics):
                if verbose:
                    print(f'Consistency check failed: list of operator IDs of edge {edge.eid} is not sorted.')
                return False
        for direction in (0, 1):
            if self.nid_terminal[direction] not in self.nodes:
                if verbose:
                    print(f'Consistency check failed: terminal node ID {self.nid_terminal[direction]} not found.')
                return False
            node = self.nodes[self.nid_terminal[direction]]
            if node.eids[direction]:
                if verbose:
                    print(f'Consistency check failed: terminal node in direction {direction} '
                          f'cannot have edges in that direction.')
                return False
        for direction in (0, 1):
            # node levels (distance from start and end node)
            node_level_map = {}
            nid_start = self.nid_terminal[direction]
            nid_queue = [(nid_start, 0)]
            while nid_queue:
                nid, level = nid_queue.pop(0)
                if nid in node_level_map:
                    if level != node_level_map[nid]:
                        if verbose:
                            print(f'Consistency check failed: level of node {nid} is inconsistent.')
                        return False
                else:
                    node_level_map[nid] = level
                node = self.nodes[nid]
                # insert nodes at next level into queue
                for eid in node.eids[1-direction]:
                    nid_queue.append((self.edges[eid].nids[1-direction], level + 1))
        return True


def _subgraph_as_matrix(graph: OpGraph, nid: int, opmap: Mapping, direction: int) -> np.ndarray:
    """
    Contract the (sub-)graph in the specified direction to obtain its matrix representation.
    """
    if nid == graph.nid_terminal[direction]:
        return np.identity(1)
    op_sum = 0
    node = graph.nodes[nid]
    assert node.eids[direction], f'encountered dangling node {nid} in direction {direction}'
    for eid in node.eids[direction]:
        edge = graph.edges[eid]
        op_sub = _subgraph_as_matrix(graph, edge.nids[direction], opmap, direction)
        op_loc = sum(c * opmap[i] for i, c in edge.opics)
        if direction == 0:
            op = np.kron(op_sub, op_loc)
        else:
            op = np.kron(op_loc, op_sub)
        # not using += here to allow for "up-casting" to complex entries
        op_sum = op_sum + op
    return op_sum
