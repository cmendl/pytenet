from typing import Sequence

# data structures in this file are intended for internal usage
__all__ = []


class OpGraphNode:
    """
    Operator graph node, corresponding to a virtual bond in an MPO.
    """
    def __init__(self, nid: int, eids_in: Sequence[int], eids_out: Sequence[int], qnum: int):
        assert len(eids_in)  == len(set(eids_in)),  f"incoming edge indices must be pairwise different, received {eids_in}"
        assert len(eids_out) == len(set(eids_out)), f"outgoing edge indices must be pairwise different, received {eids_out}"
        self.nid = nid
        self.eids = (list(eids_in), list(eids_out))
        self.qnum = qnum


class OpGraphEdge:
    """
    Operator graph edge, representing a sum of local operators
    which are indexed by their IDs.
    """
    def __init__(self, eid: int, nids: Sequence[int], oids: Sequence[int]):
        if len(nids) != 2:
            raise ValueError(f"expecting exactly two node IDs per edge, received {len(nids)}")
        self.eid  = eid
        self.nids = list(nids)
        self.oids = sorted(list(oids))  # logical sum


class OpGraph:
    """
    Operator graph: internal data structure for generating MPO representations.
    """
    def __init__(self, nodes: Sequence[OpGraphNode], edges: Sequence[OpGraphEdge]):
        # dictionary of nodes
        self.nodes = {}
        for node in nodes:
            self.add_node(node)
        # dictionary of edges
        self.edges = {}
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node: OpGraphNode):
        """
        Add a node to the graph.
        """
        if node.nid in self.nodes:
            raise ValueError(f"node with ID {node.nid} already exists")
        self.nodes[node.nid] = node

    def add_edge(self, edge: OpGraphEdge):
        """
        Add an edge to the graph.
        """
        if edge.eid in self.edges:
            raise ValueError(f"edge with ID {edge.eid} already exists")
        self.edges[edge.eid] = edge

    def start_node_id(self, direction: int) -> int:
        """
        Get the node ID at the beginning or end of the graph.
        """
        if direction not in (0, 1):
            raise ValueError(f"'direction' must be 0 or 1, received {direction}")
        ids = []
        for node in self.nodes.values():
            if not node.eids[direction]:
                ids.append(node.nid)
        if len(ids) != 1:
            raise RuntimeError(f"expecting a single start node in direction {direction}, found {len(ids)} node(s)")
        return ids[0]

    def merge_edges(self, eid1: int, eid2: int, direction: int):
        """
        Merge paths along edges `eid1` and `eid2` in upstream (0) or downstream (1) direction.
        Edges must originate from same node.
        """
        if direction not in (0, 1):
            raise ValueError(f"'direction' must be 0 or 1, received {direction}")
        edge1 = self.edges[eid1]
        edge2 = self.edges.pop(eid2)
        assert edge1.nids[direction] == edge2.nids[direction], "to-be merged edges must originate from same node"
        # remove reference to edge2 from base node
        self.nodes[edge2.nids[direction]].eids[1-direction].remove(edge2.eid)
        if edge1.nids[1-direction] == edge2.nids[1-direction]:
            # edges have same upstream node -> add operators
            edge1.oids = sorted(edge1.oids + edge2.oids)
            # remove reference to edge2 from upstream node
            self.nodes[edge2.nids[1-direction]].eids[direction].remove(edge2.eid)
            return
        assert edge1.oids == edge2.oids, "can only merge edges with same logical operators"
        # merge upstream nodes
        node1 = self.nodes[edge1.nids[1-direction]]
        node2 = self.nodes.pop(edge2.nids[1-direction])
        assert len(node1.eids[direction]) == 1, "to-be merged upstream node can only have one input edge"
        assert len(node2.eids[direction]) == 1, "to-be merged upstream node can only have one input edge"
        assert node1.qnum == node2.qnum, f"can only merge nodes with same quantum numbers, encountered {node1.qnum} and {node2.qnum}"
        # make former edges from node2 to point to node1
        for eid in node2.eids[1-direction]:
            self.edges[eid].nids[direction] = node1.nid
        # explicit variable assignment to avoid "tuple item assignment" error
        node1_eids = node1.eids[1-direction]
        node1_eids += node2.eids[1-direction]

    def is_consistent(self, verbose=False) -> bool:
        """
        Perform an internal consistency check.
        """
        for k, node in self.nodes.items():
            if k != node.nid:
                if verbose:
                    print(f"Consistency check failed: dictionary key {k} does not match node ID {node.nid}.")
                return False
            for direction in (0, 1):
                for eid in node.eids[direction]:
                    # edge with ID 'eid' must exist
                    if eid not in self.edges:
                        if verbose:
                            print(f"Consistency check failed: edge with ID {eid} referenced by node {k} does not exist.")
                        return False
                    # edge must refer back to node
                    edge = self.edges[eid]
                    if edge.nids[1-direction] != node.nid:
                        if verbose:
                            print(f"Consistency check failed: edge with ID {eid} does not refer to node {node.nid}.")
                        return False
        for k, edge in self.edges.items():
            if k != edge.eid:
                if verbose:
                    print(f"Consistency check failed: dictionary key {k} does not match edge ID {edge.eid}.")
                return False
            for direction in (0, 1):
                if edge.nids[direction] not in self.nodes:
                    if verbose:
                        print(f"Consistency check failed: node with ID {edge.nids[direction]} referenced by edge {k} does not exist.")
                    return False
                node = self.nodes[edge.nids[direction]]
                if edge.eid not in node.eids[1-direction]:
                    if verbose:
                        print(f"Consistency check failed: node {node.nid} does not refer to edge {edge.eid}.")
                    return False
            if edge.oids != sorted(edge.oids):
                if verbose:
                    print(f"Consistency check failed: list of operator IDs of edge {edge.eid} is not sorted.")
                return False
        return True
