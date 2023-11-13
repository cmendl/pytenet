"""
Implementation of the Hopcroft-Karp algorithm, based on
https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm
"""

from queue import Queue
from collections.abc import Sequence

__all__ = ['BipartiteGraph', 'HopcroftKarp']


class BipartiteGraph:
    """
    Data structure representing a bipartite graph G = ((U, V), E),
    where 'U' and 'V' are the vertices in the left and right partition, respectively,
    and 'E' the edges.

    Vertices in 'U' and 'V' are assumed to be sequentially indexed: 0, 1, ...
    """
    def __init__(self, num_u: int, num_v: int, edges: Sequence[tuple[int, int]]):
        assert num_u >= 1
        assert num_v >= 1
        self.num_u = num_u
        self.num_v = num_v
        # construct adjacency maps
        self.adj_u = [[] for u in range(num_u)]
        self.adj_v = [[] for v in range(num_v)]
        for (u, v) in edges:
            assert 0 <= u < num_u
            assert 0 <= v < num_v
            if v not in self.adj_u[u]:
                self.adj_u[u].append(v)
            if u not in self.adj_v[v]:
                self.adj_v[v].append(u)


class HopcroftKarp:
    """
    Implementation of the Hopcroft-Karp algorithm to find a maximum-cardinality matching,
    storing the temporary data for running the algorithm.
    """
    def __init__(self, graph: BipartiteGraph):
        # store a reference to the graph
        self.graph = graph
        # NIL vertex is indexed by -1
        self.matched_pairs_u = self.graph.num_u * [-1]
        self.matched_pairs_v = self.graph.num_v * [-1]
        self.dist = {}

    def __connect_unmatched_vertices(self) -> bool:
        """
        Find a path of minimal length connecting
        currently unmatched vertices in 'U' to currently unmatched vertices in 'V'
        via a breadth-first search.
        """
        queue = Queue()
        inf_dist = self.graph.num_u + 1  # formally "infinite" distance
        for u in range(self.graph.num_u):
            if self.matched_pairs_u[u] == -1:
                # 'u' has not been matched yet
                self.dist[u] = 0
                queue.put(u)
            else:
                self.dist[u] = inf_dist
        self.dist[-1] = inf_dist
        while not queue.empty():
            u = queue.get()
            if self.dist[u] < self.dist[-1]:
                for v in self.graph.adj_u[u]:
                    if self.dist[self.matched_pairs_v[v]] == inf_dist:
                        self.dist[self.matched_pairs_v[v]] = self.dist[u] + 1
                        queue.put(self.matched_pairs_v[v])
        return self.dist[-1] != inf_dist

    def __add_augmenting_path(self, u: int) -> bool:
        """
        Add an augmenting path to the matching by performing a depth-first search.
        """
        inf_dist = self.graph.num_u + 1  # formally "infinite" distance
        if u != -1:
            for v in self.graph.adj_u[u]:
                if self.dist[self.matched_pairs_v[v]] == self.dist[u] + 1:
                    if self.__add_augmenting_path(self.matched_pairs_v[v]):
                        self.matched_pairs_v[v] = u
                        self.matched_pairs_u[u] = v
                        return True
            # do not visit the same vertex multiple times
            self.dist[u] = inf_dist
            return False
        return True

    def __call__(self):
        """
        Run the Hopcroft-Karp algorithm to find a maximum-cardinality matching.
        """
        # reset internal data
        # NIL vertex is indexed by -1
        self.matched_pairs_u = self.graph.num_u * [-1]
        self.matched_pairs_v = self.graph.num_v * [-1]
        self.dist = {}
        # outer loop of the algorithm
        while self.__connect_unmatched_vertices():
            for u in range(self.graph.num_u):
                if self.matched_pairs_u[u] == -1:
                    self.__add_augmenting_path(u)
        # collect matched edges
        matching = []
        for u in range(self.graph.num_u):
            if self.matched_pairs_u[u] != -1:
                matching.append((u, self.matched_pairs_u[u]))
        return matching
