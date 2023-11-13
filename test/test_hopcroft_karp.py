import unittest
import numpy as np
import pytenet as ptn


class TestHopcroftKarp(unittest.TestCase):

    def test(self):

        rng = np.random.default_rng()

        # generate a random bipartite graph
        num_u = rng.integers(1, 15)
        num_v = rng.integers(1, 15)
        edges = []
        for u in range(num_u):
            for v in range(num_v):
                if rng.uniform() < 0.2:
                    edges.append((u, v))
        graph = ptn.BipartiteGraph(num_u, num_v, edges)

        # run Hopcroft-Karp algorithm
        hopcroft_karp = ptn.HopcroftKarp(graph)
        matching = hopcroft_karp()

        # check validity of matching
        for (u, v) in matching:
            self.assertTrue((u, v) in edges)
        # every vertex can only be part of one edge from the matching
        ulist = []
        vlist = []
        for (u, v) in matching:
            ulist.append(u)
            vlist.append(v)
        self.assertTrue(len(set(ulist)) == len(ulist))
        self.assertTrue(len(set(vlist)) == len(vlist))

        # stochastic search must not return a higher-cardinality matching
        max_sms_len = 0
        for _ in range(100):
            sms = stochastic_matching_search(graph, rng)
            max_sms_len = max(len(sms), max_sms_len)
        self.assertLessEqual(max_sms_len, len(matching))


def stochastic_matching_search(graph: ptn.BipartiteGraph, rng: np.random.Generator):
    """
    Perform a stochastic matching search.
    """
    # collect all edges
    edges = []
    for u in range(graph.num_u):
        for v in graph.adj_u[u]:
            edges.append((u, v))
    matching = []
    while edges:
        # randomly pick one of the remaining edges
        i = rng.integers(len(edges))
        edge = edges.pop(i)
        # filter out edges with a vertex overlapping with the current edge
        edges = [e for e in edges if (e[0] != edge[0] and e[1] != edge[1])]
        matching.append(edge)
    return matching


if __name__ == '__main__':
    unittest.main()
