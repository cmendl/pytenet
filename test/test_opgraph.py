import unittest
import numpy as np
import pytenet as ptn


class TestOpGraph(unittest.TestCase):

    def generate_graph(self):

        # generate a symbolic operator graph
        return ptn.opgraph.OpGraph(
            [ptn.opgraph.OpGraphNode( 3, [11, 14, 10], [              ], 2),
             ptn.opgraph.OpGraphNode( 2, [12        ], [11            ], 0),
             ptn.opgraph.OpGraphNode( 7, [20, 15    ], [14            ], 0),
             ptn.opgraph.OpGraphNode( 4, [17        ], [10            ], 1),
             ptn.opgraph.OpGraphNode( 1, [16        ], [12            ], 2),
             ptn.opgraph.OpGraphNode( 5, [13        ], [20            ], 2),
             ptn.opgraph.OpGraphNode( 9, [19        ], [15            ], 1),
             ptn.opgraph.OpGraphNode( 6, [21        ], [17            ], 1),
             ptn.opgraph.OpGraphNode( 8, [          ], [21, 13, 16, 19], 2)],
            [ptn.opgraph.OpGraphEdge(11, [ 2,  3], [-4, -3]),
             ptn.opgraph.OpGraphEdge(14, [ 7,  3], [-4, -3]),
             ptn.opgraph.OpGraphEdge(10, [ 4,  3], [ 0    ]),
             ptn.opgraph.OpGraphEdge(12, [ 1,  2], [-2    ]),
             ptn.opgraph.OpGraphEdge(20, [ 5,  7], [-2    ]),
             ptn.opgraph.OpGraphEdge(15, [ 9,  7], [-5    ]),
             ptn.opgraph.OpGraphEdge(17, [ 6,  4], [-7    ]),
             ptn.opgraph.OpGraphEdge(16, [ 8,  1], [-6    ]),
             ptn.opgraph.OpGraphEdge(13, [ 8,  5], [-8    ]),
             ptn.opgraph.OpGraphEdge(19, [ 8,  9], [-1    ]),
             ptn.opgraph.OpGraphEdge(21, [ 8,  6], [-1    ])])


    def test_merge_edges(self):

        # physical quantum numbers
        qd = np.array([-1, 0, 2, 0])

        graph = self.generate_graph()

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in list(range(-8, 1)) }
        # enforce sparsity pattern according to quantum numbers
        for edge in graph.edges.values():
            qDloc = [graph.nodes[nid].qnum for nid in edge.nids]
            mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
            for opid in edge.oids:
                opmap[opid] = np.where(mask == 0, opmap[opid], 0)

        # convert initial graph to an MPO
        mpo0 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertEqual(mpo0.bond_dims, [1, 4, 3, 1])

        self.assertTrue(graph.is_consistent())
        self.assertEqual(graph.start_node_id(0), 8)
        self.assertEqual(graph.start_node_id(1), 3)

        graph.merge_edges(11, 14, 1)
        self.assertTrue(graph.is_consistent())
        self.assertTrue(7 not in graph.nodes)
        node2 = graph.nodes[2]
        self.assertEqual(sorted(node2.eids[0]), [12, 15, 20])

        graph.merge_edges(12, 20, 1)
        self.assertTrue(graph.is_consistent())

        graph.merge_edges(16, 13, 1)
        self.assertTrue(graph.is_consistent())
        self.assertEqual(graph.edges[16].oids, [-8, -6])

        graph.merge_edges(21, 19, 0)
        self.assertTrue(graph.is_consistent())

        # nodes at beginning and end must not change
        self.assertEqual(graph.start_node_id(0), 8)
        self.assertEqual(graph.start_node_id(1), 3)

        # convert final graph to an MPO
        mpo1 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertEqual(mpo1.bond_dims, [1, 2, 2, 1])

        # compare matrix representations of MPOs
        self.assertTrue(np.allclose(mpo1.as_matrix(), mpo0.as_matrix()))


    def test_simplify(self):

        # physical quantum numbers
        qd = np.array([-1, 0, 2, 0])

        graph = self.generate_graph()
        self.assertTrue(graph.is_consistent())

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in list(range(-8, 1)) }
        # enforce sparsity pattern according to quantum numbers
        for edge in graph.edges.values():
            qDloc = [graph.nodes[nid].qnum for nid in edge.nids]
            mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
            for opid in edge.oids:
                opmap[opid] = np.where(mask == 0, opmap[opid], 0)

        # convert initial graph to an MPO
        mpo0 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertEqual(mpo0.bond_dims, [1, 4, 3, 1])

        graph.simplify()
        self.assertTrue(graph.is_consistent())

        # convert final graph to an MPO
        mpo1 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertEqual(mpo1.bond_dims, [1, 2, 2, 1])

        # compare matrix representations of MPOs
        self.assertTrue(np.allclose(mpo1.as_matrix(), mpo0.as_matrix()))


if __name__ == '__main__':
    unittest.main()
