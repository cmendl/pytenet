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
            [ptn.opgraph.OpGraphEdge(11, [ 2,  3], [(-4, -0.4), (-3,  0.1)]),
             ptn.opgraph.OpGraphEdge(14, [ 7,  3], [(-4, -0.4), (-3,  0.1)]),
             ptn.opgraph.OpGraphEdge(10, [ 4,  3], [( 0,  0.7)            ]),
             ptn.opgraph.OpGraphEdge(12, [ 1,  2], [(-2, -1.2)            ]),
             ptn.opgraph.OpGraphEdge(20, [ 5,  7], [(-2, -1.2)            ]),
             ptn.opgraph.OpGraphEdge(15, [ 9,  7], [(-5,  0.8)            ]),
             ptn.opgraph.OpGraphEdge(17, [ 6,  4], [(-7, -0.9)            ]),
             ptn.opgraph.OpGraphEdge(16, [ 8,  1], [(-6,  0.3)            ]),
             ptn.opgraph.OpGraphEdge(13, [ 8,  5], [(-8,  1.0)            ]),
             ptn.opgraph.OpGraphEdge(19, [ 8,  9], [(-1,  0.6)            ]),
             ptn.opgraph.OpGraphEdge(21, [ 8,  6], [(-1,  0.6)            ]),
             ],
            [8, 3])


    def test_node_depth(self):

        graph = self.generate_graph()

        self.assertEqual(graph.node_depth(8, 0), 0)
        self.assertEqual(graph.node_depth(8, 1), 3)
        self.assertEqual(graph.node_depth(5, 0), 1)
        self.assertEqual(graph.node_depth(5, 1), 2)
        self.assertEqual(graph.node_depth(4, 0), 2)
        self.assertEqual(graph.node_depth(4, 1), 1)
        self.assertEqual(graph.node_depth(3, 0), 3)
        self.assertEqual(graph.node_depth(3, 1), 0)


    def test_merge_edges(self):

        # physical quantum numbers
        qd = np.array([-1, 0, 2, 0])

        graph = self.generate_graph()
        self.assertTrue(graph.is_consistent())
        self.assertEqual(graph.length, 3)
        self.assertEqual(graph.nid_terminal[0], 8)
        self.assertEqual(graph.nid_terminal[1], 3)

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in range(-8, 1) }
        # enforce sparsity pattern according to quantum numbers
        for edge in graph.edges.values():
            qDloc = [graph.nodes[nid].qnum for nid in edge.nids]
            mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
            for opid, _ in edge.opics:
                opmap[opid] = np.where(mask == 0, opmap[opid], 0)

        # logical operation of initial graph as matrix
        mat0 = graph.as_matrix(opmap, 1)
        # must be independent of direction
        self.assertTrue(np.allclose(graph.as_matrix(opmap, 0), mat0))

        # convert initial graph to an MPO
        mpo0 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertEqual(mpo0.bond_dims, [1, 4, 3, 1])
        self.assertTrue(np.allclose(mat0, mpo0.as_matrix()))

        graph.merge_edges(11, 14, 1)
        self.assertTrue(graph.is_consistent())
        self.assertTrue(7 not in graph.nodes)
        node2 = graph.nodes[2]
        self.assertEqual(sorted(node2.eids[0]), [12, 15, 20])

        graph.merge_edges(12, 20, 1)
        self.assertTrue(graph.is_consistent())

        graph.merge_edges(16, 13, 1)
        self.assertTrue(graph.is_consistent())
        self.assertEqual(graph.edges[16].opics, [(-8, 1.0), (-6, 0.3)])

        graph.merge_edges(21, 19, 0)
        self.assertTrue(graph.is_consistent())

        # logical operation of final graph as matrix
        mat1 = graph.as_matrix(opmap, 1)
        # must be independent of direction
        self.assertTrue(np.allclose(graph.as_matrix(opmap, 1), mat1))

        # convert final graph to an MPO
        mpo1 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertEqual(mpo1.bond_dims, [1, 2, 2, 1])
        self.assertTrue(np.allclose(mat1, mpo1.as_matrix()))

        # compare matrix representations
        self.assertTrue(np.allclose(mat1, mat0))


    def test_insert_opchain(self):

        # physical quantum numbers
        qd = np.array([-1, 0, 2, 0])

        for direction in (0, 1):

            graph = self.generate_graph()
            self.assertTrue(graph.is_consistent())
            self.assertEqual(graph.length, 3)

            opids_chain = [1, 2]
            coeff_chain = [0.7, -0.4]
            qnums_chain = [1]
            nid_chain_terminals = [3, 6]
            nid_start_chain = nid_chain_terminals[direction]
            nid_end_chain   = nid_chain_terminals[1-direction]

            # random local operators
            rng = np.random.default_rng()
            opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in range(-8, 3) }
            # enforce sparsity pattern according to quantum numbers
            for edge in graph.edges.values():
                qDloc = [graph.nodes[nid].qnum for nid in edge.nids]
                mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
                for opid, _ in edge.opics:
                    opmap[opid] = np.where(mask == 0, opmap[opid], 0)
            # quantum numbers for to-be inserted quantum chain
            qnums_chain_ext = [graph.nodes[nid_start_chain].qnum] + qnums_chain + [graph.nodes[nid_end_chain].qnum]
            for i, opid in enumerate(opids_chain):
                qDloc = qnums_chain_ext[i:i+2]
                mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[1-direction]], [-qDloc[direction]]])[:, :, 0, 0]
                opmap[opid] = np.where(mask == 0, opmap[opid], 0)

            # matrix representation of operator chain
            mat_chain = np.identity(1)
            for opid, coeff in (zip(opids_chain, coeff_chain) if direction == 1 else zip(reversed(opids_chain), reversed(coeff_chain))):
                mat_chain = np.kron(mat_chain, coeff * opmap[opid])

            # logical operation of initial graph as matrix
            mat0 = graph.as_matrix(opmap)

            # insert the operator chain
            graph._insert_opchain(nid_start_chain, nid_end_chain, opids_chain, coeff_chain, qnums_chain, direction)
            self.assertTrue(graph.is_consistent())

            # logical operation of final graph as matrix
            mat1 = graph.as_matrix(opmap)

            # compare matrix representations, taking upstream connections of terminal nodes into account
            self.assertTrue(np.allclose(mat1,
                                        mat0 + np.kron(0.6 * opmap[-1], mat_chain)))


    def test_from_opchains(self):

        rng = np.random.default_rng()

        # physical quantum numbers
        qd = np.array([-1, 0, 2, 0])
        # overall system size of operator graph
        size = 5

        # identity operator ID
        oid_identity = 0

        chains = []
        for _ in range(6):
            # construct randomized operator chain
            istart = rng.integers(0, 3)
            length = rng.integers(1, size - istart + 1)
            oids  = rng.integers(1, 17, size=length)  # exclude identity ID to avoid incompatibility with sparsity pattern
            coeff = rng.standard_normal()
            qnums = [0] + list(rng.integers(-1, 2, size=length-1)) + [0]
            chain = ptn.OpChain(oids, qnums, coeff, istart)
            chains.append(chain)

        graph = ptn.OpGraph.from_opchains(chains, size, oid_identity)
        self.assertTrue(graph.is_consistent())
        self.assertEqual(graph.length, size)

        # random local operators
        opmap = { opid: np.identity(len(qd)) if opid == oid_identity else ptn.crandn(2 * (len(qd),), rng)
                 for opid in range(17) }
        # enforce sparsity pattern according to quantum numbers
        for chain in chains:
            for i, opid in enumerate(chain.oids):
                qDloc = chain.qnums[i:i+2]
                mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
                opmap[opid] = np.where(mask == 0, opmap[opid], 0)

        # reference matrix representation of operator chains
        mat_ref = 0
        for chain in chains:
            # including leading and trailing identity maps
            mat_ref = mat_ref + np.kron(np.kron(
                np.identity(len(qd)**chain.istart),
                chain.as_matrix(opmap)),
                np.identity(len(qd)**(size - (chain.istart + chain.length))))

        # compare matrix representations
        self.assertTrue(np.allclose(graph.as_matrix(opmap), mat_ref))
        mpo = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertTrue(np.allclose(mpo.as_matrix(), mat_ref))


    def generate_tree1(self):

        # node depth 3
        node_i = ptn.OpTreeNode([], 0)
        node_j = ptn.OpTreeNode([], 0)
        node_k = ptn.OpTreeNode([], 0)
        # node depth 2
        node_d = ptn.OpTreeNode([ptn.OpTreeEdge(-7, -0.9, node_i), ptn.OpTreeEdge(-2,  0.7, node_j)], -1)
        node_e = ptn.OpTreeNode([], 0)
        node_f = ptn.OpTreeNode([ptn.OpTreeEdge(-5,  0.4, node_k)], -1)
        node_g = ptn.OpTreeNode([], 0)
        node_h = ptn.OpTreeNode([], 0)
        # node depth 1
        node_b = ptn.OpTreeNode([ptn.OpTreeEdge(-8, -0.2, node_d), ptn.OpTreeEdge(-4, -0.6, node_e)], 0)
        node_c = ptn.OpTreeNode([ptn.OpTreeEdge(-1,  1.1, node_f), ptn.OpTreeEdge(-6,  0.4, node_g), ptn.OpTreeEdge(-10,  0.8, node_h)], 1)
        # node depth 0
        node_a = ptn.OpTreeNode([ptn.OpTreeEdge(-9,  1.7, node_b), ptn.OpTreeEdge(-3, -0.5, node_c)], 0)

        return ptn.OpTree(node_a, 0)

    def generate_tree2(self):

        # node depth 2
        node_e = ptn.OpTreeNode([], 0)
        node_f = ptn.OpTreeNode([], 0)
        node_g = ptn.OpTreeNode([], 0)
        # node depth 1
        node_b = ptn.OpTreeNode([ptn.OpTreeEdge( 2,  2.1, node_e)], 1)
        node_c = ptn.OpTreeNode([ptn.OpTreeEdge( 1, -0.7, node_f), ptn.OpTreeEdge( 4, -1.3, node_g)], -2)
        node_d = ptn.OpTreeNode([], 0)
        # node depth 0
        node_a = ptn.OpTreeNode([ptn.OpTreeEdge( 5,  0.4, node_b), ptn.OpTreeEdge(-1,  0.1, node_c), ptn.OpTreeEdge( 3,  1.5, node_d)], 0)

        return ptn.OpTree(node_a, 2)

    def test_from_optrees(self):

        # physical quantum numbers
        qd = np.array([1, 0, -2, 0])

        # construct two trees
        tree1 = self.generate_tree1()
        tree2 = self.generate_tree2()

        length = 4

        graph = ptn.OpGraph.from_optrees([tree1, tree2], length, 0)
        self.assertTrue(graph.is_consistent())
        self.assertEqual(graph.length, length)

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: np.identity(len(qd)) if opid == 0 else ptn.crandn(2 * (len(qd),), rng) for opid in range(-10, 6) }
        enforce_tree_operator_sparsity(tree1.root, qd, opmap)
        enforce_tree_operator_sparsity(tree2.root, qd, opmap)

        # reference matrix representation
        mat_ref = (np.kron(tree1.as_matrix(opmap), np.identity(len(qd)))
                 + np.kron(np.identity(len(qd)**tree2.istart), tree2.as_matrix(opmap)))

        # compare matrix representations
        self.assertTrue(np.allclose(graph.as_matrix(opmap), mat_ref))
        mpo = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertTrue(np.allclose(mpo.as_matrix(), mat_ref))


    def test_simplify(self):

        # physical quantum numbers
        qd = np.array([-1, 0, 2, 0])

        graph = self.generate_graph()
        self.assertTrue(graph.is_consistent())

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in range(-8, 1) }
        # enforce sparsity pattern according to quantum numbers
        for edge in graph.edges.values():
            qDloc = [graph.nodes[nid].qnum for nid in edge.nids]
            mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
            for opid, _ in edge.opics:
                opmap[opid] = np.where(mask == 0, opmap[opid], 0)

        # logical operation of initial graph as matrix
        mat0 = graph.as_matrix(opmap)

        # convert initial graph to an MPO
        mpo0 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertEqual(mpo0.bond_dims, [1, 4, 3, 1])
        self.assertTrue(np.allclose(mat0, mpo0.as_matrix()))

        graph.simplify()
        self.assertTrue(graph.is_consistent())

        # logical operation of final graph as matrix
        mat1 = graph.as_matrix(opmap)

        # convert final graph to an MPO
        mpo1 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertEqual(mpo1.bond_dims, [1, 2, 2, 1])
        self.assertTrue(np.allclose(mat1, mpo1.as_matrix()))

        # compare matrix representations
        self.assertTrue(np.allclose(mat1, mat0))


    def test_flip(self):

        # physical quantum numbers
        qd = np.array([-1, 0, 2, 0])

        graph = self.generate_graph()
        self.assertTrue(graph.is_consistent())

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in range(-8, 1) }
        # enforce sparsity pattern according to quantum numbers
        for edge in graph.edges.values():
            qDloc = [graph.nodes[nid].qnum for nid in edge.nids]
            mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
            for opid, _ in edge.opics:
                opmap[opid] = np.where(mask == 0, opmap[opid], 0)

        # logical operation of graph as matrix
        mat = graph.as_matrix(opmap)

        # flip the graph
        graph.flip()
        self.assertTrue(graph.is_consistent())

        mat_flip = graph.as_matrix(opmap)

        # compare matrix representations
        self.assertTrue(np.allclose(mat_flip, permute_operation(len(qd), mat, reversed(range(graph.length)))))


    def test_rename(self):

        # physical quantum numbers
        qd = np.array([-1, 0, 2, 0])

        graph = self.generate_graph()
        self.assertTrue(graph.is_consistent())

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in range(-8, 1) }
        # enforce sparsity pattern according to quantum numbers
        for edge in graph.edges.values():
            qDloc = [graph.nodes[nid].qnum for nid in edge.nids]
            mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
            for opid, _ in edge.opics:
                opmap[opid] = np.where(mask == 0, opmap[opid], 0)

        # logical operation of initial graph as matrix
        mat0 = graph.as_matrix(opmap)

        # convert initial graph to an MPO
        mpo0 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertTrue(np.allclose(mat0, mpo0.as_matrix()))

        # interleaved node and edge renaming
        graph.rename_node_id( 8, -8)
        graph.rename_edge_id(20,  3)
        graph.rename_edge_id(12,  7)
        graph.rename_node_id( 2, 17)
        graph.rename_edge_id(14, -1)
        self.assertTrue(graph.is_consistent())

        self.assertEqual(graph.nid_terminal[0], -8)
        self.assertEqual(graph.nid_terminal[1], 3)

        # logical operation of final graph as matrix
        mat1 = graph.as_matrix(opmap)

        # convert final graph to an MPO
        mpo1 = ptn.MPO.from_opgraph(qd, graph, opmap)
        self.assertTrue(np.allclose(mat1, mpo1.as_matrix()))

        # compare matrix representations
        self.assertTrue(np.allclose(mat1, mat0))


    def generate_another_graph(self):

        # generate a symbolic operator graph
        return ptn.opgraph.OpGraph(
            [ptn.opgraph.OpGraphNode( 5, [          ], [18, 30, 14],  0),
             ptn.opgraph.OpGraphNode( 8, [30,       ], [10        ],  0),
             ptn.opgraph.OpGraphNode( 1, [14        ], [11, 21    ],  1),
             ptn.opgraph.OpGraphNode( 3, [15, 10, 11], [13        ],  1),
             ptn.opgraph.OpGraphNode( 4, [18        ], [15, 27    ],  1),
             ptn.opgraph.OpGraphNode( 2, [27, 21    ], [16        ], -1),
             ptn.opgraph.OpGraphNode( 9, [13, 16    ], [          ],  0)],
            [ptn.opgraph.OpGraphEdge(18, [ 5,  4], [(-11, -0.7)           ]),
             ptn.opgraph.OpGraphEdge(30, [ 5,  8], [( -6,  0.6)           ]),
             ptn.opgraph.OpGraphEdge(14, [ 5,  1], [(  0,  1.3)           ]),
             ptn.opgraph.OpGraphEdge(15, [ 4,  3], [( -7, -0.3)           ]),
             ptn.opgraph.OpGraphEdge(10, [ 8,  3], [(  0,  0.9)           ]),
             ptn.opgraph.OpGraphEdge(27, [ 4,  2], [(-10, -0.2), (-2, 0.4)]),
             ptn.opgraph.OpGraphEdge(11, [ 1,  3], [( -9,  0.8)           ]),
             ptn.opgraph.OpGraphEdge(21, [ 1,  2], [(-10,  0.5)           ]),
             ptn.opgraph.OpGraphEdge(13, [ 3,  9], [( -5,  0.9)           ]),
             ptn.opgraph.OpGraphEdge(16, [ 2,  9], [(-13, -1.1)           ])],
            [5, 9])


    def test_add(self):

        # physical quantum numbers
        qd = np.array([-1, 0, 2, 0])

        graph = self.generate_graph()
        graph_b = self.generate_another_graph()
        self.assertTrue(graph.is_consistent())
        self.assertTrue(graph_b.is_consistent())

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in range(-13, 1) }
        # enforce sparsity pattern according to quantum numbers
        for graph in (graph, graph_b):
            for edge in graph.edges.values():
                qDloc = [graph.nodes[nid].qnum for nid in edge.nids]
                mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
                for opid, _ in edge.opics:
                    opmap[opid] = np.where(mask == 0, opmap[opid], 0)

        # logical operation of graph as matrix
        mat_a = graph.as_matrix(opmap)
        mat_b = graph_b.as_matrix(opmap)

        graph.add(graph_b)
        self.assertTrue(graph.is_consistent())
        self.assertTrue(graph_b.is_consistent())

        # compare matrix representations
        self.assertTrue(np.allclose(graph.as_matrix(opmap), mat_a + mat_b))


def enforce_tree_operator_sparsity(root: ptn.OpTreeNode, qd, opmap):
    """
    Enforce sparsity pattern of local operators in the tree according to quantum numbers.
    """
    for edge in root.children:
        mask = ptn.qnumber_outer_sum([qd, -qd, [root.qnum], [-edge.node.qnum]])[:, :, 0, 0]
        opmap[edge.oid] = np.where(mask == 0, opmap[edge.oid], 0)
        enforce_tree_operator_sparsity(edge.node, qd, opmap)


def permute_operation(d: int, U, perm):
    """
    Find the representation of a matrix after permuting lattice sites.
    """
    perm = list(perm)
    nsites = len(perm)
    U = np.asarray(U)
    assert U.shape == (d**nsites, d**nsites)
    U = np.reshape(U, (2*nsites) * (d,))
    U = np.transpose(U, perm + [nsites + p for p in perm])
    U = np.reshape(U, (d**nsites, d**nsites))
    return U


if __name__ == '__main__':
    unittest.main()
