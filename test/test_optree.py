import unittest
import numpy as np
import pytenet as ptn


class TestOpTree(unittest.TestCase):

    def test_as_matrix(self):

        # physical quantum numbers
        qd = np.array([1, 0, -2, 0])

        # construct the tree
        # node depth 3
        node_i = ptn.OpTreeNode([], 0)
        node_j = ptn.OpTreeNode([], 0)
        node_k = ptn.OpTreeNode([], 0)
        # node depth 2
        node_d = ptn.OpTreeNode([ptn.OpTreeEdge(-7, 1.3, node_i), ptn.OpTreeEdge(-2, -0.4, node_j)], -1)
        node_e = ptn.OpTreeNode([], 0)
        node_f = ptn.OpTreeNode([ptn.OpTreeEdge(-5, 0.6, node_k)], -1)
        node_g = ptn.OpTreeNode([], 0)
        node_h = ptn.OpTreeNode([], 0)
        # node depth 1
        node_b = ptn.OpTreeNode([ptn.OpTreeEdge(-8, 0.1, node_d), ptn.OpTreeEdge(-4, -0.7, node_e)], 0)
        node_c = ptn.OpTreeNode([ptn.OpTreeEdge(-1, 0.5, node_f), ptn.OpTreeEdge(-6,  1.7, node_g), ptn.OpTreeEdge(-10, 0.2, node_h)], 1)
        # node depth 0
        node_a = ptn.OpTreeNode([ptn.OpTreeEdge(-9, 0.9, node_b), ptn.OpTreeEdge(-3,  1.1, node_c)], 0)

        tree = ptn.OpTree(node_a, 0)
        self.assertEqual(tree.height(), 3)

        # random local operators
        rng = np.random.default_rng()
        opmap = { opid: ptn.crandn(2 * (len(qd),), rng) for opid in range(-10, 0) }
        enforce_tree_operator_sparsity(tree.root, qd, opmap)

        # reference matrix representation
        mat_ref = (
              np.kron(0.9*opmap[-9], np.kron( 0.1*opmap[ -8], -0.4*opmap[-2] + 1.3*opmap[-7])
                                   + np.kron(-0.7*opmap[ -4],  np.identity(len(qd))))
            + np.kron(1.1*opmap[-3], np.kron( 0.5*opmap[ -1],  0.6*opmap[-5])
                                   + np.kron( 1.7*opmap[ -6],  np.identity(len(qd)))
                                   + np.kron( 0.2*opmap[-10],  np.identity(len(qd)))))

        # compare
        self.assertTrue(np.allclose(tree.as_matrix(opmap), mat_ref))


def enforce_tree_operator_sparsity(root: ptn.OpTreeNode, qd, opmap):
    """
    Enforce sparsity pattern of local operators in the tree according to quantum numbers.
    """
    for edge in root.children:
        mask = ptn.qnumber_outer_sum([qd, -qd, [root.qnum], [-edge.node.qnum]])[:, :, 0, 0]
        opmap[edge.oid] = np.where(mask == 0, opmap[edge.oid], 0)
        enforce_tree_operator_sparsity(edge.node, qd, opmap)


if __name__ == '__main__':
    unittest.main()
