from typing import Sequence, Dict
import numpy as np

__all__ = ['OpTreeEdge', 'OpTreeNode', 'OpTree']


class OpTreeEdge:
    """
    Operator tree edge, representing a local operation.
    """
    def __init__(self, oid: int, node):
        if not isinstance(node, OpTreeNode):
            raise ValueError("'node' argument must be of type 'OpTreeNode'")
        self.oid = oid      # operator ID
        self.node = node    # child node


class OpTreeNode:
    """
    Operator tree node, representing a logical summation of its children,
    or denoting a tree leaf.
    """
    def __init__(self, children: Sequence[OpTreeEdge], qnum: int):
        if not all(isinstance(child, OpTreeEdge) for child in children):
            raise ValueError("children of a 'OpTreeNode' must be of type 'OpTreeEdge'")
        self.children = list(children)
        self.qnum = qnum    # quantum number

    def add_child(self, child):
        """
        Add a child to the operator tree node.
        """
        if not isinstance(child, OpTreeEdge):
            raise ValueError("child nodes of a 'OpTreeNode' must be of type 'OpTreeEdge'")
        self.children.append(child)

    def is_leaf(self) -> bool:
        """
        Whether the node is a leaf.
        """
        return not self.children


class OpTree:
    """
    Operator tree.
    """
    def __init__(self, root: OpTreeNode, istart: int):
        self.root = root
        self.istart = istart

    def height(self) -> int:
        """
        Height of the tree.
        """
        return _subtree_height(self.root)

    def as_matrix(self, opmap: Dict) -> np.ndarray:
        """
        Represent the logical operation of the tree as a matrix.
        """
        return _subtree_as_matrix(self.root, opmap)


def _subtree_height(node: OpTreeNode) -> int:
    """
    Compute the height of a subtree.
    """
    if node.is_leaf():
        return 0
    return 1 + max(_subtree_height(child.node) for child in node.children)


def _subtree_as_matrix(node: OpTreeNode, opmap: Dict) -> np.ndarray:
    """
    Contract the (sub-)tree to obtain its matrix representation.
    """
    op_sum = np.zeros((1, 1))
    for edge in node.children:
        if edge.node.is_leaf():
            op_subtree = np.identity(1)
        else:
            op_subtree = _subtree_as_matrix(edge.node, opmap)
        op = np.kron(opmap[edge.oid], op_subtree)
        # subtrees can have different heights
        if op_sum.shape[0] < op.shape[0]:
            assert op.shape[0] % op_sum.shape[0] == 0
            m = op.shape[0] // op_sum.shape[0]
            op_sum = np.kron(op_sum, np.identity(m))
        elif op.shape[0] < op_sum.shape[0]:
            assert op_sum.shape[0] % op.shape[0] == 0
            m = op_sum.shape[0] // op.shape[0]
            op = np.kron(op, np.identity(m))
        # not using += here to allow for "up-casting" to complex entries
        op_sum = op_sum + op
    return op_sum
