"""
Construct the Ising Hamiltonian as a matrix product operator (MPO).
"""

from enum import IntEnum
import numpy as np
from .mpo import MPO
from .autop import AutOpNode, AutOpEdge, AutOp
from .opgraph import OpGraph

__all__ = ["ising_1d_mpo"]


def ising_1d_mpo(nsites: int, J: float, h: float, g: float) -> MPO:
    """
    Construct Ising Hamiltonian `sum J sz sz + h sz + g sx`
    on a one-dimensional lattice as an MPO.

    Args:
        nsites: number of lattice sites
        J:      interaction parameter
        h:      longitudinal-field strength
        g:      transverse-field strength

    Returns:
        MPO: Ising Hamiltonian
    """
    # set physical quantum numbers to zero
    qsite = [0, 0]
    # Pauli matrices
    sigma_x = np.array([[0., 1.], [1.,  0.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])
    # operator map
    class OID(IntEnum):
        """
        Local operator identifiers.
        """
        I = 0
        Z = 1
        X = 2
    opmap = {
        OID.I: np.identity(2),
        OID.Z: sigma_z,
        OID.X: sigma_x }
    # operator state automaton:
    #
    #             Z     Z
    #   ╭──>─╮ ╭──>──o──>──╮ ╭──>─╮
    #  I│     o             o     │I
    #   ╰─<──╯ ╰─────>─────╯ ╰─<──╯
    #            h Z + g X
    #
    node_term0 = AutOpNode(0, [], [], 0)
    node_term1 = AutOpNode(1, [], [], 0)
    node_z     = AutOpNode(2, [], [], 0)
    autop = AutOp([node_term0, node_term1, node_z], [], [node_term0.nid, node_term1.nid])
    # identities looping around terminal nodes
    autop.add_connect_edge(AutOpEdge(0, [node_term0.nid, node_term0.nid], [(OID.I, 1.)]))
    autop.add_connect_edge(AutOpEdge(1, [node_term1.nid, node_term1.nid], [(OID.I, 1.)]))
    # Z Z terms
    autop.add_connect_edge(AutOpEdge(2, [node_term0.nid,     node_z.nid], [(OID.Z, J )]))
    autop.add_connect_edge(AutOpEdge(3, [node_z.nid,     node_term1.nid], [(OID.Z, 1.)]))
    # h Z + g X terms
    autop.add_connect_edge(AutOpEdge(4, [node_term0.nid, node_term1.nid], [(OID.Z, h )]))
    autop.add_connect_edge(AutOpEdge(5, [node_term0.nid, node_term1.nid], [(OID.X, g )]))
    assert autop.is_consistent()
    # convert to a graph and then to an MPO
    graph = OpGraph.from_automaton(autop, nsites)
    return MPO.from_opgraph(qsite, graph, opmap)
