"""
Represent a product of sums of fermionic creation and annihilation operators
as a matrix product operator (MPO).
"""

from enum import IntEnum
import autoray as ar
from autoray import numpy as np
from ..mpo import MPO
from ..opgraph import OpGraphNode, OpGraphEdge, OpGraph
from ..qnumber import encode_quantum_number_pair

__all__ = ["quadratic_fermionic_mpo", "quadratic_spin_fermionic_mpo"]


def quadratic_fermionic_mpo(coeffc, coeffa, add_identity: bool = False) -> MPO:
    r"""
    Represent a product of sums of fermionic creation and
    annihilation operators of the following form as an MPO:

    .. math::

        op = (\sum_{i=1}^nsites coeffc_i a^{\dagger}_i) (\sum_{j=1}^nsites coeffa_j a_j)

    `add_identity` indicates whether to add a logical identity operation.
    """
    assert len(coeffc) == len(coeffa)
    nsites = len(coeffc)

    # creation and annihilation operators
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    z = np.array([[1., 0.], [0., -1.]])
    # operator map
    class OID(IntEnum):
        """
        Local operator identifiers.
        """
        A = -1
        I =  0
        C =  1
        N =  2
        Z =  3
    opmap = {
        OID.A: a_ann,
        OID.I: np.identity(2),
        OID.C: a_dag,
        OID.N: numop,
        OID.Z: z
    }

    # construct operator graph
    nid_next = 0
    # identity chains from the left and right
    identity_l = {}
    identity_r = {}
    for i in range(nsites):
        identity_l[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    for i in range(1, nsites + 1):
        identity_r[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    # nodes connecting creation and annihilation operators
    ca_nodes = {}
    ac_nodes = {}
    for i in range(1, nsites):
        ca_nodes[i] = OpGraphNode(nid_next, [], [], 1)
        nid_next += 1
    for i in range(1, nsites):
        ac_nodes[i] = OpGraphNode(nid_next, [], [], -1)
        nid_next += 1
    # initialize graph with nodes
    graph = OpGraph(list(identity_l.values()) +
                    list(identity_r.values()) +
                    list(ca_nodes.values()) +
                    list(ac_nodes.values()),
                    [], [identity_l[0].nid, identity_r[nsites].nid])
    # edges
    eid_next = 0
    # identities
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, identity_l[i + 1].nid], [(OID.I, 1.)]))
        eid_next += 1
    if add_identity:
        # connect identity string from left to right terminal
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[nsites - 1].nid,
                                   identity_r[nsites    ].nid], [(OID.I, 1.)]))
        eid_next += 1
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_r[i].nid, identity_r[i + 1].nid], [(OID.I, 1.)]))
        eid_next += 1
    # Z strings
    for i in range(1, nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ca_nodes[i].nid, ca_nodes[i + 1].nid], [(OID.Z, 1.)]))
        eid_next += 1
    for i in range(1, nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ac_nodes[i].nid, ac_nodes[i + 1].nid], [(OID.Z, 1.)]))
        eid_next += 1
    # number operators
    for i in range(nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, identity_r[i + 1].nid],
                        [(OID.N, coeffc[i]*coeffa[i])]))
        eid_next += 1
    # creation and annihilation operators
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, ca_nodes[i + 1].nid], [(OID.C, coeffc[i])]))
        eid_next += 1
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ca_nodes[i].nid, identity_r[i + 1].nid], [(OID.A, coeffa[i])]))
        eid_next += 1
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, ac_nodes[i + 1].nid], [(OID.A, coeffa[i])]))
        eid_next += 1
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ac_nodes[i].nid, identity_r[i + 1].nid], [(OID.C, coeffc[i])]))
        eid_next += 1
    assert graph.is_consistent()

    # convert to MPO
    dtype = complex if "complex" in ar.get_dtype_name(np.asarray(coeffc)) \
                    or "complex" in ar.get_dtype_name(np.asarray(coeffa)) else float
    return MPO.from_opgraph([0, 1], graph, opmap, dtype)


def quadratic_spin_fermionic_mpo(coeffc, coeffa, sigma: int, add_identity: bool = False) -> MPO:
    r"""
    Represent a product of sums of fermionic creation and
    annihilation operators of the following form as an MPO,
    where sigma = 1 indicates spin-up and sigma = -1 indicates spin-down:

    .. math::

        op = (\sum_{i=1}^nsites coeffc_i a^{\dagger}_{i,\sigma})
             (\sum_{j=1}^nsites coeffa_j a_{j,\sigma})

    `add_identity` indicates whether to add a logical identity operation.
    """
    assert len(coeffc) == len(coeffa)
    nsites = len(coeffc)
    assert nsites >= 1

    if sigma not in (1, -1):
        raise ValueError("`sigma` argument must be 1 (spin-up) or -1 (spin-down)")

    # physical particle number and spin quantum numbers (encoded as single integer)
    qn = [0,  1,  1,  2]
    qs = [0, -1,  1,  0]
    qsite = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qn, qs)]

    id2 = np.identity(2)
    # creation and annihilation operators
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    z = np.array([[1., 0.], [0., -1.]])
    # operator map
    class OID(IntEnum):
        """
        Local operator identifiers.
        """
        I  =  0
        IC =  1
        IA =  2
        IN =  3
        ZC =  4
        ZA =  5
        CI =  6
        AI =  7
        NI =  8
        CZ =  9
        AZ = 10
        ZZ = 11
    opmap = {
        OID.I:  np.identity(4),
        OID.IC: np.kron(id2, a_dag),
        OID.IA: np.kron(id2, a_ann),
        OID.IN: np.kron(id2, numop),
        OID.ZC: np.kron(z,   a_dag),
        OID.ZA: np.kron(z,   a_ann),
        OID.CI: np.kron(a_dag, id2),
        OID.AI: np.kron(a_ann, id2),
        OID.NI: np.kron(numop, id2),
        OID.CZ: np.kron(a_dag, z  ),
        OID.AZ: np.kron(a_ann, z  ),
        OID.ZZ: np.kron(z,     z  ),
    }

    # construct operator graph
    nid_next = 0
    # identity chains from the left and right
    identity_l = {}
    identity_r = {}
    for i in range(nsites):
        identity_l[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    for i in range(1, nsites + 1):
        identity_r[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    # nodes connecting creation and annihilation operators
    ca_nodes = {}
    ac_nodes = {}
    for i in range(1, nsites):
        qnum = encode_quantum_number_pair(1, sigma)
        ca_nodes[i] = OpGraphNode(nid_next, [], [], qnum)
        nid_next += 1
    for i in range(1, nsites):
        qnum = encode_quantum_number_pair(-1, -sigma)
        ac_nodes[i] = OpGraphNode(nid_next, [], [], qnum)
        nid_next += 1
    # initialize graph with nodes
    graph = OpGraph(list(identity_l.values()) +
                    list(identity_r.values()) +
                    list(ca_nodes.values()) +
                    list(ac_nodes.values()),
                    [], [identity_l[0].nid, identity_r[nsites].nid])
    # edges
    eid_next = 0
    # identities
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, identity_l[i + 1].nid], [(OID.I, 1.)]))
        eid_next += 1
    if add_identity:
        # connect identity string from left to right terminal
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[nsites - 1].nid,
                                   identity_r[nsites    ].nid], [(OID.I, 1.)]))
        eid_next += 1
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_r[i].nid, identity_r[i + 1].nid], [(OID.I, 1.)]))
        eid_next += 1
    # Z strings
    for i in range(1, nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ca_nodes[i].nid, ca_nodes[i + 1].nid], [(OID.ZZ, 1.)]))
        eid_next += 1
    for i in range(1, nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ac_nodes[i].nid, ac_nodes[i + 1].nid], [(OID.ZZ, 1.)]))
        eid_next += 1
    # number operators
    for i in range(nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, identity_r[i + 1].nid],
                        [(OID.NI if sigma == 1 else OID.IN, coeffc[i]*coeffa[i])]))
        eid_next += 1
    # creation and annihilation operators
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, ca_nodes[i + 1].nid],
                        [(OID.CZ if sigma == 1 else OID.IC, coeffc[i])]))
        eid_next += 1
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ca_nodes[i].nid, identity_r[i + 1].nid],
                        [(OID.AI if sigma == 1 else OID.ZA, coeffa[i])]))
        eid_next += 1
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, ac_nodes[i + 1].nid],
                        [(OID.AZ if sigma == 1 else OID.IA, coeffa[i])]))
        eid_next += 1
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ac_nodes[i].nid, identity_r[i + 1].nid],
                        [(OID.CI if sigma == 1 else OID.ZC, coeffc[i])]))
        eid_next += 1
    assert graph.is_consistent()

    # convert to MPO
    dtype = complex if "complex" in ar.get_dtype_name(np.asarray(coeffc)) \
                    or "complex" in ar.get_dtype_name(np.asarray(coeffa)) else float
    return MPO.from_opgraph(qsite, graph, opmap, dtype)
