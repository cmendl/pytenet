"""
Represent a sum of fermionic creation or annihilation operators as a matrix product operator (MPO).
"""

from enum import IntEnum
import numpy as np
from ..mpo import MPO
from ..opgraph import OpGraphNode, OpGraphEdge, OpGraph
from ..qnumber import encode_quantum_number_pair

__all__ = ["linear_fermionic_mpo", "linear_spin_fermionic_mpo"]


def linear_fermionic_mpo(coeff, ftype: str) -> MPO:
    r"""
    Represent a sum of fermionic creation or annihilation operators as an MPO:

    .. math::

        op = \sum_{i=1}^nsites coeff_i a^{\dagger}_i \text{ or } op = \sum_{i=1}^nsites coeff_i a_i
    """
    nsites = len(coeff)

    use_creation_op = (ftype in ["c", "create", "creation"])

    # creation and annihilation operators
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
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
        Z =  2
    opmap = {
        OID.A: a_ann,
        OID.I: np.identity(2),
        OID.C: a_dag,
        OID.Z: z
    }

    # construct operator graph
    nid_next = 0
    # identity and Z strings from the left and right
    identity_r = {}
    z_string_l = {}
    for i in range(1, nsites + 1):
        identity_r[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    for i in range(nsites):
        z_string_l[i] = OpGraphNode(nid_next, [], [], -1 if use_creation_op else 1)
        nid_next += 1
    # initialize graph with nodes
    graph = OpGraph(list(identity_r.values()) +
                    list(z_string_l.values()),
                    [], [z_string_l[0].nid, identity_r[nsites].nid])
    # edges
    eid_next = 0
    # identities
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_r[i].nid, identity_r[i + 1].nid], [(OID.I, 1.)]))
        eid_next += 1
    # Z strings
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [z_string_l[i].nid, z_string_l[i + 1].nid], [(OID.Z, 1.)]))
        eid_next += 1
    # creation or annihilation operators
    for i in range(nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [z_string_l[i].nid, identity_r[i + 1].nid],
                        [(OID.C if use_creation_op else OID.A, coeff[i])]))
        eid_next += 1
    assert graph.is_consistent()

    # convert to MPO
    return MPO.from_opgraph([0, 1], graph, opmap)


def linear_spin_fermionic_mpo(coeff, ftype: str, sigma: int) -> MPO:
    r"""
    Represent a sum of fermionic creation or annihilation operators as an MPO,
    where sigma = 1 indicates spin-up and sigma = -1 indicates spin-down:

    .. math::

        op = \sum_{i=1}^nsites coeff_i a^{\dagger}_{i,\sigma} \text{ or }
        op = \sum_{i=1}^nsites coeff_i a_{i,\sigma}
    """
    nsites = len(coeff)
    assert nsites >= 1

    if sigma not in (1, -1):
        raise ValueError("`sigma` argument must be 1 (spin-up) or -1 (spin-down)")

    use_creation_op = (ftype in ["c", "create", "creation"])

    # physical particle number and spin quantum numbers (encoded as single integer)
    qn = [0,  1,  1,  2]
    qs = [0, -1,  1,  0]
    qsite = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qn, qs)]

    id2 = np.identity(2)
    # creation and annihilation operators
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    z = np.array([[1., 0.], [0., -1.]])
    # operator map
    class OID(IntEnum):
        """
        Local operator identifiers.
        """
        I  = 0
        CI = 1
        AI = 2
        ZC = 3
        ZA = 4
        ZZ = 5
    opmap = {
        OID.I:  np.identity(4),
        OID.CI: np.kron(a_dag, id2),
        OID.AI: np.kron(a_ann, id2),
        OID.ZC: np.kron(z,   a_dag),
        OID.ZA: np.kron(z,   a_ann),
        OID.ZZ: np.kron(z,   z    ),
    }

    # construct operator graph
    nid_next = 0
    # identity and Z strings from the left and right
    identity_r = {}
    z_string_l = {}
    for i in range(1, nsites + 1):
        identity_r[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    for i in range(nsites):
        qnum = encode_quantum_number_pair(-1 if use_creation_op else 1,
                                          -sigma if use_creation_op else sigma)
        z_string_l[i] = OpGraphNode(nid_next, [], [], qnum)
        nid_next += 1
    # initialize graph with nodes
    graph = OpGraph(list(identity_r.values()) +
                    list(z_string_l.values()),
                    [], [z_string_l[0].nid, identity_r[nsites].nid])
    # edges
    eid_next = 0
    # identities
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_r[i].nid, identity_r[i + 1].nid], [(OID.I, 1.)]))
        eid_next += 1
    # Z strings
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [z_string_l[i].nid, z_string_l[i + 1].nid], [(OID.ZZ, 1.)]))
        eid_next += 1
    # creation or annihilation operators
    for i in range(nsites):
        if use_creation_op:
            oid = (OID.CI if sigma == 1 else OID.ZC)
        else:
            oid = (OID.AI if sigma == 1 else OID.ZA)
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [z_string_l[i].nid, identity_r[i + 1].nid], [(oid, coeff[i])]))
        eid_next += 1
    assert graph.is_consistent()

    # convert to MPO
    return MPO.from_opgraph(qsite, graph, opmap)
