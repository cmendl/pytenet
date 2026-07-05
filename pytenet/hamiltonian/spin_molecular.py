"""
Construct a spin molecular Hamiltonian as a matrix product operator (MPO).
"""

import itertools
from collections.abc import Sequence
from enum import IntEnum
import numpy as np
from ..mpo import MPO
from ..opchain import OpChain
from ..opgraph import OpGraphNode, OpGraphEdge, OpGraph
from ..qnumber import encode_quantum_number_pair
from .molecular import MolecularOID, _molecular_hamiltonian_generate_operator_map

__all__ = ["spin_molecular_hamiltonian_mpo"]


class SpinMolecularOID(IntEnum):
    """
    Local operator IDs for a molecular Hamiltonian using a spin orbital basis.
    """
    I  =  0
    IC =  1
    IA =  2
    IN =  3
    CI =  4
    CC =  5
    CA =  6
    CN =  7
    CZ =  8
    AI =  9
    AC = 10
    AA = 11
    AN = 12
    AZ = 13
    NI = 14
    NC = 15
    NA = 16
    NN = 17
    NZ = 18
    ZC = 19
    ZA = 20
    ZN = 21
    ZZ = 22


class SpinOperatorConverter:
    """
    Local operator conversion when transitioning from a spatial to a spin orbital basis.
    """

    # map a single OID pair to a combined OID
    oid_single_pair_map = {
        (MolecularOID.I, MolecularOID.I): SpinMolecularOID.I,
        (MolecularOID.I, MolecularOID.C): SpinMolecularOID.IC,
        (MolecularOID.I, MolecularOID.A): SpinMolecularOID.IA,
        (MolecularOID.I, MolecularOID.N): SpinMolecularOID.IN,
        (MolecularOID.C, MolecularOID.I): SpinMolecularOID.CI,
        (MolecularOID.C, MolecularOID.C): SpinMolecularOID.CC,
        (MolecularOID.C, MolecularOID.A): SpinMolecularOID.CA,
        (MolecularOID.C, MolecularOID.N): SpinMolecularOID.CN,
        (MolecularOID.C, MolecularOID.Z): SpinMolecularOID.CZ,
        (MolecularOID.A, MolecularOID.I): SpinMolecularOID.AI,
        (MolecularOID.A, MolecularOID.C): SpinMolecularOID.AC,
        (MolecularOID.A, MolecularOID.A): SpinMolecularOID.AA,
        (MolecularOID.A, MolecularOID.N): SpinMolecularOID.AN,
        (MolecularOID.A, MolecularOID.Z): SpinMolecularOID.AZ,
        (MolecularOID.N, MolecularOID.I): SpinMolecularOID.NI,
        (MolecularOID.N, MolecularOID.C): SpinMolecularOID.NC,
        (MolecularOID.N, MolecularOID.A): SpinMolecularOID.NA,
        (MolecularOID.N, MolecularOID.N): SpinMolecularOID.NN,
        (MolecularOID.N, MolecularOID.Z): SpinMolecularOID.NZ,
        (MolecularOID.Z, MolecularOID.C): SpinMolecularOID.ZC,
        (MolecularOID.Z, MolecularOID.A): SpinMolecularOID.ZA,
        (MolecularOID.Z, MolecularOID.N): SpinMolecularOID.ZN,
        (MolecularOID.Z, MolecularOID.Z): SpinMolecularOID.ZZ,
    }

    @classmethod
    def to_spin_opchain(cls, opchain_single: OpChain) -> OpChain:
        """
        Convert an operator chain with interleaved spin-up and spin-down
        local operators to an operator chain using a spin orbital basis.
        """
        assert opchain_single.qnums[ 0] == 0
        assert opchain_single.qnums[-1] == 0
        if opchain_single.istart % 2 == 1:
            opchain_single.oids.insert(0, MolecularOID.I)
            opchain_single.qnums.insert(0, 0)
            opchain_single.istart -= 1
        if opchain_single.length % 2 == 1:
            opchain_single.oids.append(MolecularOID.I)
            opchain_single.qnums.append(0)
        assert opchain_single.length % 2 == 0
        oids = [cls.oid_single_pair_map[pair]
                for pair in zip(opchain_single.oids[0::2], opchain_single.oids[1::2])]
        qnums = [0]
        qspin = 0
        for i in range(opchain_single.length // 2):
            # determine spin quantum number from particle quantum numbers
            qspin -= (opchain_single.qnums[2*i] - 2*opchain_single.qnums[2*i + 1] + opchain_single.qnums[2*i + 2])
            qnums.append(encode_quantum_number_pair(opchain_single.qnums[2*(i + 1)], qspin))
        assert qnums[-1] == 0
        return OpChain(oids, qnums, opchain_single.coeff, opchain_single.istart // 2)

    @classmethod
    def to_spin_operator(cls, oplist: Sequence[tuple],
                         even_parity_left: bool, even_parity_right: bool) -> SpinMolecularOID:
        """
        Convert a list of local creation and annihilation operators
        of the form [(spin_a, oid_a), ...]
        to the corresponding operator using a spin orbital basis.
        """
        oplist = sorted(oplist)
        if len(oplist) == 1:
            (spin, oid) = oplist[0]
            if spin == 0:
                transfer = MolecularOID.I if even_parity_right else MolecularOID.Z
                return cls.oid_single_pair_map[oid, transfer]
            if spin == 1:
                transfer = MolecularOID.I if even_parity_left else MolecularOID.Z
                return cls.oid_single_pair_map[transfer, oid]
            raise ValueError("`spin` must either be 0 (spin-up) or 1 (spin-down)")
        if len(oplist) == 2:
            (spin_a, oid_a), (spin_b, oid_b) = oplist
            if spin_a == spin_b:
                assert ((oid_a, oid_b) == (MolecularOID.C, MolecularOID.A)
                     or (oid_a, oid_b) == (MolecularOID.A, MolecularOID.C))
                if spin_a == 0:
                    return SpinMolecularOID.NI if even_parity_right else SpinMolecularOID.NZ
                if spin_a == 1:
                    return SpinMolecularOID.IN if even_parity_left else SpinMolecularOID.ZN
                raise ValueError("`spin` must either be 0 (spin-up) or 1 (spin-down)")
            assert (spin_a, spin_b) == (0, 1)  # must be sorted
            return cls.oid_single_pair_map[oid_a, oid_b]
        if len(oplist) == 3:
            # note: oplist is sorted
            (spin_a, oid_a), (spin_b, oid_b), (spin_c, oid_c) = oplist
            if (spin_a, spin_b, spin_c) == (0, 0, 1):
                assert ((oid_a, oid_b) == (MolecularOID.C, MolecularOID.A)
                     or (oid_a, oid_b) == (MolecularOID.A, MolecularOID.C))
                return cls.oid_single_pair_map[MolecularOID.N, oid_c]
            if (spin_a, spin_b, spin_c) == (0, 1, 1):
                assert ((oid_b, oid_c) == (MolecularOID.C, MolecularOID.A)
                     or (oid_b, oid_c) == (MolecularOID.A, MolecularOID.C))
                return cls.oid_single_pair_map[oid_a, MolecularOID.N]
            raise ValueError("`spin` must either be 0 (spin-up) or 1 (spin-down)")
        if len(oplist) == 4:
            # note: oplist is sorted
            (spin_a, oid_a), (spin_b, oid_b), (spin_c, oid_c), (spin_d, oid_d) = oplist
            assert (spin_a, spin_b, spin_c, spin_d) == (0, 0, 1, 1)
            assert ((oid_a, oid_b) == (MolecularOID.C, MolecularOID.A)
                 or (oid_a, oid_b) == (MolecularOID.A, MolecularOID.C))
            assert ((oid_c, oid_d) == (MolecularOID.C, MolecularOID.A)
                 or (oid_c, oid_d) == (MolecularOID.A, MolecularOID.C))
            return SpinMolecularOID.NN
        raise ValueError(f"`oplist` of length {len(oplist)} is invalid")


def _spin_molecular_hamiltonian_generate_operator_map():
    """
    Generate the local operator map for a molecular Hamiltonian using a spin orbital basis.
    """
    opmap_single = _molecular_hamiltonian_generate_operator_map()
    return { oid : np.kron(opmap_single[pair[0]], opmap_single[pair[1]])
             for pair, oid in SpinOperatorConverter.oid_single_pair_map.items() }


class SpinMolecularOpGraphNodes:
    """
    Operator graph nodes used for molecular Hamiltonian construction, assuming a spin orbital basis.
    """
    def __init__(self, nsites):

        self.nsites = nsites

        nid_next = 0
        # identity chains from the left and right
        self.identity_l = {}
        self.identity_r = {}
        for i in range(nsites):
            self.identity_l[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1
        for i in range(1, nsites + 1):
            self.identity_r[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to left terminal
        self.a_dag_l = {}
        for i, sigma in itertools.product(range(nsites - 1), (0, 1)):
            self.a_dag_l[i, sigma] = {}
            for j in range(i + 1, nsites):
                self.a_dag_l[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair( 1, [1, -1][sigma]))
                nid_next += 1
        # a_{i,\sigma} operators connected to left terminal
        self.a_ann_l = {}
        for i, sigma in itertools.product(range(nsites - 1), (0, 1)):
            self.a_ann_l[i, sigma] = {}
            for j in range(i + 1, nsites):
                self.a_ann_l[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair(-1, [-1, 1][sigma]))
                nid_next += 1
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to left terminal
        self.a_dag_a_dag_l = {}
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            for j, tau in itertools.product(range(i, nsites//2), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                self.a_dag_a_dag_l[i, sigma, j, tau] = {}
                for k in range(j + 1, nsites//2 + 1):
                    self.a_dag_a_dag_l[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair( 2, [1, -1][sigma] + [1, -1][tau]))
                    nid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to left terminal
        self.a_ann_a_ann_l = {}
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            for j, tau in itertools.product(range(i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                self.a_ann_a_ann_l[i, sigma, j, tau] = {}
                for k in range(i + 1, nsites//2 + 1):
                    self.a_ann_a_ann_l[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair(-2, [-1, 1][sigma] + [-1, 1][tau]))
                    nid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to left terminal
        self.a_dag_a_ann_l = {}
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            for j, tau in itertools.product(range(nsites//2), (0, 1)):
                self.a_dag_a_ann_l[i, sigma, j, tau] = {}
                for k in range(max(i, j) + 1, nsites//2 + 1):
                    self.a_dag_a_ann_l[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair(0, [1, -1][sigma] + [-1, 1][tau]))
                    nid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to right terminal
        self.a_dag_r = {}
        for i, sigma in itertools.product(range(1, nsites), (0, 1)):
            self.a_dag_r[i, sigma] = {}
            for j in range(1, i + 1):
                self.a_dag_r[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair(-1, [-1, 1][sigma]))
                nid_next += 1
        # a_{i,\sigma} operators connected to right terminal
        self.a_ann_r = {}
        for i, sigma in itertools.product(range(1, nsites), (0, 1)):
            self.a_ann_r[i, sigma] = {}
            for j in range(1, i + 1):
                self.a_ann_r[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair( 1, [1, -1][sigma]))
                nid_next += 1
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to right terminal
        self.a_dag_a_dag_r = {}
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(i, nsites), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                self.a_dag_a_dag_r[i, sigma, j, tau] = {}
                for k in range(nsites//2 + 1, i + 1):
                    self.a_dag_a_dag_r[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair(-2, [-1, 1][sigma] + [-1, 1][tau]))
                    nid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to right terminal
        self.a_ann_a_ann_r = {}
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(nsites//2 + 1, i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                self.a_ann_a_ann_r[i, sigma, j, tau] = {}
                for k in range(nsites//2 + 1, j + 1):
                    self.a_ann_a_ann_r[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair( 2, [1, -1][sigma] + [1, -1][tau]))
                    nid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to right terminal
        self.a_dag_a_ann_r = {}
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
                self.a_dag_a_ann_r[i, sigma, j, tau] = {}
                for k in range(nsites//2 + 1, min(i, j) + 1):
                    self.a_dag_a_ann_r[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair( 0, [-1, 1][sigma] + [1, -1][tau]))
                    nid_next += 1

    def get(self, oplist: Sequence[tuple], connection):
        """
        Retrieve nodes corresponding to a single or pairs of creation and annihilation operators.
        """
        if len(oplist) == 1:
            (i, sigma, oid) = oplist[0]
            if oid == MolecularOID.C:
                return self.a_dag_l[i, sigma] if connection == "left" else self.a_dag_r[i, sigma]
            if oid == MolecularOID.A:
                return self.a_ann_l[i, sigma] if connection == "left" else self.a_ann_r[i, sigma]
            raise KeyError(f"nodes for OID {oid} and (i, sigma) == ({i}, {sigma}) do not exist")
        if len(oplist) == 2:
            (i, sigma, oid0), (j, tau, oid1) = oplist
            if (oid0, oid1) == (MolecularOID.C, MolecularOID.C):
                (i, sigma), (j, tau) = sorted(((i, sigma), (j, tau)))
                return self.a_dag_a_dag_l[i, sigma, j, tau] if connection == "left" else self.a_dag_a_dag_r[i, sigma, j, tau]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.A):
                (i, sigma), (j, tau) = sorted(((i, sigma), (j, tau)), reverse=True)
                return self.a_ann_a_ann_l[i, sigma, j, tau] if connection == "left" else self.a_ann_a_ann_r[i, sigma, j, tau]
            if (oid0, oid1) == (MolecularOID.C, MolecularOID.A):
                return self.a_dag_a_ann_l[i, sigma, j, tau] if connection == "left" else self.a_dag_a_ann_r[i, sigma, j, tau]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.C):
                return self.a_dag_a_ann_l[j, tau, i, sigma] if connection == "left" else self.a_dag_a_ann_r[j, tau, i, sigma]
            raise KeyError(f"nodes for OIDs ({oid0}, {oid1}) and (i, sigma, j, tau) == ({i}, {sigma}, {j}, {tau}) do not exist")
        raise KeyError(f"nodes for operator list of length {len(oplist)} do not exist")

    def generate_graph(self) -> OpGraph:
        """
        Create and initialize an operator graph with corresponding nodes and edges.
        """
        nsites = self.nsites

        # initialize graph with nodes
        graph = OpGraph(list(self.identity_l.values()) +
                        list(self.identity_r.values()) +
                        [node for nodes in self.a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_r.values() for node in nodes.values()],
                        [], [self.identity_l[0].nid, self.identity_r[nsites].nid])

        # edges
        eid_next = 0
        # identities connected to left and right terminals
        for i in range(nsites - 1):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.identity_l[i + 1].nid],
                            [(SpinMolecularOID.I, 1.)]))
            eid_next += 1
        for i in range(1, nsites):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_r[i].nid, self.identity_r[i + 1].nid],
                            [(SpinMolecularOID.I, 1.)]))
            eid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to left terminal
        for i, sigma in itertools.product(range(nsites - 1), (0, 1)):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_dag_l[i, sigma][i + 1].nid],
                            [([SpinMolecularOID.CZ, SpinMolecularOID.IC][sigma], 1.)]))
            eid_next += 1
            # Z operators from Jordan-Wigner transformation
            for j in range(i + 1, nsites - 1):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_l[i, sigma][j].nid, self.a_dag_l[i, sigma][j + 1].nid],
                                [(SpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
        # a_{i,\sigma} operators connected to left terminal
        for i, sigma in itertools.product(range(nsites - 1), (0, 1)):
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_ann_l[i, sigma][i + 1].nid],
                            [([SpinMolecularOID.AZ, SpinMolecularOID.IA][sigma], 1.)]))
            eid_next += 1
            # Z operators from Jordan-Wigner transformation
            for j in range(i + 1, nsites - 1):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_l[i, sigma][j].nid, self.a_ann_l[i, sigma][j + 1].nid],
                                [(SpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to left terminal
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            for j, tau in itertools.product(range(i, nsites//2), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_l[i, sigma][j].nid, self.a_dag_a_dag_l[i, sigma, j, tau][j + 1].nid],
                                    [([SpinMolecularOID.CI, SpinMolecularOID.ZC][tau], 1.)]))
                else:
                    assert sigma == 0 and tau == 1
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.identity_l[j].nid, self.a_dag_a_dag_l[i, sigma, j, tau][j + 1].nid],
                                    [(SpinMolecularOID.CC, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(j + 1, nsites//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_l[i, sigma, j, tau][k].nid, self.a_dag_a_dag_l[i, sigma, j, tau][k + 1].nid],
                                    [(SpinMolecularOID.I, 1.)]))
                    eid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to left terminal
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            for j, tau in itertools.product(range(i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                if i > j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_l[j, tau][i].nid, self.a_ann_a_ann_l[i, sigma, j, tau][i + 1].nid],
                                    [([SpinMolecularOID.AI, SpinMolecularOID.ZA][sigma], 1.)]))
                else:
                    assert sigma == 1 and tau == 0
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_ann_a_ann_l[i, sigma, j, tau][i + 1].nid],
                                    [(SpinMolecularOID.AA, 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(i + 1, nsites//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_l[i, sigma, j, tau][k].nid, self.a_ann_a_ann_l[i, sigma, j, tau][k + 1].nid],
                                    [(SpinMolecularOID.I, 1.)]))
                    eid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to left terminal
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            for j, tau in itertools.product(range(nsites//2), (0, 1)):
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_l[i, sigma][j].nid, self.a_dag_a_ann_l[i, sigma, j, tau][j + 1].nid],
                                    [([SpinMolecularOID.AI, SpinMolecularOID.ZA][tau], 1.)]))
                elif i == j:
                    if sigma < tau:
                        oid = SpinMolecularOID.CA
                    elif sigma == tau:
                        oid = [SpinMolecularOID.NI, SpinMolecularOID.IN][sigma]
                    else:  # sigma > tau
                        oid = SpinMolecularOID.AC
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.identity_l[i].nid, self.a_dag_a_ann_l[i, sigma, j, tau][i + 1].nid],
                                    [(oid, 1.)]))
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_l[j, tau][i].nid, self.a_dag_a_ann_l[i, sigma, j, tau][i + 1].nid],
                                    [([SpinMolecularOID.CI, SpinMolecularOID.ZC][sigma], 1.)]))
                eid_next += 1
                # identities for transition to next site
                for k in range(max(i, j) + 1, nsites//2):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_l[i, sigma, j, tau][k].nid, self.a_dag_a_ann_l[i, sigma, j, tau][k + 1].nid],
                                    [(SpinMolecularOID.I, 1.)]))
                    eid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to right terminal
        for i, sigma in itertools.product(range(1, nsites), (0, 1)):
            # Z operators from Jordan-Wigner transformation
            for j in range(1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_dag_r[i, sigma][j].nid, self.a_dag_r[i, sigma][j + 1].nid],
                                [(SpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_dag_r[i, sigma][i].nid, self.identity_r[i + 1].nid],
                            [([SpinMolecularOID.CI, SpinMolecularOID.ZC][sigma], 1.)]))
            eid_next += 1
        # a_{i,\sigma} operators connected to right terminal
        for i, sigma in itertools.product(range(1, nsites), (0, 1)):
            # Z operators from Jordan-Wigner transformation
            for j in range(1, i):
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [self.a_ann_r[i, sigma][j].nid, self.a_ann_r[i, sigma][j + 1].nid],
                                [(SpinMolecularOID.ZZ, 1.)]))
                eid_next += 1
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [self.a_ann_r[i, sigma][i].nid, self.identity_r[i + 1].nid],
                            [([SpinMolecularOID.AI, SpinMolecularOID.ZA][sigma], 1.)]))
            eid_next += 1
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to right terminal
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(i, nsites), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                # identities for transition to next site
                for k in range(nsites//2 + 1, i):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, sigma, j, tau][k].nid, self.a_dag_a_dag_r[i, sigma, j, tau][k + 1].nid],
                                    [(SpinMolecularOID.I, 1.)]))
                    eid_next += 1
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, sigma, j, tau][i].nid, self.a_dag_r[j, tau][i + 1].nid],
                                    [([SpinMolecularOID.CZ, SpinMolecularOID.IC][sigma], 1.)]))
                else:
                    assert sigma == 0 and tau == 1
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_dag_r[i, sigma, j, tau][i].nid, self.identity_r[i + 1].nid],
                                    [(SpinMolecularOID.CC, 1.)]))
                eid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to right terminal
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(nsites//2 + 1, i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                # identities for transition to next site
                for k in range(nsites//2 + 1, j):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, sigma, j, tau][k].nid, self.a_ann_a_ann_r[i, sigma, j, tau][k + 1].nid],
                                    [(SpinMolecularOID.I, 1.)]))
                    eid_next += 1
                if i > j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, sigma, j, tau][j].nid, self.a_ann_r[i, sigma][j + 1].nid],
                                    [([SpinMolecularOID.AZ, SpinMolecularOID.IA][tau], 1.)]))
                else:
                    assert sigma == 1 and tau == 0
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_ann_a_ann_r[i, sigma, j, tau][j].nid, self.identity_r[j + 1].nid],
                                    [(SpinMolecularOID.AA, 1.)]))
                eid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to right terminal
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
                # identities for transition to next site
                for k in range(nsites//2 + 1, min(i, j)):
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, sigma, j, tau][k].nid, self.a_dag_a_ann_r[i, sigma, j, tau][k + 1].nid],
                                    [(SpinMolecularOID.I, 1.)]))
                    eid_next += 1
                if i < j:
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, sigma, j, tau][i].nid, self.a_ann_r[j, tau][i + 1].nid],
                                    [([SpinMolecularOID.CZ, SpinMolecularOID.IC][sigma], 1.)]))
                elif i == j:
                    if sigma < tau:
                        oid = SpinMolecularOID.CA
                    elif sigma == tau:
                        oid = [SpinMolecularOID.NI, SpinMolecularOID.IN][sigma]
                    else:  # sigma > tau
                        oid = SpinMolecularOID.AC
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, sigma, j, tau][i].nid, self.identity_r[i + 1].nid],
                                    [(oid, 1.)]))
                else:  # i > j
                    graph.add_connect_edge(
                        OpGraphEdge(eid_next, [self.a_dag_a_ann_r[i, sigma, j, tau][j].nid, self.a_dag_r[i, sigma][j + 1].nid],
                                    [([SpinMolecularOID.AZ, SpinMolecularOID.IA][tau], 1.)]))
                eid_next += 1

        return graph

    def copy_nids(self, target):
        """
        Transfer and store the node IDs in the target object.
        """
        nsites = self.nsites
        # identity chains from the left and right
        target.nids_identity_l = {}
        target.nids_identity_r = {}
        for i in range(nsites):
            target.nids_identity_l[i] = self.identity_l[i].nid
        for i in range(1, nsites + 1):
            target.nids_identity_r[i] = self.identity_r[i].nid
        # a^{\dagger}_{i,\sigma} operators connected to left terminal
        target.nids_a_dag_l = {}
        for i, sigma in itertools.product(range(nsites - 1), (0, 1)):
            target.nids_a_dag_l[i, sigma] = {}
            for j in range(i + 1, nsites):
                target.nids_a_dag_l[i, sigma][j] = self.a_dag_l[i, sigma][j].nid
        # a_{i,\sigma} operators connected to left terminal
        target.nids_a_ann_l = {}
        for i, sigma in itertools.product(range(nsites - 1), (0, 1)):
            target.nids_a_ann_l[i, sigma] = {}
            for j in range(i + 1, nsites):
                target.nids_a_ann_l[i, sigma][j] = self.a_ann_l[i, sigma][j].nid
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to left terminal
        target.nids_a_dag_a_dag_l = {}
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            for j, tau in itertools.product(range(i, nsites//2), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                target.nids_a_dag_a_dag_l[i, sigma, j, tau] = {}
                for k in range(j + 1, nsites//2 + 1):
                    target.nids_a_dag_a_dag_l[i, sigma, j, tau][k] = self.a_dag_a_dag_l[i, sigma, j, tau][k].nid
        # a_{i,\sigma} a_{j,\tau} operators connected to left terminal
        target.nids_a_ann_a_ann_l = {}
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            for j, tau in itertools.product(range(i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                target.nids_a_ann_a_ann_l[i, sigma, j, tau] = {}
                for k in range(i + 1, nsites//2 + 1):
                    target.nids_a_ann_a_ann_l[i, sigma, j, tau][k] = self.a_ann_a_ann_l[i, sigma, j, tau][k].nid
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to left terminal
        target.nids_a_dag_a_ann_l = {}
        for i, sigma in itertools.product(range(nsites//2), (0, 1)):
            for j, tau in itertools.product(range(nsites//2), (0, 1)):
                target.nids_a_dag_a_ann_l[i, sigma, j, tau] = {}
                for k in range(max(i, j) + 1, nsites//2 + 1):
                    target.nids_a_dag_a_ann_l[i, sigma, j, tau][k] = self.a_dag_a_ann_l[i, sigma, j, tau][k].nid
        # a^{\dagger}_{i,\sigma} operators connected to right terminal
        target.nids_a_dag_r = {}
        for i, sigma in itertools.product(range(1, nsites), (0, 1)):
            target.nids_a_dag_r[i, sigma] = {}
            for j in range(1, i + 1):
                target.nids_a_dag_r[i, sigma][j] = self.a_dag_r[i, sigma][j].nid
        # a_{i,\sigma} operators connected to right terminal
        target.nids_a_ann_r = {}
        for i, sigma in itertools.product(range(1, nsites), (0, 1)):
            target.nids_a_ann_r[i, sigma] = {}
            for j in range(1, i + 1):
                target.nids_a_ann_r[i, sigma][j] = self.a_ann_r[i, sigma][j].nid
        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to right terminal
        target.nids_a_dag_a_dag_r = {}
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(i, nsites), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                target.nids_a_dag_a_dag_r[i, sigma, j, tau] = {}
                for k in range(nsites//2 + 1, i + 1):
                    target.nids_a_dag_a_dag_r[i, sigma, j, tau][k] = self.a_dag_a_dag_r[i, sigma, j, tau][k].nid
        # a_{i,\sigma} a_{j,\tau} operators connected to right terminal
        target.nids_a_ann_a_ann_r = {}
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(nsites//2 + 1, i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                target.nids_a_ann_a_ann_r[i, sigma, j, tau] = {}
                for k in range(nsites//2 + 1, j + 1):
                    target.nids_a_ann_a_ann_r[i, sigma, j, tau][k] = self.a_ann_a_ann_r[i, sigma, j, tau][k].nid
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to right terminal
        target.nids_a_dag_a_ann_r = {}
        for i, sigma in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(nsites//2 + 1, nsites), (0, 1)):
                target.nids_a_dag_a_ann_r[i, sigma, j, tau] = {}
                for k in range(nsites//2 + 1, min(i, j) + 1):
                    target.nids_a_dag_a_ann_r[i, sigma, j, tau][k] = self.a_dag_a_ann_r[i, sigma, j, tau][k].nid


def _spin_molecular_hamiltonian_graph_add_term(graph: OpGraph, nodes: SpinMolecularOpGraphNodes,
                                               oplist: Sequence[tuple], coeff: float):
    """
    Add an operator term (operator string of creation and annihilation operators)
    to the operator graph describing a molecular Hamiltonian, assuming a spin orbital basis.
    """
    eid_next = max(graph.edges.keys()) + 1

    nsites = nodes.nsites

    # sort by site (orbital) and spin index
    oplist = sorted(oplist)

    if len(oplist) == 2:
        (i, sigma, oid0), (j, tau, oid1) = oplist
        if i == j:
            spin_oid = SpinOperatorConverter.to_spin_operator(
                [(sigma, oid0), (tau, oid1)], True, True)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid],
                            [(spin_oid, coeff)]))
        else:
            assert i < j
            if j <= nsites//2:
                nodes_l = nodes.get([oplist[0]], "left")
                spin_oid = SpinOperatorConverter.to_spin_operator([(tau, oid1)], False, True)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes.identity_r[j + 1].nid],
                                [(spin_oid, coeff)]))
            elif i >= nsites//2:
                nodes_r = nodes.get([oplist[1]], "right")
                spin_oid = SpinOperatorConverter.to_spin_operator([(sigma, oid0)], True, False)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes_r[i + 1].nid],
                                [(spin_oid, coeff)]))
            else:
                nodes_l = nodes.get([oplist[0]], "left")
                nodes_r = nodes.get([oplist[1]], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[nsites//2].nid, nodes_r[nsites//2 + 1].nid],
                                [(SpinMolecularOID.ZZ, coeff)]))
    elif len(oplist) == 4:
        (i, sigma, oid0), (j, tau, oid1), (k, mu, oid2), (l, nu, oid3) = oplist
        if i == j == k == l:
            spin_oid = SpinOperatorConverter.to_spin_operator(
                [(sigma, oid0), (tau, oid1), (mu, oid2), (nu, oid3)], True, True)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid],
                            [(spin_oid, coeff)]))
        elif i == j == k:
            nodes_r = nodes.get([oplist[3]], "right")
            spin_oid = SpinOperatorConverter.to_spin_operator(
                [(sigma, oid0), (tau, oid1), (mu, oid2)], True, False)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes_r[i + 1].nid],
                            [(spin_oid, coeff)]))
        elif j == k == l:
            nodes_l = nodes.get([oplist[0]], "left")
            spin_oid = SpinOperatorConverter.to_spin_operator(
                [(tau, oid1), (mu, oid2), (nu, oid3)], False, True)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[j].nid, nodes.identity_r[j + 1].nid],
                            [(spin_oid, coeff)]))
        elif j == k:
            nodes_l = nodes.get([oplist[0]], "left")
            nodes_r = nodes.get([oplist[3]], "right")
            spin_oid = SpinOperatorConverter.to_spin_operator(
                [(tau, oid1), (mu, oid2)], False, False)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid],
                            [(spin_oid, coeff)]))
        elif k <= nsites//2:
            nodes_l = nodes.get(oplist[:2], "left")
            if k == l:
                spin_oid = SpinOperatorConverter.to_spin_operator(
                    [(mu, oid2), (nu, oid3)], True, True)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes.identity_r[k + 1].nid],
                                [(spin_oid, coeff)]))
            else:
                nodes_r = nodes.get([oplist[3]], "right")
                spin_oid = SpinOperatorConverter.to_spin_operator([(mu, oid2)], True, False)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes_r[k + 1].nid],
                                [(spin_oid, coeff)]))
        elif j >= nsites//2:
            nodes_r = nodes.get(oplist[2:], "right")
            if i == j:
                spin_oid = SpinOperatorConverter.to_spin_operator(
                    [(sigma, oid0), (tau, oid1)], True, True)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[j].nid, nodes_r[j + 1].nid],
                                [(spin_oid, coeff)]))
            else:
                nodes_l = nodes.get([oplist[0]], "left")
                spin_oid = SpinOperatorConverter.to_spin_operator([(tau, oid1)], False, True)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid],
                                [(spin_oid, coeff)]))
        else:
            nodes_l = nodes.get(oplist[:2], "left")
            nodes_r = nodes.get(oplist[2:], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[nsites//2].nid, nodes_r[nsites//2 + 1].nid],
                            [(SpinMolecularOID.I, coeff)]))
    else:
        raise NotImplementedError


def spin_molecular_hamiltonian_mpo(tkin, vint, optimize=True) -> MPO:
    r"""
    Construct a molecular Hamiltonian as an MPO, assuming a spin orbital basis and
    using physicists' convention for the interaction term (note ordering of k and \ell):

    .. math::

        H = \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
            + \frac{1}{2} \sum_{i,j,k,\ell,\sigma,\tau} v_{i,j,k,\ell}
              a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{\ell,\tau} a_{k,\sigma}
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    nsites = tkin.shape[0]
    assert tkin.shape == (nsites, nsites)
    assert vint.shape == (nsites, nsites, nsites, nsites)

    # physical particle number and spin quantum numbers (encoded as single integer)
    qn = [0,  1,  1,  2]
    qs = [0, -1,  1,  0]
    qsite = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qn, qs)]

    # interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau,\mu,\nu} v_{i,j,k,l}
    #   \delta_{\sigma,\mu} \delta_{\tau,\nu}
    #   a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\nu} a_{k,\mu}:
    # can anti-commute fermionic operators such that (i,\sigma) < (j,\tau) and (k,\mu) < (l,\nu)
    gint0 = 0.5 * (vint                             + np.transpose(vint, (1, 0, 3, 2)))
    gint1 = 0.5 * (np.transpose(vint, (1, 0, 2, 3)) + np.transpose(vint, (0, 1, 3, 2)))

    def get_vint_coeff(spatial_idx, spin_idx):
        valid = False
        coeff = 0
        if (spin_idx[0] == spin_idx[2]) and (spin_idx[1] == spin_idx[3]):
            coeff += gint0[spatial_idx]
            valid = True
        if (spin_idx[0] == spin_idx[3]) and (spin_idx[1] == spin_idx[2]):
            coeff -= gint1[spatial_idx]
            valid = True
        return coeff, valid

    if optimize:
        # optimize MPO bond dimensions based on bipartite graph theory

        opchains = []
        # kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
        for i in range(2*nsites):
            for j in range(2*nsites):
                if (i - j) % 2 != 0:
                    continue
                if i == j:
                    # diagonal hopping term
                    opchains.append(SpinOperatorConverter.to_spin_opchain(
                        OpChain([MolecularOID.N], [0, 0], tkin[i//2, i//2], i)))
                else:
                    (a, p), (b, q) = sorted([(i, MolecularOID.C), (j, MolecularOID.A)])
                    opchains.append(SpinOperatorConverter.to_spin_opchain(
                        OpChain([p] + (b - a - 1)*[MolecularOID.Z] + [q],
                                [0] + (b - a)*[int(p)] + [0], tkin[i//2, j//2], a)))
        # interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau} v_{i,j,k,l}
        #   a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\tau} a_{k,\sigma}
        for i in range(2*nsites):
            for j in range(i + 1, 2*nsites):  # i < j
                for k in range(2*nsites):
                    for l in range(k + 1, 2*nsites):  # k < l
                        coeff, valid = get_vint_coeff((i // 2, j // 2, k // 2, l // 2),
                                                      (i %  2, j %  2, k %  2, l %  2))
                        if not valid:
                            continue
                        (a, p), (b, q), (c, r), (d, s) = sorted([(i, MolecularOID.C),
                                                                 (j, MolecularOID.C),
                                                                 (l, MolecularOID.A),
                                                                 (k, MolecularOID.A)])
                        if a == b:
                            assert b < c
                            if c == d:
                                # two number operators
                                oids  = (  [MolecularOID.N]
                                         + (c - b - 1)*[MolecularOID.I]
                                         + [MolecularOID.N])
                                qnums = (c - b + 2)*[0]
                            else:
                                # number operator at the beginning
                                oids  = (  [MolecularOID.N]
                                         + (c - b - 1)*[MolecularOID.I]
                                         + [r]
                                         + (d - c - 1)*[MolecularOID.Z]
                                         + [s])
                                qnums = (c - b + 1)*[0] + (d - c)*[int(r)] + [0]
                        elif b == c:
                            # number operator in the middle
                            oids  = (  [p]
                                     + (b - a - 1)*[MolecularOID.Z]
                                     + [MolecularOID.N]
                                     + (d - c - 1)*[MolecularOID.Z]
                                     + [s])
                            qnums = [0] + (d - a)*[int(p)] + [0]
                        elif c == d:
                            # number operator at the end
                            oids  = (  [p]
                                     + (b - a - 1)*[MolecularOID.Z]
                                     + [q]
                                     + (c - b - 1)*[MolecularOID.I]
                                     + [MolecularOID.N])
                            qnums = [0] + (b - a)*[int(p)] + (c - b + 1)*[0]
                        else:
                            # generic case: i, j, k, l pairwise different
                            oids  = (  [p]
                                     + (b - a - 1)*[MolecularOID.Z]
                                     + [q]
                                     + (c - b - 1)*[MolecularOID.I]
                                     + [r]
                                     + (d - c - 1)*[MolecularOID.Z]
                                     + [s])
                            qnums = (  [0]
                                     + (b - a)*[int(p)]
                                     + (c - b)*[int(p) + int(q)]
                                     + (d - c)*[-int(s)]
                                     + [0])
                        opchains.append(SpinOperatorConverter.to_spin_opchain(
                            OpChain(oids, qnums, coeff, a)))
        graph = OpGraph.from_opchains(opchains, nsites, SpinMolecularOID.I)

    else:
        # explicit construction (typically faster, but does not optimize cases
        # of zero coefficients, and is slightly sub-optimal close to boundary)
        assert nsites >= 2
        nodes = SpinMolecularOpGraphNodes(nsites)
        graph = nodes.generate_graph()
        # kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
        for i in range(nsites):
            for j in range(nsites):
                for sigma in (0, 1):
                    _spin_molecular_hamiltonian_graph_add_term(
                        graph, nodes, [(i, sigma, MolecularOID.C),
                                       (j, sigma, MolecularOID.A)], tkin[i, j])
        # interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau} v_{i,j,k,l}
        #   a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\tau} a_{k,\sigma}
        for i, sigma in itertools.product(range(nsites), (0, 1)):
            for j, tau in itertools.product(range(i, nsites), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                for k, mu in itertools.product(range(nsites), (0, 1)):
                    for l, nu in itertools.product(range(k, nsites), (0, 1)):
                        if (k, mu) >= (l, nu):
                            continue
                        coeff, valid = get_vint_coeff((i, j, k, l), (sigma, tau, mu, nu))
                        if not valid:
                            continue
                        oplist = [(i, sigma, MolecularOID.C),
                                  (j, tau,   MolecularOID.C),
                                  (l, nu,    MolecularOID.A),
                                  (k, mu,    MolecularOID.A)]
                        _spin_molecular_hamiltonian_graph_add_term(
                            graph, nodes, oplist, coeff)

    # skip consistency check for larger nsites (would take very long)
    if nsites <= 10:
        assert graph.is_consistent()
    opmap = _spin_molecular_hamiltonian_generate_operator_map()
    # convert to MPO
    mpo = MPO.from_opgraph(qsite, graph, opmap, compute_nid_map=(not optimize))
    # store node information in MPO, to identify virtual
    # bonds by creation and annihilation operators
    if not optimize:
        nodes.copy_nids(mpo)

    return mpo
