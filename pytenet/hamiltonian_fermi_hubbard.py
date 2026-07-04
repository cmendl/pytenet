"""
Construct the Fermi-Hubbard Hamiltonian as a matrix product operator (MPO).
"""

from enum import IntEnum
import numpy as np
from .mpo import MPO
from .opchain import OpChain
from .qnumber import encode_quantum_number_pair
from .hamiltonian_util import local_opchains_to_mpo

__all__ = ["fermi_hubbard_1d_mpo"]


def fermi_hubbard_1d_mpo(nsites: int, t: float, u: float, mu: float) -> MPO:
    """
    Construct Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping on a one-dimensional lattice as an MPO.

    States for each spin and site are `|0>` and `|1>`.

    Args:
        nsites: number of lattice sites
        t:      kinetic hopping parameter
        u:      Hubbard interaction strength
        mu:     chemical potential (mu = 0 corresponds to half-filling)

    Returns:
        MPO: Fermi-Hubbard Hamiltonian
    """
    # physical particle number and spin quantum numbers (encoded as single integer)
    qn = [0,  1,  1,  2]
    qs = [0, -1,  1,  0]
    qsite = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qn, qs)]
    id2 = np.identity(2)
    # creation and annihilation operators for a single spin and lattice site
    a_dag = np.array([[0., 0.], [1., 0.]])
    a_ann = np.array([[0., 1.], [0., 0.]])
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
        CI =  1
        AI =  2
        CZ =  3
        AZ =  4
        IC =  5
        IA =  6
        ZC =  7
        ZA =  8
        NT =  9
        NI = 10
    opmap = {
        OID.I:  np.identity(4),
        OID.CI: np.kron(a_dag, id2),
        OID.AI: np.kron(a_ann, id2),
        OID.CZ: np.kron(a_dag, z  ),
        OID.AZ: np.kron(a_ann, z  ),
        OID.IC: np.kron(id2, a_dag),
        OID.IA: np.kron(id2, a_ann),
        OID.ZC: np.kron(z,   a_dag),
        OID.ZA: np.kron(z,   a_ann),
        OID.NT: np.kron(numop, id2) + np.kron(id2, numop),  # n_up + n_dn
        OID.NI: np.diag([0.25, -0.25, -0.25, 0.25])         # (n_up - 1/2) (n_dn - 1/2)
    }
    # local two-site and single-site terms
    lopchains = [
        # spin-up kinetic hopping
        OpChain([OID.CZ, OID.AI], [0, encode_quantum_number_pair( 1,  1), 0], -t, 0),
        OpChain([OID.AZ, OID.CI], [0, encode_quantum_number_pair(-1, -1), 0], -t, 0),
        # spin-down kinetic hopping
        OpChain([OID.IC, OID.ZA], [0, encode_quantum_number_pair( 1, -1), 0], -t, 0),
        OpChain([OID.IA, OID.ZC], [0, encode_quantum_number_pair(-1,  1), 0], -t, 0),
        # number operator - mu (n_up + n_dn)
        OpChain([OID.NT], [0, 0], -mu, 0),
        # interaction u (n_up - 1/2) (n_dn - 1/2)
        OpChain([OID.NI], [0, 0],  u,  0)]
    # convert to MPO
    return local_opchains_to_mpo(qsite, lopchains, nsites, opmap, OID.I)
