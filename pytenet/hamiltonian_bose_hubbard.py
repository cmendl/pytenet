"""
Construct the Bose-Hubbard Hamiltonian as a matrix product operator (MPO).
"""

from enum import IntEnum
import numpy as np
from .mpo import MPO
from .opchain import OpChain
from .hamiltonian_util import local_opchains_to_mpo

__all__ = ["bose_hubbard_1d_mpo"]


def bose_hubbard_1d_mpo(nsites: int, d: int, t: float, u: float, mu: float) -> MPO:
    """
    Construct Bose-Hubbard Hamiltonian
    with nearest-neighbor hopping on a 1D lattice as an MPO.

    Args:
        nsites: number of lattice sites
        d:      physical dimension per site
                (allowed local occupancies are 0, 1, ..., d - 1)
        t:      kinetic hopping parameter
        u:      interaction strength
        mu:     chemical potential

    Returns:
        MPO: Bose-Hubbard Hamiltonian
    """
    # physical quantum numbers (particle number)
    qsite = np.arange(d)
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    # operator map
    class OID(IntEnum):
        """
        Local operator identifiers.
        """
        B  = -1
        I  =  0
        BD =  1
        N  =  2
        NI =  3
    opmap = {
        OID.B:  b_ann,
        OID.I: np.identity(d),
        OID.BD: b_dag,
        OID.N:  numop,
        OID.NI: numop @ (numop - np.identity(d)) / 2 }
    # local two-site and single-site terms
    lopchains = [OpChain([OID.BD, OID.B ], [0,  1, 0], -t,  0),
                 OpChain([OID.B,  OID.BD], [0, -1, 0], -t,  0),
                 OpChain([OID.N         ], [0,  0   ], -mu, 0),
                 OpChain([OID.NI        ], [0,  0   ],  u,  0)]
    # convert to MPO
    return local_opchains_to_mpo(qsite, lopchains, nsites, opmap, OID.I)
