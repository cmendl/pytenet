"""
Construct the Heisenberg Hamiltonian as a matrix product operator (MPO).
"""

from enum import IntEnum
import numpy as np
from .mpo import MPO
from .opchain import OpChain
from .hamiltonian_util import local_opchains_to_mpo

__all__ = ["heisenberg_xxz_1d_mpo", "heisenberg_xxz_spin1_1d_mpo"]


def heisenberg_xxz_1d_mpo(nsites: int, J: float, D: float, h: float) -> MPO:
    """
    Construct XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a one-dimensional lattice as an MPO.

    Args:
        nsites: number of lattice sites
        J:      J parameter
        D:      Delta parameter
        h:      field strength

    Returns:
        MPO: XXZ Heisenberg Hamiltonian
    """
    # physical quantum numbers (multiplied by 2)
    qsite = [1, -1]
    # spin operators
    sup = np.array([[0.,  1.], [0.,  0. ]])
    sdn = np.array([[0.,  0.], [1.,  0. ]])
    sz  = np.array([[0.5, 0.], [0., -0.5]])
    # operator map
    class OID(IntEnum):
        """
        Local operator identifiers.
        """
        SD = -1
        I  =  0
        SU =  1
        SZ =  2
    opmap = {
        OID.SD: sdn,
        OID.I:  np.identity(2),
        OID.SU: sup,
        OID.SZ: sz }
    # local two-site and single-site terms
    lopchains = [OpChain([OID.SU, OID.SD], [0,  2, 0], 0.5*J, 0),
                 OpChain([OID.SD, OID.SU], [0, -2, 0], 0.5*J, 0),
                 OpChain([OID.SZ, OID.SZ], [0,  0, 0],   D,   0),
                 OpChain([OID.SZ        ], [0,  0   ],  -h,   0)]
    # convert to MPO
    return local_opchains_to_mpo(qsite, lopchains, nsites, opmap, OID.I)


def heisenberg_xxz_spin1_1d_mpo(nsites: int, J: float, D: float, h: float) -> MPO:
    """
    Construct spin-1 XXZ Heisenberg Hamiltonian `sum J X X + J Y Y + D Z Z - h Z`
    on a one-dimensional lattice as an MPO.

    Args:
        nsites: number of lattice sites
        J:      J parameter
        D:      Delta parameter
        h:      field strength

    Returns:
        MPO: spin-1 XXZ Heisenberg Hamiltonian
    """
    # physical quantum numbers
    qsite = [1, 0, -1]
    # spin operators
    sq2 = np.sqrt(2.)
    sup = np.array([[0.,  sq2, 0.], [0.,  0.,  sq2], [0.,  0.,  0.]])
    sdn = np.array([[0.,  0.,  0.], [sq2, 0.,  0. ], [0.,  sq2, 0.]])
    sz  = np.array([[1.,  0.,  0.], [0.,  0.,  0. ], [0.,  0., -1.]])
    # operator map
    class OID(IntEnum):
        """
        Local operator identifiers.
        """
        SD = -1
        I  =  0
        SU =  1
        SZ =  2
    opmap = {
        OID.SD: sdn,
        OID.I:  np.identity(3),
        OID.SU: sup,
        OID.SZ: sz }
    # local two-site and single-site terms
    lopchains = [OpChain([OID.SU, OID.SD], [0,  1, 0], 0.5*J, 0),
                 OpChain([OID.SD, OID.SU], [0, -1, 0], 0.5*J, 0),
                 OpChain([OID.SZ, OID.SZ], [0,  0, 0],   D,   0),
                 OpChain([OID.SZ        ], [0,  0   ],  -h,   0)]
    # convert to MPO
    return local_opchains_to_mpo(qsite, lopchains, nsites, opmap, OID.I)
