"""
Construct a Hamiltonian as an MPO based on local operator chains,
which are shifted along a 1D lattice.
"""

import copy
from collections.abc import Sequence, Mapping
from ..mpo import MPO
from ..opchain import OpChain
from ..opgraph import OpGraph

__all__ = ["local_opchains_to_mpo"]


def local_opchains_to_mpo(qsite, lopchains: Sequence[OpChain],
                          size: int, opmap: Mapping, oid_identity: int) -> MPO:
    """
    Construct a Hamiltonian as an MPO based on local operator chains,
    which are shifted along a 1D lattice.

    Args:
        qsite:        physical quantum numbers at each site
        lopchains:    local operator chains
        size:         system size, i.e., number of lattice sites
        opmap:        local operators as dictionary, using operator IDs as keys
        oid_identity: operator ID for identity map

    Returns:
        MPO: Hamiltonian as MPO
    """
    opchains = []
    for lopc in lopchains:
        for i in range(size - lopc.length + 1):
            # add shifted opchain; shallow copy sufficient here
            chain = copy.copy(lopc)
            chain.istart = i
            opchains.append(chain)
    graph = OpGraph.from_opchains(opchains, size, oid_identity)
    return MPO.from_opgraph(qsite, graph, opmap)
