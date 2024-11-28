"""
Implements Compression algorithms for MPS
"""

__all__ = ['rounding']

from numpy import array, tensordot, zeros_like
from numpy.linalg import eigh

from .bond_ops import eigh
from .operation import compute_left_state_blocks_conj
from .mps import MPS
from .qnumber import qnumber_flatten

def rounding(mps: MPS, tol: float) -> MPS:
    """
    Implements the MPS rounding found in doi: 10.1137/090752286
    
    Args:
        mps: The MPS to be rounded.
    """
    nsites = mps.nsites
    left_envs = compute_left_state_blocks_conj(mps)
    new_mps_tensors = [None for _ in range(nsites)]
    new_virtual_qnums = [zeros_like(qDs) for qDs in mps.qD]
    new_virtual_qnums[-1] = mps.qD[-1]
    # Implement trivial initial tensors
    b_block = array([1], dtype=complex).reshape(1,1,1)
    current_u = array([1], dtype=complex).reshape(1,1,1)
    for site in reversed(range(1,nsites)):
        # Compute new b_block
        b_block = tensordot(b_block,
                            current_u,
                            axes=([0,2], [0,2]))
        b_block = tensordot(mps.A[site],
                            b_block,
                            axes=([2], [0]))
        # Compute density matrix
        rho = tensordot(b_block,
                        left_envs[site],
                        axes=([1], [0]))
        rho = tensordot(rho,
                        b_block.conj(),
                        axes=([2],[1]))
        # Diagonalise density matrix
        orig_shape = rho.shape[0:2]
        new_shape = orig_shape[0]*orig_shape[1]
        rho = rho.reshape(new_shape, new_shape)
        rho_qn = qnumber_flatten([mps.qd, -new_virtual_qnums[site+1]])
        evecs, _, new_qn = eigh(rho, rho_qn, tol=tol)
        # Get current_u
        current_u = evecs.reshape(evecs.shape[1],
                                  orig_shape[0],
                                  orig_shape[1]).transpose([1,0,2])
        new_mps_tensors[site] = current_u
        new_virtual_qnums[site] = new_qn
    # For the leftmost site, we merely nedd to find the block, as the bond is already truncated
    b_block = tensordot(b_block,
                        current_u,
                        axes=([0,2], [0,2]))
    b_block = tensordot(mps.A[0],
                        b_block,
                        axes=([2], [0]))
    new_mps_tensors[0] = b_block
    new_virtual_qnums[0] = mps.qD[0]
    # Create compressed mps
    new_mps = MPS(mps.qd, new_virtual_qnums, fill='postpone')
    new_mps.A = new_mps_tensors
    return new_mps
