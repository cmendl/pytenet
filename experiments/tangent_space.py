"""
Numerically investigate properties of the MPS tangent space,
like dimensionality and its projector.

Reference:
  - J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete\n
    Unifying time evolution and optimization with matrix product states\n
    Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
"""

import copy
import numpy as np
import scipy
import pytenet as ptn


def tangent_space_projector(psi: ptn.MPS):
    """
    Construct tangent space projector as matrix based on MPS formalism.
    """
    # physical local site dimension
    d = len(psi.qsite)

    # number of lattice sites
    nsites = psi.nsites

    psi_c = copy.deepcopy(psi)

    # construct P_L operators
    psi_c.orthonormalize(mode="left")
    p_left = []
    x = np.array([[[1.]]])
    for i in range(nsites):
        x = ptn.mps_merge_tensor_pair(x, psi_c.a[i])
        assert x.ndim == 3 and x.shape[1] == 1
        xmat = x.reshape((x.shape[0], x.shape[2]))
        # check orthonormalization
        assert np.allclose(xmat.conj().T @ xmat, np.identity(xmat.shape[1]))
        p_left.append(xmat @ xmat.conj().T)

    # construct P_R operators
    psi_c.orthonormalize(mode="right")
    p_right = []
    x = np.array([[[1.]]])
    for i in reversed(range(nsites)):
        x = ptn.mps_merge_tensor_pair(psi_c.a[i], x)
        assert x.ndim == 3 and x.shape[2] == 1
        xmat = x.reshape(x.shape[:2])
        # check orthonormalization
        assert np.allclose(xmat.conj().T @ xmat, np.identity(xmat.shape[1]))
        p_right.append(xmat @ xmat.conj().T)
    p_right = list(reversed(p_right))

    # construct projector
    proj = 0
    proj += np.kron(np.identity(d), p_right[1])
    for i in range(1, nsites-1):
        proj += np.kron(np.kron(p_left[i-1], np.identity(d)), p_right[i+1])
    proj += np.kron(p_left[nsites-2], np.identity(d))
    for i in range(0, nsites-1):
        proj -= np.kron(p_left[i], p_right[i+1])

    return proj


def main():

    rng = np.random.default_rng()

    # physical dimension
    d = 3
    # fictitious bond dimensions (should be bounded by d^i and d^(nsites-i))
    b = [1, 2, 5, 7, 3, 1]

    # number of lattice sites
    nsites = len(b) - 1
    print("nsites:", nsites)

    psi = ptn.MPS(np.zeros(d, dtype=int), [np.zeros(bi, dtype=int) for bi in b],
                  fill="random", rng=rng)

    # construct MPS derivatives with respect to entries of the `a` tensors
    t = []
    for i in range(nsites):
        s = psi.a[i].shape
        print("s:", s)
        for j in range(s[0]):
            for u in range(s[1]):
                for v in range(s[2]):
                    # derivative in direction (i, j, u, v)
                    ddir = np.zeros_like(psi.a[i])
                    ddir[j, u, v] = 1
                    # backup original a[i] tensor
                    ai = psi.a[i]
                    psi.a[i] = ddir
                    t.append(psi.to_vector())
                    # restore a[i]
                    psi.a[i] = ai

    t = np.array(t)
    num_entries = np.sum([ai.size for ai in psi.a])
    print("t.shape:", t.shape)
    print("expected:", (num_entries, d**nsites))
    print("rank of t:", np.linalg.matrix_rank(t))
    # number of degrees of freedom based on sandwiching "X" matrices between bonds,
    # -2 for omitting the leading and trailing entry 1 in `b`
    rank = num_entries - ((np.array(b)**2).sum() - 2)
    print("expected: ", rank)

    # realization of random X matrices
    x_list = [np.identity(1, dtype=complex)]
    for i in range(nsites - 1):
        x_list.append(ptn.crandn((b[i+1], b[i+1]), rng))
    x_list.append(np.identity(1, dtype=complex))
    n = []
    for i in range(nsites):
        bmat = (np.tensordot(x_list[i], psi.a[i], axes=(1, 1)).transpose((1, 0, 2)) -
                np.tensordot(psi.a[i], x_list[i+1], axes=(2, 0)))
        # backup original a[i] tensor
        ai = psi.a[i]
        psi.a[i] = bmat
        n.append(psi.to_vector())
        # restore a[i]
        psi.a[i] = ai
    n = np.array(n)
    print("n.shape:", n.shape)
    # `n` should be contained in range of `t`
    print(f"rank of [t, n]: {np.linalg.matrix_rank(np.concatenate((t, n), axis=0))} "
          f"(should agree with rank of t)")
    # should be numerically zero by construction
    z = n.transpose() @ np.ones(nsites)
    print(f"|z|: {np.linalg.norm(z)} (should be numerically zero)")

    # reference tangent space projector based on t
    # rank-revealing QR decomposition
    q, _, _ = scipy.linalg.qr(t.T, mode="economic", pivoting=True)
    proj_ref = q[:, :rank] @ q[:, :rank].conj().T

    # construct tangent space projector based on MPS formalism
    proj = tangent_space_projector(psi)
    # compare
    print(f"|proj - proj_ref|: {np.linalg.norm(proj - proj_ref)} (should be numerically zero)")

    # apply projector to psi (psi should remain unaffected)
    x = psi.to_vector()
    print(f"|proj psi - psi|: {np.linalg.norm(proj @ x - x)} (should be numerically zero)")

    # define another state
    # fictitious bond dimensions (should be bounded by d^i and d^(nsites-i))
    b = [1, 4, 7, 5, 3, 1]
    chi = ptn.MPS(np.zeros(d, dtype=int), [np.zeros(bi, dtype=int) for bi in b],
                  fill="random", rng=rng)

    # tangent space projector corresponding to the sum of two states
    proj_sum = tangent_space_projector(psi + chi)
    # apply projector to psi (should remain unaffected)
    x = psi.to_vector()
    print(f"|proj_sum psi - psi|: {np.linalg.norm(proj_sum @ x - x)} (should be numerically zero)")
    # apply projector to chi (should remain unaffected)
    x = chi.to_vector()
    print(f"|proj_sum chi - chi|: {np.linalg.norm(proj_sum @ x - x)} (should be numerically zero)")
    # apply projector to psi + chi (should remain unaffected)
    x = psi.to_vector() + chi.to_vector()
    print(f"|proj_sum (psi + chi) - (psi + chi)|: {np.linalg.norm(proj_sum @ x - x)} "
          f"(should be numerically zero)")


if __name__ == "__main__":
    main()
