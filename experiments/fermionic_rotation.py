"""
Represent a fermionic rotation (orbital base change) operation as a product of simple MPOs.
"""

import time
import autoray as ar
from autoray import numpy as np
import pytenet as ptn
import sys
sys.path.append("../test/hamiltonian/")
from fermi_operators import construct_fermi_operators


def main():

    rng = ar.do("random.default_rng", int(time.time()))

    # number of sites (or fermionic modes)
    nsites = 7

    # random single-particle Hamiltonian,
    # generating the orbital rotation matrix expm(-i h)
    h = ptn.crandn((nsites, nsites), rng=rng)
    h = 0.5*(h + h.conj().T)
    # Hamiltonian on the whole Fock space
    clist, alist, _ = construct_fermi_operators(nsites)
    hfock = sum(h[i, j] * (clist[i] @ alist[j]) for i in range(nsites) for j in range(nsites))

    # rotation matrix on the whole Fock space
    ufock_ref = ar.do("linalg.expm", -1j*np.asarray(hfock.toarray()))

    # express the rotation matrix in terms of "orbital" number operators
    # diagonalize 'h'
    eigvals, eigvecs = np.linalg.eigh(h)
    # represent I - (1 - exp(-i lambda_j)) \tilde{n}_j as MPO
    n_mpo_list = [ptn.quadratic_fermionic_mpo(
            -(1 - np.exp(-1j*eigval)) * eigvec, eigvec.conj(), add_identity=True)
        for eigval, eigvec in zip(eigvals, eigvecs.T)]
    # product of number operator terms
    ufock = ptn.MPO.identity([0, 1], nsites, dtype=complex)
    for i in range(nsites):
        ufock = ufock @ n_mpo_list[i]
        nrm = ufock.orthonormalize(mode="left")
        ufock.a[-1] *= nrm
        nrm = ufock.orthonormalize(mode="right")
        ufock.a[0] *= nrm
        print("ufock.bond_dims:", ufock.bond_dims)
    # compare
    err = np.linalg.norm(ufock.to_matrix() - ufock_ref)
    print(f"|ufock_mpo - ufock_ref|: {err} (should be numerically zero)")


if __name__ == "__main__":
    main()
