import numpy as np
from .hamiltonian import quadratic_spin_fermionic_mpo

__all__ = ['THCSpinMolecularHamiltonian']


class THCSpinMolecularHamiltonian:
    """
    Tensor hypercontraction (THC) representation of a molecular Hamiltonian,
    assuming a spin orbital basis.
    """
    def __init__(self, tkin, thc_kernel, thc_transform):

        tkin          = np.asarray(tkin)
        thc_kernel    = np.asarray(thc_kernel)
        thc_transform = np.asarray(thc_transform)
        # dimension, data type and symmetry consistency checks
        assert tkin.ndim == 2
        assert tkin.shape[0] == tkin.shape[1]
        assert tkin.shape[0] >= 1
        assert np.allclose(tkin, tkin.conj().T)
        assert np.isrealobj(thc_kernel)  # require real data types due to use of elementary THC MPOs
        assert np.isrealobj(thc_transform)
        assert thc_kernel.ndim == 2
        assert thc_transform.ndim == 2
        assert thc_kernel.shape[0] == thc_kernel.shape[1]
        assert np.allclose(thc_kernel, thc_kernel.T)
        assert thc_transform.shape[1] == thc_kernel.shape[0]
        assert thc_transform.shape[0] == tkin.shape[0]

        self.tkin          = tkin
        self.thc_kernel    = thc_kernel
        self.thc_transform = thc_transform

        # generate the internal elementary MPO terms:
        # MPOs for the kinetic term
        self.en_kin, self.u_kin = np.linalg.eigh(self.tkin) # diagonalize kinetic coefficient matrix
        self.mpo_kin = [[
            quadratic_spin_fermionic_mpo(
                self.u_kin[:, i],
                self.u_kin[:, i].conj(),
                [1, -1][sigma])
            for sigma in (0, 1)]
            for i in range(self.u_kin.shape[1])]
        # elementary MPOs for the interaction (Coulomb) term in THC representation
        self.mpo_thc = [[
            quadratic_spin_fermionic_mpo(
                self.thc_transform[:, mu],
                self.thc_transform[:, mu],
                [1, -1][sigma])
            for sigma in (0, 1)]
            for mu in range(self.thc_rank)]

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return self.tkin.shape[0]

    @property
    def thc_rank(self) -> int:
        """
        Tensor hypercontraction rank (dimension of the kernel).
        """
        return self.thc_kernel.shape[0]

    def as_matrix(self, sparse_format:bool=False):
        """
        Generate the matrix representation of the Hamiltonian on the full Hilbert space.
        """
        # kinetic term
        mat = sum(self.en_kin[i] * self.mpo_kin[i][sigma].as_matrix(sparse_format)
                  for sigma in (0, 1)
                  for i in range(self.nsites))
        # convert individual THC MPOs to sparse matrices
        mat_thc = [[self.mpo_thc[mu][sigma].as_matrix(sparse_format)
                    for sigma in (0, 1)]
                    for mu in range(self.thc_rank)]
        # diagonalize the THC kernel
        lambda_kernel, u_kernel = np.linalg.eigh(self.thc_kernel)
        assert len(lambda_kernel) == self.thc_rank
        # add interaction terms
        for i in range(self.thc_rank):
            g = sum(u_kernel[mu, i] * mat_thc[mu][sigma]
                    for sigma in (0, 1)
                    for mu in range(self.thc_rank))
            mat += 0.5 * lambda_kernel[i] * (g @ g)
        return mat
