---
title: 'PyTeNet: A concise Python implementation of quantum tensor network algorithms'
tags:
  - quantum mechanics and simulation
  - DMRG / matrix product state framework
  - quantum tensor networks
  - TDVP time evolution
  - computational physics
  - Python
authors:
 - name: Christian B. Mendl
   orcid: 0000-0002-6386-0230
   affiliation: 1
affiliations:
 - name: Technische Universität Dresden, Institute of Scientific Computing, Zellescher Weg 12-14, 01069 Dresden, Germany
   index: 1
date: 11 October 2018
bibliography: paper.bib
---

# Summary

The simulation of strongly correlated quantum systems on classical computers is an essential tool for investigating and understanding quantum materials. Due to the inherent "curse of dimensionality" of quantum systems, such simulations pose a challenging task, and have spurred the development of several computational approaches and approximations. For quasi one-dimensional lattice systems, the density matrix renormalization group (DMRG) framework [@WhiteDMRG1992; @WhiteDMRG1993; @Schollwock2005] has emerged as one of the most successful methods. Modern formulations are based on matrix product states (MPS) and operators (MPO) [@Schollwock2011]: briefly speaking, a matrix product state is an Ansatz for the quantum many-body wavefunction amendable to efficient numerical manipulation. The favorable approximation properties of matrix product states have been theoretically proven for ground states of gapped Hamiltonians with local interactions [@VerstraeteCirac2006; @Hastings2007], and recent work has extended this to quantum systems at finite temperature [@Barthel2017]. Over the years many variations and generalizations of the DMRG framework have been devised [@Vidal2003; @Vidal2008; @VerstraeteMurgCirac2008; @StoudenmireWhite2010; @Barthel2013], and further developments and improvements of associated algorithms remain an active field of research.

``PyTeNet`` is a concise Python implementation of the core MPS framework; besides its use for research on quantum system, it serves as tool for algorithmic experimentation, as didactic reference and as blueprint for future implementations using, e.g., compiled languages to optimize computational efficiency. Specifically, ``PyTeNet`` employs NumPy [@WaltColbertVaroquaux2011] for tensor manipulations, and is structured around MPS and MPO classes and associated operations, like left- or right-orthonormalization, computing expectation values, local gate operations etc. Features of ``PyTeNet`` include the conversion of arbitrary operator chains (like $\sigma^x_i \sigma^x_{i+1} + \sigma^y_i \sigma^y_{i+1} + \Delta \sigma^z_i \sigma^z_{i+1}$) to MPOs, which facilitates the concise construction of common quantum Hamiltonian operators. ``PyTeNet`` can also generate vector and matrix representations of matrix product states and operators, respectively, thus enabling reference calculations for example. Another major feature is the (real and imaginary) time evolution based on the time-dependent variational principle (TDVP) [@Haegeman2016], making use of Krylov subspace methods for local operations [@HochbruckLubich1997]. Energy minimization and ground state computation are achieved via one-site local energy minimization using Lanczos iteration. ``PyTeNet`` unobtrusively supports additive quantum numbers by associating an integer with each physical and virtual bond dimension, and uses corresponding reference checks on the sparsity patterns of tensors. Several simultaneously conserved quantum numbers (like particle and spin for the Fermi-Hubbard model) are then realized by encoding them as single integers.

TDVP time evolution example usage:
```python
import pytenet as ptn

# number of lattice sites (1D with open boundary conditions)
L = 10

# construct matrix product operator representation of
# Heisenberg XXZ Hamiltonian (arguments are L, J, \Delta, h)
mpoH = ptn.heisenberg_XXZ_MPO(L, 1.0, 0.8, -0.1)
mpoH.zero_qnumbers()

# initial wavefunction as MPS with random entries
# maximally allowed virtual bond dimensions
D = [1, 2, 4, 8, 16, 28, 16, 8, 4, 2, 1]
psi = ptn.MPS(mpoH.qd, [Di*[0] for Di in D], fill='random')
# effectively clamp virtual bond dimension of initial state
Dinit = 8
for i in range(L):
    psi.A[i][:, Dinit:, :] = 0
    psi.A[i][:, :, Dinit:] = 0
psi.orthonormalize(mode='left')

# time step can have both real and imaginary parts;
# for real time evolution use purely imaginary dt!
dt = 0.01 - 0.05j
numsteps = 100

# run TDVP time evolution
ptn.integrate_local_singlesite(mpoH, psi, dt, numsteps, numiter_lanczos=5)
# psi now stores the (approximated) time-evolved state exp(-dt*numsteps H) psi
```

# Acknowledgments

I'd like to thank Thomas Barthel, Lexing Ying, Miles Stoudenmire and Christian Lubich for inspiring discussions.

# References
