[![Build Status](https://travis-ci.com/cmendl/pytenet.svg?branch=master)](https://travis-ci.com/cmendl/pytenet)


Tensor networks for quantum simulations
=======================================

Python implementation of quantum tensor network operations and simulations, using NumPy to handle tensors.


Features
--------
- matrix product state and operator classes
- construct common Hamiltonians as MPOs, straightforward to support custom Hamiltonians
- convert arbitrary operator chains to MPOs
- TDVP time evolution (real and imaginary)
- generate vector / matrix representations of matrix product states / operators
- Krylov subspace methods for local operations
- one-site local energy minimization using Lanczos iteration
- built-in support for additive quantum numbers


References
----------
1. U. Schollwöck  
   The density-matrix renormalization group in the age of matrix product states  
   Ann. Phys. 326, 96-192 (2011) [arXiv:1008.3477](https://arxiv.org/abs/1008.3477), [DOI](https://doi.org/10.1016/j.aop.2010.09.012)
2. J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete  
   Unifying time evolution and optimization with matrix product states  
   Phys. Rev. B 94, 165116 (2016) [arXiv:1408.5056](https://arxiv.org/abs/1408.5056), [DOI](https://doi.org/10.1103/PhysRevB.94.165116)
3. T. Barthel  
   Precise evaluation of thermal response functions by optimized density matrix renormalization group schemes  
   New J. Phys. 15, 073010 (2013) [arXiv:1301.2246](https://arxiv.org/abs/1301.2246), [DOI](https://doi.org/10.1088/1367-2630/15/7/073010)
