.. image:: https://travis-ci.com/cmendl/pytenet.svg?branch=master
  :target: https://travis-ci.com/cmendl/pytenet


PyTeNet
=======

PyTeNet is a Python implementation of quantum tensor network operations and simulations within the
matrix product state framework, using NumPy to handle tensors.

Example usage for TDVP time evolution (assuming `psi` is a matrix product state):

.. code-block:: python

    import pytenet as ptn
    
    # number of lattice sites (1D with open boundary conditions)
    L = 10
    
    # construct matrix product operator representation of
    # Heisenberg XXZ Hamiltonian (arguments are L, J, \Delta, h)
    mpoH = ptn.heisenberg_XXZ_MPO(L, 1.0, 0.8, -0.1)
    
    # run TDVP time evolution
    # time step can have both real and imaginary parts;
    # for real-time evolution use purely imaginary dt!
    dt = 0.02 - 0.05j
    numsteps = 100
    ptn.integrate_local_singlesite(mpoH, psi, dt, numsteps, numiter_lanczos=5)
    # psi now stores the time-evolved state exp(-dt*numsteps H) psi


Features
--------
- matrix product state and operator classes
- construct common Hamiltonians as MPOs, straightforward to adapt to custom Hamiltonians
- convert arbitrary operator chains to MPOs
- TDVP time evolution (real and imaginary)
- generate vector / matrix representations of matrix product states / operators
- Krylov subspace methods for local operations
- one-site local energy minimization using Lanczos iteration
- built-in support for additive quantum numbers


Installation
------------
To install PyTeNet, download the `source code <https://github.com/cmendl/pytenet>`_ and
run ``python setup.py install`` from within the main PyTeNet directory, or
add the `pytenet/` subfolder to your Python search path.


Directory structure
-------------------
- **pytenet**: source code of the actual PyTeNet package
- **doc**: documentation and tutorials
- **test**: unit tests (might serve as detailed documentation, too)
- **experiments**: numerical experiments on more advanced, in-depth topics
- **paper**: JOSS manuscript


License
-------
PyTeNet is licensed under the BSD 2-Clause license.


References
----------
1. | U. Schollwöck
   | The density-matrix renormalization group in the age of matrix product states
   | Ann. Phys. 326, 96-192 (2011) `arXiv:1008.3477 <https://arxiv.org/abs/1008.3477>`_, `DOI <https://doi.org/10.1016/j.aop.2010.09.012>`_
2. | J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
   | Unifying time evolution and optimization with matrix product states
   | Phys. Rev. B 94, 165116 (2016) `arXiv:1408.5056 <https://arxiv.org/abs/1408.5056>`_, `DOI <https://doi.org/10.1103/PhysRevB.94.165116>`_
3. | I. P. McCulloch
   | From density-matrix renormalization group to matrix product states
   | J. Stat. Mech. (2007) P10014 `arXiv:cond-mat/0701428 <https://arxiv.org/abs/cond-mat/0701428>`_, `DOI <https://doi.org/10.1088/1742-5468/2007/10/P10014>`_
4. | T. Barthel
   | Precise evaluation of thermal response functions by optimized density matrix renormalization group schemes
   | New J. Phys. 15, 073010 (2013) `arXiv:1301.2246 <https://arxiv.org/abs/1301.2246>`_, `DOI <https://doi.org/10.1088/1367-2630/15/7/073010>`_
