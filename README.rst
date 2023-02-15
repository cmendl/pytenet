.. image:: https://github.com/cmendl/pytenet/actions/workflows/ci.yml/badge.svg?branch=master
  :target: https://github.com/cmendl/pytenet/actions/workflows/ci.yml
.. image:: http://joss.theoj.org/papers/10.21105/joss.00948/status.svg
  :target: https://doi.org/10.21105/joss.00948


PyTeNet
=======

.. doc-inclusion-marker1-start

`PyTeNet <https://github.com/cmendl/pytenet>`_ is a Python implementation of quantum
tensor network operations and simulations within the matrix product state framework,
using NumPy to handle tensors.

Example usage for TDVP time evolution:

.. code-block:: python

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


Features
--------
- matrix product state and operator classes
- construct common Hamiltonians as MPOs, straightforward to adapt to custom Hamiltonians
- convert arbitrary operator chains to MPOs
- TDVP time evolution (single- and two-site, both real and imaginary time)
- generate vector / matrix representations of matrix product states / operators
- Krylov subspace methods for local operations
- one-site local energy minimization using Lanczos iteration
- built-in support for additive quantum numbers


Installation
------------
To install PyTeNet from PyPI, call

.. code-block:: python

    python3 -m pip install pytenet

Alternatively, you can clone the `repository <https://github.com/cmendl/pytenet>`_ and install it in development mode via

.. code-block:: python

    python3 -m pip install -e <path/to/repo>

.. doc-inclusion-marker1-end


Documentation
-------------
The full documentation is available at `pytenet.readthedocs.io <https://pytenet.readthedocs.io>`_.


Directory structure
-------------------
- **pytenet**: source code of the actual PyTeNet package
- **doc**: documentation and tutorials
- **test**: unit tests (might serve as detailed documentation, too)
- **experiments**: numerical experiments on more advanced, in-depth topics
- **paper**: JOSS manuscript


.. doc-inclusion-marker2-start

Contributing
------------
Feature requests, discussions and code contributions to PyTeNet in the form of
`pull requests <https://github.com/cmendl/pytenet/pulls>`_ are of course welcome.
Creating an `issue <https://github.com/cmendl/pytenet/issues>`_ might be a good starting point.
New code should be well documented (`Google style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_)
and unit-tested (see the `test/` subfolder).
For questions and additional support, fell free to contact christian.b.mendl@gmail.com


Citing
------
PyTeNet is `published <https://doi.org/10.21105/joss.00948>`_ in the Journal of Open Source Software -
if it's ever useful for a research project please consider citing it:

.. code-block:: latex

    @ARTICLE{pytenet,
      author = {Mendl, C. B.},
      title = {PyTeNet: A concise Python implementation of quantum tensor network algorithms},
      journal = {Journal of Open Source Software},
      year = {2018},
      volume = {3},
      number = {30},
      pages = {948},
      doi = {10.21105/joss.00948},
    }


License
-------
PyTeNet is licensed under the BSD 2-Clause license.


References
----------
1. | U. Schollwöck
   | The density-matrix renormalization group in the age of matrix product states
   | `Ann. Phys. 326, 96-192 (2011) <https://doi.org/10.1016/j.aop.2010.09.012>`_ (`arXiv:1008.3477 <https://arxiv.org/abs/1008.3477>`_)
2. | J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
   | Unifying time evolution and optimization with matrix product states
   | `Phys. Rev. B 94, 165116 (2016) <https://doi.org/10.1103/PhysRevB.94.165116>`_ (`arXiv:1408.5056 <https://arxiv.org/abs/1408.5056>`_)
3. | I. P. McCulloch
   | From density-matrix renormalization group to matrix product states
   | `J. Stat. Mech. (2007) P10014 <https://doi.org/10.1088/1742-5468/2007/10/P10014>`_ (`arXiv:cond-mat/0701428 <https://arxiv.org/abs/cond-mat/0701428>`_)
4. | T. Barthel
   | Precise evaluation of thermal response functions by optimized density matrix renormalization group schemes
   | `New J. Phys. 15, 073010 (2013) <https://doi.org/10.1088/1367-2630/15/7/073010>`_ (`arXiv:1301.2246 <https://arxiv.org/abs/1301.2246>`_)

.. doc-inclusion-marker2-end
