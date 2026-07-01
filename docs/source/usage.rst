Usage
=====

.. currentmodule:: atomic_femdvr

The two top-level entry points are re-exported on the package itself,
so ``from atomic_femdvr import solve_atomic, solve_pseudo_atomic`` works
out of the box. Detailed signatures live in the submodule sections below.

Atomic (all-electron) solver
----------------------------

.. automodule:: atomic_femdvr.full_atomic
    :members:

Pseudo-atomic solver
--------------------

.. automodule:: atomic_femdvr.pseudo_atomic
    :members:

Input models
------------

.. automodule:: atomic_femdvr.input
    :members:

FEM-DVR basis
-------------

.. automodule:: atomic_femdvr.femdvr
    :members:

Bessel transforms and projector output
--------------------------------------

.. automodule:: atomic_femdvr.bessel_transform
    :members:

.. automodule:: atomic_femdvr.projector_output
    :members:
