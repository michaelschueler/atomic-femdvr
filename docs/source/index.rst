atomic-femdvr |release| Documentation
=====================================

``atomic-femdvr`` solves the radial Kohn-Sham equation for an isolated
atom on a finite-element discrete-variable (FEM-DVR) basis. It supports
both all-electron calculations and norm-conserving pseudopotentials read
from UPF files.

The package exposes two top-level entry points:

- :func:`atomic_femdvr.solve_atomic` — all-electron DFT, configured via
  :class:`atomic_femdvr.FullAtomicInput`.
- :func:`atomic_femdvr.solve_pseudo_atomic` — pseudo-atomic DFT,
  configured via :class:`atomic_femdvr.PseudoAtomicInput`, returning a
  :class:`atomic_femdvr.PseudoAtomicResult`.

A console script ``atomic_femdvr`` (also runnable as
``python -m atomic_femdvr``) wraps both solvers with JSON-driven inputs.

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :name: start

    installation
    usage
    cli

Indices and Tables
------------------

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
