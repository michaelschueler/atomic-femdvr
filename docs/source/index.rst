atomic-femdvr |release|
=======================

``atomic-femdvr`` solves the radial Kohn-Sham equations of an isolated atom with the
finite-element discrete variable representation (FE-DVR). It handles

- **pseudo-atoms** with norm-conserving pseudopotentials in UPF format (the main use case),
  including pseudopotentials without pseudo-atomic orbitals such as the SG15 set;
- **all-electron atoms** (non-relativistic; validated against the NIST atomic reference data);
- approximate **scalar-relativistic** (Koelling-Harmon) and **ZORA** treatments of the
  all-electron atom.

For pseudo-atoms it generates confined, numerically exact atomic orbitals (including
polarization orbitals) that can be exported for use as localized basis functions or
projectors, e.g. in Quantum ESPRESSO format or as spherical Bessel transforms.

.. toctree::
    :maxdepth: 2
    :caption: Getting started

    installation
    usage

.. toctree::
    :maxdepth: 2
    :caption: Reference

    input
    cli
    method
    validation
    api

Indices and tables
------------------

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
