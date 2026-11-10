Validation
==========

The benchmarks below are part of the test suite (``tests/``).

All-electron atoms: NIST reference data
---------------------------------------

The non-relativistic all-electron solver reproduces the LDA eigenvalues of the NIST atomic
reference data [NIST97]_ (``lda_x`` + ``lda_c_vwn``). Test case: Ne with
``h_min = 0.25``, ``h_max = 10``, ``Rmax = 100``, ``ng = 12``; tolerance 2e-6 Ha.

.. list-table::
    :header-rows: 1

    * - Ne
      - NIST (Ha)
    * - 1s
      - −30.305855
    * - 2s
      - −1.322809
    * - 2p
      - −0.498034

The scalar-relativistic and ZORA options are approximate and are not benchmarked.

Pseudo-atoms: ONCVPSP reference energies
----------------------------------------

A norm-conserving pseudopotential reproduces the all-electron valence eigenvalues of its
reference configuration. ONCVPSP [Ham13]_ stores these all-electron eigenvalues (column
``ep`` in ``PP_INPUTFILE``, 5 decimals) in the UPF file. Results with the default grid
(``ng = 12``, PBE):

.. list-table::
    :header-rows: 1

    * - Pseudopotential
      - State
      - atomic-femdvr (Ha)
      - ONCVPSP (Ha)
      - Difference (meV)
    * - Mo (``tests/data``)
      - 4s
      - −2.364727
      - −2.36471
      - −0.5
    * -
      - 4p
      - −1.414339
      - −1.41432
      - −0.5
    * -
      - 4d
      - −0.137903
      - −0.13792
      - +0.5
    * - Si (``tests/data``)
      - 3s
      - −0.397366
      - −0.39736
      - −0.2
    * -
      - 3p
      - −0.149982
      - −0.14998
      - −0.05
    * - Si, SG15 v1.2 [SG15V]_ (no PAOs, ``nscf``)
      - 3s
      - −0.397414
      - −0.39736
      - −1.5
    * -
      - 3p
      - −0.149996
      - −0.14998
      - −0.4

The remaining differences of about 0.5 meV come from interpolating the UPF data and from
differences between the PBE implementations. For the SG15 file, the density in
``PP_RHOATOM`` ends at the end of the radial mesh (6 bohr) and is extrapolated beyond it,
which adds an uncertainty of about 1–5 meV.

For pseudopotentials with pseudo-atomic orbitals, an ``nscf`` calculation on the density
stored in the UPF file reproduces the self-consistent result to 0.2 meV (tested by removing
the orbitals from the file).

.. [NIST97] S. Kotochigova, Z. H. Levine, E. L. Shirley, M. D. Stiles, and C. W. Clark,
   Phys. Rev. A **55**, 191 (1997); https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations
.. [Ham13] D. R. Hamann, Phys. Rev. B **88**, 085117 (2013).
.. [SG15V] M. Schlipf and F. Gygi, Comput. Phys. Commun. **196**, 36 (2015).
