Input file
==========

The input is a JSON file with the sections below. Keys are validated: unknown or
misspelled keys are rejected. String options are case-insensitive. Energies are in
Hartree, lengths in bohr.

.. list-table:: Sections
    :header-rows: 1
    :widths: 20 20 60

    * - Section
      - Used by
      - Content
    * - ``control``
      - all
      - Storage of the density/potential
    * - ``sysparams``
      - all
      - Element, pseudopotential, number of orbitals
    * - ``solver``
      - all
      - Radial grid and eigensolver
    * - ``dft``
      - all
      - Exchange-correlation, self-consistency, relativistic treatment
    * - ``electrons``
      - all-electron
      - Electronic configuration
    * - ``confinement``
      - pseudo-atom
      - Confinement and polarization orbitals
    * - ``output``
      - pseudo-atom (and ``output_wfc_pyimpurity`` for all-electron)
      - Exported files
    * - ``photoemission``
      - photoemission commands
      - Energy grid and output

``control``
-----------

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Key
      - Default
      - Description
    * - ``storage_dir``
      - ``"."``
      - Directory for ``<element>_density_potential.h5`` (density and potential after
        ``scf``). Later runs on the same grid restart from this file.
    * - ``store_density``
      - ``true``
      - Write the file above.

``sysparams``
-------------

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Key
      - Default
      - Description
    * - ``element``
      - required
      - Element symbol, e.g. ``"Si"``.
    * - ``file_upf``
      - required for pseudo-atoms
      - Norm-conserving pseudopotential in UPF v2 format.
    * - ``lmax``
      - ``0``
      - Highest angular momentum of the ``nscf`` orbitals. Channels above the highest
        valence channel of the pseudopotential are polarization channels.
    * - ``nmax``
      - ``4``
      - ``nscf`` computes ``nmax + 1`` radial functions per channel (``nmax = 1``: double
        zeta).
    * - ``oc_override``
      - ``null``
      - Replacement occupations of the pseudo-atomic orbitals, in the order of
        ``PP_PSWFC``. For files with wrong occupations (e.g. GTH pseudopotentials converted
        with ``cpmd2upf``) or for ions.

``solver``
----------

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Key
      - Default
      - Description
    * - ``h_min``
      - ``0.5``
      - Smallest finite-element size.
    * - ``h_max``
      - ``4.0``
      - Largest finite-element size.
    * - ``Rmax``
      - pseudo: ``30``; all-electron: ``50 (lmax + 1)``
      - Radius of the radial box.
    * - ``ng``
      - ``12``
      - Gauss-Lobatto points per element (polynomial order). ``12`` converges eigenvalues
        to about 1e-7 Ha.
    * - ``elem_tol``
      - ``0.01``
      - Tolerance of the adaptive element placement; smaller values give more elements.
    * - ``eigensolver``
      - ``"full"``
      - All-electron only: ``"full"`` or ``"banded"`` (faster for large grids).
    * - ``nuclear_sigma``
      - ``0.001``
      - Width of the Gaussian nuclear charge used in the relativistic mass term.

For all-electron atoms, ``h_min``, ``h_max`` and ``Rmax`` are scaled by
:math:`Z^{-1/3}`, so the same values work across the periodic table.

``dft``
-------

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Key
      - Default
      - Description
    * - ``driver``
      - ``"internal"``
      - ``"internal"`` (built-in PBE) or ``"pylibxc"`` (any LDA/GGA functional of libxc).
    * - ``xc_functional``
      - ``"PBE"``
      - ``internal``: ``"PBE"``. ``pylibxc``: name of a combined exchange-correlation
        functional; leave empty (``""``) to use ``x_functional`` and ``c_functional``.
    * - ``x_functional``
      - ``"gga_x_pbe"``
      - libxc exchange functional, e.g. ``"lda_x"``.
    * - ``c_functional``
      - ``"gga_c_pbe"``
      - libxc correlation functional, e.g. ``"lda_c_vwn"``.
    * - ``alpha_x``
      - ``1.0``
      - Scaling of the exchange part.
    * - ``max_iter``
      - ``100``
      - Maximum number of self-consistency iterations (``0``: skip the SCF loop).
    * - ``conv_tol``
      - ``1e-6``
      - Convergence threshold for the density residual
        :math:`\|\rho_\mathrm{out} - \rho_\mathrm{in}\|_2`.
    * - ``mixing_scheme``
      - ``"linear"``
      - ``"linear"`` or ``"anderson"`` (``"diis"``/``"pulay"`` are accepted as aliases).
    * - ``alpha_mix``
      - ``0.6``
      - Mixing parameter.
    * - ``mixing_history``
      - ``5``
      - Number of previous iterations used by Anderson mixing (alias: ``diis_history``).
    * - ``theory_level``
      - ``"non-relativistic"``
      - All-electron only: ``"non-relativistic"``, ``"scalar-relativistic"``
        (Koelling-Harmon) or ``"zora"``. The relativistic options are approximate and
        not benchmarked.
    * - ``relativistic_scf``
      - ``true``
      - Relativistic theory levels: iterate the relativistic equations to self-consistency
        after a non-relativistic SCF. If ``false``, the relativistic eigenvalues are
        computed with the non-relativistic density.

``electrons``
-------------

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Key
      - Default
      - Description
    * - ``configuration``
      - ``["1s1"]``
      - Occupied shells, e.g. ``["1s2", "2s2", "2p6"]``. Fractional occupations are allowed.
    * - ``Z``
      - ``0``
      - Nuclear charge; ``0`` takes the atomic number of ``element``.

``confinement``
---------------

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Key
      - Default
      - Description
    * - ``type``
      - ``"none"``
      - ``"none"``, ``"softstep"`` or ``"harmonic"`` (see :doc:`method`).
    * - ``rc``
      - ``20.0``
      - Confinement radius :math:`r_c`.
    * - ``ri_factor``
      - ``0.9``
      - Onset of the confinement, :math:`r_i = \mathrm{ri\_factor} \cdot r_c`.
    * - ``Vbarrier``
      - ``1.0``
      - Height of the soft-step barrier (Hartree).
    * - ``polarization_mode``
      - ``null``
      - ``"softcoul"``: compute the polarization orbitals in a soft-Coulomb potential.
    * - ``softcoul_charge``
      - ``1.0``
      - Charge :math:`Q` of the soft-Coulomb potential (set by the ``optimize`` task).
    * - ``softcoul_delta``
      - ``0.1``
      - Softening length :math:`\delta`.

``output``
----------

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Key
      - Default
      - Description
    * - ``output_wfc_qe``
      - ``false``
      - Write the orbitals on a logarithmic grid (``*_qe.dat``).
    * - ``qe_num_points``
      - ``1001``
      - Number of points of the logarithmic grid.
    * - ``qe_rmin``
      - ``1e-8``
      - Smallest radius of the logarithmic grid.
    * - ``output_wfc_hdf5``
      - ``false``
      - Write the orbitals on the FE-DVR grid (``*_wfc.h5``).
    * - ``output_wfc_bessel``
      - ``false``
      - Write the spherical Bessel transform of the orbitals (``*_bessel.h5``).
    * - ``bessel_qmax``
      - ``50.0``
      - Largest wavenumber :math:`q` (1/bohr).
    * - ``bessel_nq``
      - ``201``
      - Number of :math:`q` points.
    * - ``bessel_rpow``
      - ``1``
      - Power :math:`p` in :math:`\int r^p u(r) j_l(qr)\,dr`.
    * - ``bessel_quad_method``
      - ``"simpson"``
      - Quadrature per element: ``"simpson"`` or ``"lobatto"``.
    * - ``bessel_quad_npoints``
      - ``41``
      - Quadrature points per element.
    * - ``output_dipole_moments``
      - ``false``
      - Write the dipole matrix elements between the orbitals.
    * - ``output_wfc_pyimpurity``
      - ``false``
      - All-electron only: write the bound orbitals to ``<element>_wfc_ae.h5``.

``photoemission``
-----------------

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Key
      - Default
      - Description
    * - ``energy_min``, ``energy_max``
      - ``0.1``, ``3.0``
      - Range of photoelectron kinetic energies (Hartree).
    * - ``n_energies``
      - ``50``
      - Number of energies.
    * - ``gauges``
      - ``["length"]``
      - ``"length"`` and/or ``"velocity"``.
    * - ``lmax_scatter``
      - ``null``
      - Highest angular momentum of the final states (default: all allowed by the dipole
        selection rule).
    * - ``store_wavefunctions``
      - ``false``
      - Also write the continuum wavefunctions.
    * - ``output_prefix``
      - ``"photoemission"``
      - Results are written to ``<storage_dir>/<output_prefix>.h5``.

.. _output-files:

Output files
------------

``<element>_density_potential.h5`` (in ``storage_dir``)
    ``grid``, density ``rho`` (:math:`\int \rho(r) r^2 dr = N`) and effective potential
    ``Veff`` (Hartree).

The ``nscf`` export (``pseudoatomic ... EXPORT_DIR``) names files by a basis tag:
``SZ``, ``DZ``, ``TZ``, ... for ``nmax + 1`` radial functions per channel, one ``P`` per
polarization channel, and ``_rc<rc>`` if a confinement is used (e.g. ``Si_DZP_rc20.0``).
All orbitals are reduced radial functions :math:`u_{nl}(r) = r R_{nl}(r)`, normalized to
:math:`\int u^2 dr = 1`.

``<element>_<tag>_eigenvalues.dat``
    Columns ``l``, ``n``, eigenvalue (Hartree); the tag omits the confinement.
``<element>_<tag>_qe.dat``
    First line: number of radial points and of orbitals; second line: ``l`` of each
    orbital; then one line per point with :math:`\ln r`, :math:`r` and :math:`u(r)` of all
    orbitals.
``<element>_<tag>_wfc.h5``
    Element boundaries ``xp``, attributes ``ne``, ``ng``, ``lmax``, ``nmax``, and the
    orbitals ``wf[l, n, i]`` on the FE-DVR grid points.
``<element>_<tag>_bessel.h5``
    ``qgrid`` and ``wf_bessel[l, n, q]`` :math:`= \int r^p u_{nl}(r) j_l(qr)\,dr`.
``<element>_dipole_moments.h5``
    ``Indices`` (``l``, ``n``, ``m``) of the orbitals in the complex spherical-harmonic
    basis and the dipole matrix elements ``real_part``/``imag_part`` with shape
    ``(3, norb, norb)`` (Cartesian components x, y, z).
