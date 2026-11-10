Usage
=====

Calculations are defined by a JSON input file (see :doc:`input`) and run with the
``atomic_femdvr`` command (see :doc:`cli`). All energies are in Hartree and all lengths
in bohr. Example inputs are in the ``examples/`` directory of the repository.

Pseudo-atoms
------------

A pseudo-atom calculation needs a norm-conserving pseudopotential in UPF (v2) format.
The example ``examples/Si_Pseudo.json`` uses the silicon pseudopotential shipped with the
tests:

.. code-block:: json

    {
        "control":     {"storage_dir": "./Si_pseudo"},
        "sysparams":   {"element": "Si", "file_upf": "../tests/data/Si/Si.oncvpsp.upf",
                        "lmax": 2, "nmax": 1},
        "solver":      {"h_min": 0.5, "h_max": 4.0, "Rmax": 30.0, "ng": 8},
        "dft":         {"driver": "internal", "xc_functional": "PBE",
                        "mixing_scheme": "anderson", "alpha_mix": 0.5, "conv_tol": 1.0e-8},
        "confinement": {"type": "SoftStep", "rc": 20.0, "ri_factor": 0.5, "Vbarrier": 1.0,
                        "polarization_mode": "SoftCoul", "softcoul_delta": 0.1},
        "output":      {"output_wfc_qe": true, "output_wfc_bessel": true}
    }

Run it from the ``examples/`` directory:

.. code-block:: console

    $ atomic_femdvr pseudoatomic Si_Pseudo.json -t scf,nscf out

The tasks are executed in this order:

``scf``
    Solves the Kohn-Sham equations self-consistently for the valence configuration of the
    pseudopotential (the occupations of its pseudo-atomic orbitals). The density and
    potential are saved to ``storage_dir`` and reused by later runs.
``optimize``
    Optimizes the soft-Coulomb charge for the polarization orbitals (see :doc:`method`).
``nscf``
    Solves for ``nmax + 1`` states in each channel ``l = 0 ... lmax`` in the self-consistent
    potential plus the confinement potential, and prints the energy shift of the valence
    states caused by the confinement.

If an export directory is given (``out`` above), the ``nscf`` orbitals are written there
in the formats selected in the ``output`` section (see :ref:`output-files`).

The example asks for ``lmax = 2`` and ``nmax = 1``: two radial functions ("double zeta")
for each of s, p and d. The d channel is not occupied in silicon; it is a polarization
channel. The exported files are tagged accordingly, e.g. ``Si_DZP_rc20.0_qe.dat``.

Pseudopotentials without pseudo-atomic orbitals
    Some pseudopotential libraries, e.g. SG15, contain no pseudo-atomic orbitals
    (``PP_PSWFC``). The occupations are then unknown and ``scf`` is not possible. Instead,
    ``nscf`` uses the atomic density stored in ``PP_RHOATOM``:

    .. code-block:: console

        $ atomic_femdvr pseudoatomic Si_sg15.json -t nscf out

All-electron atoms
------------------

The all-electron solver needs the electronic configuration instead of a pseudopotential
(``examples/Ne_full.json``):

.. code-block:: json

    {
        "control":   {"storage_dir": "./atomic_data"},
        "sysparams": {"element": "Ne"},
        "solver":    {"h_min": 0.25, "h_max": 10.0, "Rmax": 100.0, "ng": 12,
                      "eigensolver": "banded"},
        "dft":       {"driver": "pylibxc", "xc_functional": "",
                      "x_functional": "lda_x", "c_functional": "lda_c_vwn",
                      "mixing_scheme": "anderson", "alpha_mix": 0.8, "conv_tol": 1.0e-8},
        "electrons": {"configuration": ["1s2", "2s2", "2p6"]}
    }

.. code-block:: console

    $ atomic_femdvr atomic Ne_full.json -t scf

Setting ``"theory_level"`` in the ``dft`` section to ``"scalar-relativistic"`` or
``"zora"`` switches to an approximate relativistic treatment; these have not been
benchmarked and print a warning.

Photoemission
-------------

After an ``scf`` run, ``atomic_femdvr photoemission`` (all-electron) or
``atomic_femdvr pseudophotoemission`` (pseudo-atom) compute scattering phase shifts and
radial dipole matrix elements between the occupied orbitals and continuum states over a
range of kinetic energies. Both read the same input file with an additional
``photoemission`` section (see :doc:`input`; example ``examples/Si_photo.json``).

Python
------

The same calculations can be run from Python; the drivers return the eigenvalues
(Hartree) per task, keyed by the angular momentum:

.. code-block:: python

    import json
    from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, solve_pseudo_atomic

    with open("Si_Pseudo.json") as f:
        inp = PseudoAtomicInput(**json.load(f))

    eigenvalues = solve_pseudo_atomic(inp, ("scf", "nscf"), export_dir="out")
    print(eigenvalues["scf"]["0"])  # [-0.397365...]
