<h1 align="center">
  atomic-femdvr
</h1>

<p align="center">
    <a href="https://github.com/michaelschueler/atomic-femdvr/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/michaelschueler/atomic-femdvr/actions/workflows/tests.yml/badge.svg" /></a>
    <a href="https://pypi.org/project/atomic_femdvr">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/atomic_femdvr" /></a>
    <a href="https://pypi.org/project/atomic_femdvr">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/atomic_femdvr" /></a>
    <a href="https://github.com/michaelschueler/atomic-femdvr/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/pypi/l/atomic_femdvr" /></a>
    <a href="https://doi.org/10.24435/materialscloud:xt-zm">
        <img alt="DOI" src="https://img.shields.io/badge/DOI-10.24435%2Fmaterialscloud%3Axt--zm-blue" /></a>
    <a href="https://atomic-femdvr.readthedocs.io/en/latest/?badge=latest">
        <img alt="Documentation Status" src="https://readthedocs.org/projects/atomic-femdvr/badge/?version=latest" /></a>
</p>

Kohn-Sham solver for isolated atoms using the finite-element discrete variable
representation (FE-DVR).

- **Pseudo-atoms** with norm-conserving pseudopotentials in UPF format, including
  pseudopotentials without pseudo-atomic orbitals (e.g. SG15)
- **All-electron atoms** (non-relativistic), plus approximate scalar-relativistic
  (Koelling-Harmon) and ZORA treatments
- LDA and GGA functionals: PBE built in, all others via
  [libxc](https://libxc.gitlab.io)
- **Confined atomic orbitals** with polarization functions for use as localized
  basis functions or projectors, exported on a logarithmic grid (Quantum ESPRESSO
  style), on the FE-DVR grid, or as spherical Bessel transforms; dipole matrix
  elements
- **Photoemission**: scattering phase shifts and dipole matrix elements to
  continuum states

## 🚀 Installation

```console
$ python3 -m pip install atomic_femdvr
```

The built-in exchange-correlation driver provides PBE. Other functionals (e.g. the
LDA used for the NIST benchmark) need the libxc Python bindings, which are not on
PyPI; see the
[installation guide](https://atomic-femdvr.readthedocs.io/en/latest/installation.html).

## 💪 Getting started

Calculations are defined by a JSON input file. A pseudo-atom with a confinement
potential and polarization orbitals (`examples/Si_Pseudo.json`):

```json
{
  "control": { "storage_dir": "./Si_pseudo" },
  "sysparams": {
    "element": "Si",
    "file_upf": "../tests/data/Si/Si.oncvpsp.upf",
    "lmax": 2,
    "nmax": 1
  },
  "dft": { "xc_functional": "PBE", "mixing_scheme": "anderson", "alpha_mix": 0.5 },
  "confinement": {
    "type": "SoftStep",
    "rc": 20.0,
    "ri_factor": 0.5,
    "polarization_mode": "SoftCoul"
  },
  "output": { "output_wfc_qe": true, "output_wfc_bessel": true }
}
```

```console
$ cd examples
$ atomic_femdvr pseudoatomic Si_Pseudo.json -t scf,nscf out
```

This solves the Kohn-Sham equations self-consistently (`scf`), computes two
confined radial functions each for s, p and d (`nscf`), and writes them to `out/`.
All energies are in Hartree and all lengths in bohr. The all-electron solver is
run with `atomic_femdvr atomic` (see `examples/Ne_full.json`).

See the [documentation](https://atomic-femdvr.readthedocs.io) for the input
reference, the method and the output formats.

## ✅ Validation

- **All-electron**: LDA eigenvalues agree with the
  [NIST atomic reference data](https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations)
  (Ne, to 2e-6 Ha).
- **Pseudo-atoms**: eigenvalues reproduce the all-electron reference energies
  stored by ONCVPSP in the UPF file to about 0.5 meV (Mo, Si).

The scalar-relativistic and ZORA options are approximate and have not been
benchmarked. Details are in the
[documentation](https://atomic-femdvr.readthedocs.io/en/latest/validation.html).

## 📖 Citation

If you use atomic-femdvr in your work, please cite the archived release:

> M. Schüler, _atomic-femdvr_, Materials Cloud Archive (2026),
> [doi:10.24435/materialscloud:xt-zm](https://doi.org/10.24435/materialscloud:xt-zm)

```bibtex
@software{atomic_femdvr,
  author    = {Schüler, Michael},
  title     = {atomic-femdvr},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Materials Cloud Archive},
  doi       = {10.24435/materialscloud:xt-zm},
  url       = {https://github.com/michaelschueler/atomic-femdvr}
}
```

See also [CITATION.cff](CITATION.cff).

## ⚖️ License

The code is licensed under the MIT License. The test pseudopotentials in
`tests/data` are taken from [PseudoDojo](http://www.pseudo-dojo.org) (M. J. van
Setten et al., Comput. Phys. Commun. **226**, 39 (2018)) and distributed under
CC BY 4.0; see [tests/data/README.md](tests/data/README.md).

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are
appreciated. See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for more
information.

### Development installation

```console
$ git clone https://github.com/michaelschueler/atomic-femdvr.git
$ cd atomic-femdvr
$ python3 -m pip install -e .
$ python3 -m pytest tests
```

The NIST benchmark test is skipped unless `pylibxc` is installed. With
[tox](https://tox.wiki) installed (`pip install tox tox-uv`), `tox -e py` runs
the tests, `tox -e lint` the linter and `tox -e docs` builds the documentation in
`docs/build/html`.

### Making a release

`tox -e finish` removes the `-dev` suffix from the version, builds and uploads
the package to PyPI, tags the release, and bumps the version. It needs a PyPI
token in the keyring (`keyring set https://upload.pypi.org/legacy/ __token__`).
After pushing, create a release on GitHub from the new tag.

### 🍪 Template

This package was created from
[@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack)
template.
