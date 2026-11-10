"""Tests for the `atomic_femdvr.pseudo_atomic` module.

Benchmark: a norm-conserving pseudopotential reproduces the all-electron valence
eigenvalues in its reference configuration. ONCVPSP records these all-electron
eigenvalues as ``ep`` (Hartree, 5 decimals) in the PP_INPUTFILE section of the UPF file.
Grid-converged results agree to about 2e-5 Ha (0.5 meV); the remainder comes from
interpolating the UPF data and from differences between the PBE implementations.
"""

import re
from pathlib import Path

import pytest

from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, solve_pseudo_atomic

# ONCVPSP all-electron reference eigenvalues (Hartree) from the UPF headers, keyed by l.
ONCVPSP_REFERENCE = {
    "molybdenum_input_dict": {"0": -2.36471, "1": -1.41432, "2": -0.13792},
    "silicon_input_dict": {"0": -0.39736, "1": -0.14998},
}


@pytest.mark.parametrize("system", list(ONCVPSP_REFERENCE))
def test_scf_matches_oncvpsp(system: str, request: pytest.FixtureRequest) -> None:
    """Lowest SCF eigenvalue per l agrees with the ONCVPSP all-electron reference."""
    inp = PseudoAtomicInput(**request.getfixturevalue(system))

    eigenvalues = solve_pseudo_atomic(inp, task_list=('scf',))['scf']

    for l, reference in ONCVPSP_REFERENCE[system].items():
        assert eigenvalues[l][0] == pytest.approx(reference, abs=3.0e-5)


def test_nscf_with_confinement(molybdenum_input_dict: dict) -> None:
    """Confinement shifts the occupied states up only slightly and adds polarization (f) states."""
    inp = PseudoAtomicInput(**molybdenum_input_dict)

    eigenvalues = solve_pseudo_atomic(inp, task_list=('scf,nscf',))

    assert set(eigenvalues['nscf']) == {"0", "1", "2", "3"}
    for l in ("0", "1", "2"):
        shift = eigenvalues['nscf'][l][0] - eigenvalues['scf'][l][0]
        assert 0.0 <= shift < 1.0e-3


def test_nscf_without_confinement(silicon_input_dict: dict) -> None:
    """Without confinement, nscf reproduces the bound SCF states."""
    inp = PseudoAtomicInput(**silicon_input_dict)

    eigenvalues = solve_pseudo_atomic(inp, task_list=('scf', 'nscf'))

    for l, eps in eigenvalues['scf'].items():
        assert eigenvalues['nscf'][l][0] == pytest.approx(eps[0], abs=1.0e-8)


def test_restart(silicon_input_dict: dict) -> None:
    """An nscf run restarts from the density saved by a previous scf run."""
    inp = PseudoAtomicInput(**silicon_input_dict)
    scf = solve_pseudo_atomic(inp, task_list=('scf',))['scf']

    nscf = solve_pseudo_atomic(PseudoAtomicInput(**silicon_input_dict), task_list=('nscf',))['nscf']

    assert nscf["0"][0] == pytest.approx(scf["0"][0], abs=1.0e-8)


def test_unknown_task(silicon_input_dict: dict) -> None:
    """Misspelled tasks are rejected instead of silently ignored."""
    inp = PseudoAtomicInput(**silicon_input_dict)
    with pytest.raises(ValueError, match="Unknown task"):
        solve_pseudo_atomic(inp, task_list=('scf,nsfc',))


def _strip_paos(upf: Path, out: Path, empty_block: bool) -> Path:
    """Write a copy of a UPF file without pseudo-atomic orbitals (PP_PSWFC)."""
    text = re.sub(r'number_of_wfc="\s*\d+"', 'number_of_wfc="0"', upf.read_text())
    replacement = "<PP_PSWFC>\n</PP_PSWFC>" if empty_block else ""
    out.write_text(re.sub(r"<PP_PSWFC>.*?</PP_PSWFC>", replacement, text, flags=re.S))
    return out


@pytest.mark.parametrize("empty_block", [False, True])
def test_nscf_without_paos(silicon_input_dict: dict, tmp_path: Path, empty_block: bool) -> None:
    """Without PP_PSWFC, nscf uses PP_RHOATOM and reproduces the SCF results."""
    reference = solve_pseudo_atomic(PseudoAtomicInput(**silicon_input_dict), ('scf', 'nscf'))['nscf']

    upf = _strip_paos(silicon_input_dict["sysparams"]["file_upf"], tmp_path / "Si_nopao.upf", empty_block)
    silicon_input_dict["sysparams"]["file_upf"] = upf
    silicon_input_dict["control"]["storage_dir"] = tmp_path / "nopao"
    export_dir = tmp_path / "export"
    eigenvalues = solve_pseudo_atomic(PseudoAtomicInput(**silicon_input_dict), ('nscf',),
                                      export_dir=str(export_dir))['nscf']

    for l, eps in reference.items():
        assert eigenvalues[l] == pytest.approx(eps, abs=1.0e-5)
    assert (export_dir / "Si_DZP_eigenvalues.dat").exists()


def test_scf_without_paos(silicon_input_dict: dict, tmp_path: Path) -> None:
    """SCF needs the occupations of the pseudo-atomic orbitals."""
    silicon_input_dict["sysparams"]["file_upf"] = _strip_paos(
        silicon_input_dict["sysparams"]["file_upf"], tmp_path / "Si_nopao.upf", empty_block=False)
    with pytest.raises(ValueError, match="PP_PSWFC"):
        solve_pseudo_atomic(PseudoAtomicInput(**silicon_input_dict), ('scf',))
