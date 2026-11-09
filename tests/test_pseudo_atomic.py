"""Tests for the `atomic_femdvr.pseudo_atomic` module.

Benchmark: a norm-conserving pseudopotential reproduces the all-electron valence
eigenvalues in its reference configuration. ONCVPSP records these all-electron
eigenvalues as ``ep`` (Hartree, 5 decimals) in the PP_INPUTFILE section of the UPF file.
Grid-converged results agree to about 2e-5 Ha (0.5 meV); the remainder comes from
interpolating the UPF data and from differences between the PBE implementations.
"""

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
