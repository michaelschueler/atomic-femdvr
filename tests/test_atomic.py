"""Tests for the all-electron atomic solver (full_atomic)."""

from pathlib import Path

import pytest

from atomic_femdvr.full_atomic import FullAtomicInput, solve_atomic

AE_SOLVER = {"h_min": 0.25, "h_max": 10.0, "Rmax": 60.0, "ng": 12, "eigensolver": "banded"}


def _ae_input(tmp_path: Path, element: str, configuration: list[str], dft: dict,
              solver: dict = AE_SOLVER) -> FullAtomicInput:
    return FullAtomicInput(
        control={"storage_dir": tmp_path},
        sysparams={"element": element},
        solver=solver,
        dft={"max_iter": 200, "conv_tol": 1.0e-9, "mixing_scheme": "anderson", "alpha_mix": 0.5,
             **dft},
        electrons={"configuration": configuration},
    )


def test_neon_lda_nist(tmp_path: Path) -> None:
    """Ne LDA eigenvalues agree with the NIST atomic reference data (Kotochigova et al.)."""
    pytest.importorskip("pylibxc")
    inp = _ae_input(tmp_path, "Ne", ["1s2", "2s2", "2p6"],
                    {"driver": "pylibxc", "xc_functional": "",
                     "x_functional": "lda_x", "c_functional": "lda_c_vwn"},
                    solver={**AE_SOLVER, "Rmax": 100.0, "elem_tol": 1.0e-2})

    eigenvalues = solve_atomic(inp, ("scf",))["scf"]

    nist = {"0": [-30.305855, -1.322637], "1": [-0.498034]}
    for l, reference in nist.items():
        assert eigenvalues[l] == pytest.approx(reference, abs=2.0e-6)


@pytest.mark.parametrize("mixing_scheme", ["linear", "anderson"])
def test_silicon_pbe_regression(tmp_path: Path, mixing_scheme: str) -> None:
    """Si PBE eigenvalues with the internal functional (regression values, not a benchmark)."""
    inp = _ae_input(tmp_path, "Si", ["1s2", "2s2", "2p6", "3s2", "3p2"],
                    {"driver": "internal", "xc_functional": "PBE",
                     "mixing_scheme": mixing_scheme, "alpha_mix": 0.3 if mixing_scheme == "linear" else 0.5})

    eigenvalues = solve_atomic(inp, ("scf",))["scf"]

    reference = {"0": [-65.457499040, -5.102404871, -0.395730545],
                 "1": [-3.512898918, -0.150316961]}
    for l, ref in reference.items():
        assert eigenvalues[l] == pytest.approx(ref, abs=1.0e-7)
