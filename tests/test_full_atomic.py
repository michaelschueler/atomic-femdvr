"""Tests for the all-electron solver in :mod:`atomic_femdvr.full_atomic`."""

import pytest

from atomic_femdvr.full_atom_dft import FullAtomDFT
from atomic_femdvr.full_atomic import FullAtomicInput, solve_atomic


@pytest.fixture
def neon_input_dict(tmp_path) -> dict:
    """Neon all-electron SCF input (no UPF; defines its own configuration)."""
    return {
        "control": {"storage_dir": str(tmp_path / "atomic_data_neon")},
        "sysparams": {"element": "Ne"},
        "solver": {
            "h_min": 0.25,
            "h_max": 10.0,
            "Rmax": 100.0,
            "elem_tol": 1.0e-2,
            "ng": 12,
            "eigensolver": "banded",
        },
        "dft": {
            "driver": "internal",
            "xc_functional": "PBE",
            "x_functional": "lda_x",
            "c_functional": "lda_c_pz",
            "mixing_scheme": "anderson",
            "alpha_mix": 0.8,
            "alpha_x": 1.0,
            "max_iter": 100,
            "conv_tol": 1.0e-8,
            "diis_history": 3,
        },
        "electrons": {"Z": 10, "configuration": ["1s2", "2s2", "2p6"]},
    }


def test_neon_scf(neon_input_dict, num_regression):
    """All-electron SCF on neon: regression-test bound-state eigenvalues."""
    inp = FullAtomicInput(**neon_input_dict)

    # Capture the eigenvalues directly from FullAtomDFT, since solve_atomic
    # currently returns None.
    atom = FullAtomDFT(inp.control, inp.sysparams, inp.electrons, inp.solver, inp.dft)
    atom.initialize_density()
    atom.ks_self_consistency(theory_level="non-relativistic")
    eigenvalues, _psi = atom.get_bound_states(theory_level="non-relativistic")

    flat = {f"l{l}": vals for l, vals in eigenvalues.items()}
    num_regression.check(flat, default_tolerance={"rtol": 1e-7})


def test_solve_atomic_smoke(neon_input_dict):
    """``solve_atomic`` runs end-to-end on neon without raising."""
    inp = FullAtomicInput(**neon_input_dict)
    # Tighter assertions live in test_neon_scf; here we just verify no exception.
    solve_atomic(inp, task_list=("scf",))
