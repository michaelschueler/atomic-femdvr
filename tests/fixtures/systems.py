"""Fixtures providing input parameters for test calculations on different elements."""

from pathlib import Path
from typing import Any

import pytest

PSEUDO_SOLVER = {"h_min": 0.5, "h_max": 4.0, "Rmax": 30.0, "ng": 8}
PSEUDO_DFT = {"driver": "internal", "xc_functional": "PBE", "max_iter": 200,
              "conv_tol": 1.0e-8, "alpha_mix": 0.4}


@pytest.fixture
def molybdenum_input_dict(data_directory: Path, tmp_path: Path) -> dict[str, dict[str, Any]]:
    """Fixture providing input parameters for a calculation on molybdenum (ONCVPSP, PBE)."""
    return {
        "control": {"storage_dir": tmp_path},
        "sysparams": {"file_upf": data_directory / "Mo/Mo.upf", "element": "Mo",
                      "lmax": 3, "nmax": 2},
        "solver": PSEUDO_SOLVER,
        "dft": PSEUDO_DFT,
        "confinement": {"type": "SoftStep", "rc": 10.0, "ri_factor": 0.5, "Vbarrier": 1.0,
                        "polarization_mode": "SoftCoul", "softcoul_delta": 0.1},
    }

@pytest.fixture
def silicon_input_dict(data_directory: Path, tmp_path: Path) -> dict[str, dict[str, Any]]:
    """Fixture providing input parameters for a calculation on silicon (ONCVPSP, PBE)."""
    return {
        "control": {"storage_dir": tmp_path},
        "sysparams": {"file_upf": data_directory / "Si/Si.oncvpsp.upf", "element": "Si",
                      "lmax": 2, "nmax": 1},
        "solver": PSEUDO_SOLVER,
        "dft": PSEUDO_DFT,
    }
