"""Fixtures providing input parameters for test calculations on different elements."""

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def molybdenum_input_dict(data_directory: Path, tmp_path) -> dict[str, dict[str, Any]]:
    """Fixture providing input parameters for a calculation on molybdenum."""
    return {
        "control": {"storage_dir": str(tmp_path / "Mo_Pseudo")},
        "sysparams": {
            "file_upf": data_directory / "Mo/Mo.upf",
            "element": "Mo",
            "lmax": 3,
            "nmax": 2,
        },
        "solver": {"h_min": 0.5, "h_max": 4.0, "Rmax": 30.0, "elem_tol": 1.0e-2, "ng": 8},
        "dft": {
            "driver": "internal",
            "x_functional": "GGA_X_PBE",
            "c_functional": "GGA_C_PBE",
            "alpha_x": 1.0,
            "max_iter": 100,
            "conv_tol": 1.0e-6,
        },
        "confinement": {
            "type": "SoftStep",
            "rc": 10.0,
            "ri_factor": 0.5,
            "Vbarrier": 1.0,
            "polarization_mode": "SoftCoul",
            "softcoul_delta": 0.1,
        },
        "output": {"qe_num_points": 1001, "qe_rmin": 1.0e-8},
    }


@pytest.fixture
def silicon_input_dict(data_directory: Path) -> dict[str, dict[str, Any]]:
    """Fixture providing input parameters for a calculation on silicon."""
    return {
        "pseudo_config": {},
        "sysparams": {
            "file_upf": data_directory / "Si/Si.oncvpsp.upf",
            "file_pot": data_directory / "Si/Si_Vpsloc.plot",
            "file_vhx": data_directory / "Si/Si_Vhxc.plot",
            "pot_columns": [0, 1],
            "pot_energy_unit": "Hartree",
            "lmax": 2,
            "nmax": 2,
        },
        "solver": {"h_min": 0.5, "h_max": 5.0, "Rmax": 30.0, "tol": 1.0e-2, "ng": 16},
        "dft": {
            "driver": "pylibxc",
            "xc_functional": "",
            "x_functional": "GGA_X_PBE",
            "c_functional": "GGA_C_PBE",
            "alpha_x": 1.0,
            "max_iter": 100,
            "conv_tol": 1.0e-6,
        },
        "confinement": {"type": "SoftStep", "rc": 20.0, "ri_factor": 0.5, "Vbarrier": 1.0},
    }
