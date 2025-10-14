import json
import os
from time import perf_counter

import numpy as np
from scipy.interpolate import UnivariateSpline

from atomic_femdvr.adaptive_elements import OptimizeElements
from atomic_femdvr.SchrodingerSolver import SolveNR, SolveSR, SolveZORA
from atomic_femdvr.utils import PrintTime

from pathlib import Path
from typing import Literal
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, FilePath, ConfigDict, field_validator, create_model, DirectoryPath
from enum import Enum

class EnergyUnit(str, Enum):
    RYDBERG = "Rydberg"
    EV = "eV"


class BaseModel(PydanticBaseModel):

    model_config = ConfigDict(extra="forbid")

class SysParamsInput(BaseModel):
    file_pot: FilePath
    file_upf: FilePath | None = None
    file_vhx: FilePath | None = None
    pot_columns: tuple[int, int] = Field(default=(0, 4))
    pot_energy_unit: Literal["Ry", "eV", "Ha"] = Field(default="Rydberg")
    lmax: int = Field(default=0, ge=0)
    nmax: int = Field(default=4, ge=1)

    @field_validator("pot_energy_unit", mode="before")
    @classmethod
    def map_assume_isolated(cls, v: str) -> str:
        """Map equivalent values for assume_isolated to the same string so that comparisons work as expected."""
        if v.lower() in ["ry", "rydberg"]:
            return "Ry"
        elif v.lower() in ["ev"]:
            return "eV"
        elif v.lower() in ["ha", "hartree"]:
            return "Ha"
        else:
            raise ValueError(f"Invalid value for pot_energy_unit: {v}")

class SolverInput(BaseModel):
    method: Literal["non-relativistic", "zora", "scalar-relativistic"] = Field(default="non-relativistic")
    h_min: float
    h_max: float
    Rmax: float | None = Field(default=None, gt=0)
    tol: float = Field(default=1.0e-3, gt=0)
    ng: int = Field(default=8, ge=1)
    elem_tol: float = Field(default=1.0e-2, gt=0)

def solver_input_factory(default_hmin: float, default_hmax: float) -> type[SolverInput]:
    model = create_model(
        "SolverInput",
        __base__ = SolverInput,
        h_min = (float, Field(default=default_hmin, gt=0)),
        h_max = (float, Field(default=default_hmax, gt=0)),
    )
    assert issubclass(model, SolverInput)
    return model

class DFTInput(BaseModel):
    driver: str = "internal"
    xc_functional: str = "PBE"
    x_functional: str | None = "gga_x_pbe"
    c_functional: str | None = "gga_c_pbe"
    alpha: float = 0.0
    alpha_x: float = 1.0
    max_iter: int = 100
    conv_tol: float = 1.0e-6

class PseudoConfigInput(BaseModel):
    upflib_dir: DirectoryPath = Field(default=Path())
    lib_ext: str = "so"
    storage_dir: Path


class ConfinementInput(BaseModel):
    type: str
    rc: float
    ri_factor: float
    Vbarrier: float
    polarization_mode: Literal[None, "softcoul"] = None
    softcoul_delta: float = 0.1

class ProjectorInput(BaseModel):
    pass
