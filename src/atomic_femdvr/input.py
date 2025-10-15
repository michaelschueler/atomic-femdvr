


from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, DirectoryPath, Field, FilePath, create_model, field_validator


class EnergyUnit(str, Enum):
    RYDBERG = "Ry"
    ELECTRONVOLTS = "eV"
    HARTREE = "Ha"


class BaseModel(PydanticBaseModel):

    model_config = ConfigDict(extra="forbid")

class SysParamsInput(BaseModel):
    file_pot: FilePath
    file_upf: FilePath | None = None
    file_vhx: FilePath | None = None
    pot_columns: tuple[int, int] = Field(default=(0, 4))
    pot_energy_unit: EnergyUnit = Field(default=EnergyUnit.RYDBERG)
    lmax: int = Field(default=0, ge=0)
    nmax: int = Field(default=4, ge=1)
    element: str | None = None

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
    storage_dir: Path = Field(default=Path())


class ConfinementType(str, Enum):
    SOFTSTEP = "softstep"
    HARMONIC = "harmonic"
    NONE = None

class ConfinementInput(BaseModel):
    type: ConfinementType = ConfinementType.NONE
    rc: float = 20.0
    ri_factor: float = 0.9
    Vbarrier: float = 1.0
    polarization_mode: Literal[None, "softcoul"] = None
    softcoul_delta: float = 0.1
    softcoul_charge: float = 1.0

    @field_validator("type", mode="before")
    @classmethod
    def make_lower(cls, v: str) -> str:
        """Convert to lower case to make the input case-insensitive."""
        return v.lower()

class ProjectorInput(BaseModel):
    nr: int = 1001
    rmin: float = 1.0e-8
    pass
