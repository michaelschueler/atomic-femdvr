

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, FilePath, create_model, field_validator


class EnergyUnit(str, Enum):
    RYDBERG = "Ry"
    ELECTRONVOLTS = "eV"
    HARTREE = "Ha"


class BaseModel(PydanticBaseModel):

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class SysParamsInput(BaseModel):
    file_pot: FilePath | None = None
    file_upf: FilePath | None = None
    file_vhx: FilePath | None = None
    file_rho: FilePath | None = None
    pot_columns: tuple[int, int] = Field(default=(0, 4))
    pot_energy_unit: EnergyUnit = Field(default=EnergyUnit.RYDBERG)
    lmax: int = Field(default=0, ge=0)
    nmax: int = Field(default=4, ge=0)
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
        
    @field_validator("element", mode="before")
    @classmethod
    def validate_element(cls, v: str | None) -> None:
        if v is None:
            raise ValueError("Element must be specified in sysparams.")
        if len(v.strip()) == 0:
            raise ValueError("Element must be a non-empty string.")
        if len(v.strip()) > 2:
            raise ValueError("Element symbol must be one or two characters.")
        return v.strip().capitalize()

class ControlInput(BaseModel):
    storage_dir: Path = Field(default=Path())
    restart: bool = Field(default=False)
    store_density: bool = Field(default=True)

class OutputInput(BaseModel):
    output_wfc_qe : bool = Field(default=False)
    output_wfc_hdf5 : bool = Field(default=False)
    output_wfc_bessel : bool = Field(default=False)
    output_dipole_moments : bool = Field(default=False)
    bessel_quad_npoints : int = Field(default=41, ge=3)
    bessel_quad_method : Literal['simpson', 'lobatto'] = Field(default='simpson')
    bessel_qmax : float = Field(default=50.0, gt=0.0)
    bessel_rpow : int = Field(default=1, ge=1)
    bessel_nq: int = Field(default=201, ge=3)
    qe_num_points : int = Field(default=1001, ge=3)
    qe_rmin : float = Field(default=1.0e-8, gt=0.0)

    @field_validator("bessel_quad_method", mode="before")
    @classmethod
    def make_lower(cls, v: str) -> str:
        """Convert to lower case to make the input case-insensitive."""
        return v.lower()

class ElectronsInput(BaseModel):
    Z: float = Field(default=0.0, ge=0)
    configuration: list[str] = Field(default_factory=lambda: ["1s1"])

class SolverInput(BaseModel):
    theory_level: Literal["non-relativistic", "zora", "scalar-relativistic"] = Field(default="non-relativistic")
    eigensolver: Literal["full", "banded"] = Field(default="full")
    h_min: float = Field(default=0.5, gt=0)
    h_max: float = Field(default=4.0, gt=0)
    Rmax: float | None = Field(default=None, gt=0)
    tol: float = Field(default=1.0e-3, gt=0)
    ng: int = Field(default=8, ge=1)
    elem_tol: float = Field(default=1.0e-2, gt=0)

    @field_validator("theory_level", "eigensolver", mode="before")
    @classmethod
    def make_lower(cls, v: str) -> str:
        """Convert to lower case to make the input case-insensitive."""
        return v.lower()

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
    mixing_scheme: Literal["linear", "diis", "anderson"] = Field(default="linear")
    diis_history: int = Field(default=5, ge=2)
    alpha_mix: float = 0.6
    alpha_x: float = 1.0
    max_iter: int = 100
    conv_tol: float = 1.0e-6

    @field_validator("mixing_scheme", mode="before")
    @classmethod
    def make_lower(cls, v: str) -> str:
        """Convert to lower case to make the input case-insensitive."""
        return v.lower()

class PseudoConfigInput(BaseModel):
    storage_dir: Path = Field(default=Path())


class ConfinementType(str, Enum):
    SOFTSTEP = "softstep"
    HARMONIC = "harmonic"
    NONE = None

class ConfinementInput(BaseModel):
    type: ConfinementType = ConfinementType.NONE
    rc: float = Field(default=20.0, gt=0)
    ri_factor: float = Field(default=0.9, ge=0.0, le=1.0)
    Vbarrier: float = Field(default=1.0, gt=0.0)
    polarization_mode: Literal[None, "softcoul"] = None
    softcoul_delta: float = 0.1
    softcoul_charge: float = 1.0

    @field_validator("type", "polarization_mode", mode="before")
    @classmethod
    def make_lower(cls, v: str) -> str:
        """Convert to lower case to make the input case-insensitive."""
        return v.lower()
