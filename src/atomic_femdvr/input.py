"""Input models (JSON sections) for the atomic solvers."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, ConfigDict, Field, FilePath, create_model, field_validator
from pydantic import BaseModel as PydanticBaseModel


def _lower(v: Any) -> Any:
    """Lower-case strings so that string options are case-insensitive."""
    return v.lower() if isinstance(v, str) else v


class BaseModel(PydanticBaseModel):

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class SysParamsInput(BaseModel):
    file_upf: FilePath | None = None
    lmax: int = Field(default=0, ge=0)
    nmax: int = Field(default=4, ge=0)
    element: str | None = None
    oc_override: list[float] | None = Field(default=None,
        description="Override UPF pseudo-wavefunction occupations (one value per chi entry, "
                    "same order as the UPF PP_PSWFC block). Two uses: (1) fixing UPF files "
                    "with wrong occupations, e.g. GTH pseudopotentials converted by cpmd2upf, "
                    "which write spin-majority occupations instead of total occupations; "
                    "(2) setting up a charged/ionized pseudo-atom by giving occupations that "
                    "sum to z_valence minus the desired charge.")

    @field_validator("element", mode="before")
    @classmethod
    def validate_element(cls, v: str | None) -> str:
        if v is None:
            raise ValueError("Element must be specified in sysparams.")
        if len(v.strip()) == 0:
            raise ValueError("Element must be a non-empty string.")
        if len(v.strip()) > 2:
            raise ValueError("Element symbol must be one or two characters.")
        return v.strip().capitalize()

class ControlInput(BaseModel):
    storage_dir: Path = Field(default=Path())
    store_density: bool = Field(default=True,
        description="Save the converged density and potential to storage_dir for later restarts.")

class OutputInput(BaseModel):
    output_wfc_qe : bool = Field(default=False)
    output_wfc_hdf5 : bool = Field(default=False)
    output_wfc_bessel : bool = Field(default=False)
    output_wfc_pyimpurity : bool = Field(default=False)
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
    def make_lower(cls, v: Any) -> Any:
        """Make string options case-insensitive."""
        return _lower(v)

class ElectronsInput(BaseModel):
    Z: float = Field(default=0.0, ge=0)
    configuration: list[str] = Field(default_factory=lambda: ["1s1"])

class SolverInput(BaseModel):
    eigensolver: Literal["full", "banded"] = Field(default="full")
    h_min: float = Field(default=0.5, gt=0)
    h_max: float = Field(default=4.0, gt=0)
    Rmax: float | None = Field(default=None, gt=0)
    tol: float = Field(default=1.0e-3, gt=0)
    ng: int = Field(default=12, ge=1,
        description="Gauss-Lobatto points per finite element. 12 converges eigenvalues to ~1e-7 Ha.")
    elem_tol: float = Field(default=1.0e-2, gt=0)
    nuclear_sigma: float = Field(default=1.0e-3, gt=0)

    @field_validator("eigensolver", mode="before")
    @classmethod
    def make_lower(cls, v: Any) -> Any:
        """Make string options case-insensitive."""
        return _lower(v)

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
    theory_level: Literal["non-relativistic", "zora", "scalar-relativistic"] = Field(default="non-relativistic")
    relativistic_scf: bool = Field(default=True)
    driver: Literal["internal", "pylibxc"] = "internal"
    xc_functional: str = "PBE"
    x_functional: str | None = "gga_x_pbe"
    c_functional: str | None = "gga_c_pbe"
    mixing_scheme: Literal["linear", "anderson"] = Field(default="linear",
        description="Density mixing. 'diis' (Pulay) is accepted as an alias for 'anderson'.")
    mixing_history: int = Field(default=5, ge=2,
        validation_alias=AliasChoices("mixing_history", "diis_history"))
    alpha_mix: float = 0.6
    alpha_x: float = 1.0
    max_iter: int = 100
    conv_tol: float = 1.0e-6

    @field_validator("theory_level", "driver", mode="before")
    @classmethod
    def make_lower(cls, v: Any) -> Any:
        """Make string options case-insensitive."""
        return _lower(v)

    @field_validator("mixing_scheme", mode="before")
    @classmethod
    def map_diis(cls, v: Any) -> Any:
        v = _lower(v)
        return "anderson" if v in ("diis", "pulay") else v

class PhotoemissionInput(BaseModel):
    energy_min: float = Field(default=0.1, gt=0)
    energy_max: float = Field(default=3.0, gt=0)
    n_energies: int = Field(default=50, ge=2)
    gauges: list[Literal["length", "velocity"]] = Field(default_factory=lambda: ["length"])
    store_wavefunctions: bool = Field(default=False)
    output_prefix: str = Field(default="photoemission")
    lmax_scatter: int | None = Field(default=None, ge=0,
        description="Maximum l for scattering final states. "
                    "Defaults to None (all l_f required by dipole selection rule).")


class ConfinementType(str, Enum):
    NONE = "none"
    SOFTSTEP = "softstep"
    HARMONIC = "harmonic"

class ConfinementInput(BaseModel):
    type: ConfinementType = ConfinementType.NONE
    rc: float = Field(default=20.0, gt=0)
    ri_factor: float = Field(default=0.9, ge=0.0, le=1.0)
    Vbarrier: float = Field(default=1.0, gt=0.0)
    polarization_mode: Literal[None, "softcoul"] = None
    softcoul_delta: float = 0.1
    softcoul_charge: float = 1.0

    @field_validator("type", mode="before")
    @classmethod
    def map_type(cls, v: Any) -> Any:
        return "none" if v is None else _lower(v)

    @field_validator("polarization_mode", mode="before")
    @classmethod
    def make_lower(cls, v: Any) -> Any:
        """Make string options case-insensitive."""
        return _lower(v)
