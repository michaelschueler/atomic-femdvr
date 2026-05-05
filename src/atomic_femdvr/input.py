"""Pydantic input models for atomic-femdvr.

These models define the JSON / Python API used to configure
:func:`atomic_femdvr.solve_atomic` and
:func:`atomic_femdvr.solve_pseudo_atomic`. Each top-level solver input
groups several of these models (see :class:`atomic_femdvr.FullAtomicInput`
and :class:`atomic_femdvr.PseudoAtomicInput`).
"""

from enum import Enum
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, FilePath, create_model, field_validator


class EnergyUnit(str, Enum):
    """Energy unit used when reading external potential files."""

    RYDBERG = "Ry"
    ELECTRONVOLTS = "eV"
    HARTREE = "Ha"


class BaseModel(PydanticBaseModel):
    """Project-wide pydantic base model.

    Forbids extra fields (so typos in JSON inputs raise a clear error) and
    re-runs validators on assignment.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SysParamsInput(BaseModel):
    """System-level parameters: element identity and potential / pseudopotential files.

    At least ``element`` must be set. For all-electron runs ``file_pot``
    typically points at a tabulated Coulomb potential; for pseudo-atomic
    runs ``file_upf`` points at a UPF pseudopotential.
    """

    file_pot: FilePath | None = Field(
        default=None,
        description="Path to a tabulated potential file used by the all-electron solver.",
    )
    file_upf: FilePath | None = Field(
        default=None,
        description="Path to a UPF pseudopotential file used by the pseudo-atomic solver.",
    )
    file_vhx: FilePath | None = Field(
        default=None,
        description="Path to a tabulated Hartree + exchange-correlation potential.",
    )
    file_rho: FilePath | None = Field(
        default=None,
        description="Path to a tabulated electron density file (used as an SCF starting guess).",
    )
    pot_columns: tuple[int, int] = Field(
        default=(0, 4),
        description="Column indices ``(r_col, V_col)`` into ``file_pot``.",
    )
    pot_energy_unit: EnergyUnit = Field(
        default=EnergyUnit.RYDBERG,
        description="Energy unit of ``file_pot``; converted to Hartree internally.",
    )
    lmax: int = Field(
        default=0,
        ge=0,
        description="Maximum angular momentum quantum number to compute.",
    )
    nmax: int = Field(
        default=4,
        ge=0,
        description="Maximum number of bound states per angular-momentum channel.",
    )
    element: str | None = Field(
        default=None,
        description="Periodic-table symbol (e.g. ``'Mo'``); case-insensitive on input.",
    )

    @field_validator("pot_energy_unit", mode="before")
    @classmethod
    def map_assume_isolated(cls, v: str) -> str:
        """Normalise ``pot_energy_unit`` strings (``'rydberg'`` -> ``'Ry'`` etc.)."""
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
    def validate_element(cls, v: str | None) -> str:
        """Strip / capitalise the element symbol and reject empty / overlong values."""
        if v is None:
            raise ValueError("Element must be specified in sysparams.")
        if len(v.strip()) == 0:
            raise ValueError("Element must be a non-empty string.")
        if len(v.strip()) > 2:
            raise ValueError("Element symbol must be one or two characters.")
        return v.strip().capitalize()


class ControlInput(BaseModel):
    """Run-control flags: storage location, restart behaviour, density caching."""

    storage_dir: Path = Field(
        default=Path(),
        description=(
            "Directory used for SCF restart files (``density_potential.pkl``) "
            "and other run artifacts."
        ),
    )
    restart: bool = Field(
        default=False,
        description="If ``True``, attempt to resume from a previously saved state.",
    )
    store_density: bool = Field(
        default=True,
        description="Write the converged density / potential to ``storage_dir`` on completion.",
    )


class OutputInput(BaseModel):
    """Output toggles for the pseudo-atomic solver."""

    output_wfc_qe: bool = Field(
        default=False,
        description="Write radial wavefunctions in Quantum ESPRESSO text format.",
    )
    output_wfc_hdf5: bool = Field(
        default=False,
        description="Write radial wavefunctions in HDF5 format.",
    )
    output_wfc_bessel: bool = Field(
        default=False,
        description="Write spherical-Bessel-transformed wavefunctions (HDF5).",
    )
    output_dipole_moments: bool = Field(
        default=False,
        description="Write dipole matrix elements between bound states.",
    )
    bessel_quad_npoints: int = Field(
        default=41,
        ge=3,
        description="Quadrature points per element used in the Bessel transform.",
    )
    bessel_quad_method: Literal["simpson", "lobatto"] = Field(
        default="simpson",
        description="Quadrature method: uniform Simpson or Gauss-Lobatto.",
    )
    bessel_qmax: float = Field(
        default=50.0,
        gt=0.0,
        description="Maximum momentum (atomic units) sampled in the Bessel transform.",
    )
    bessel_rpow: int = Field(
        default=1,
        ge=1,
        description="Power of :math:`r` multiplying the integrand in the Bessel transform.",
    )
    bessel_nq: int = Field(
        default=201,
        ge=3,
        description="Number of momentum-grid points between 0 and ``bessel_qmax``.",
    )
    qe_num_points: int = Field(
        default=1001,
        ge=3,
        description="Number of points on the Quantum ESPRESSO output radial grid (logarithmic).",
    )
    qe_rmin: float = Field(
        default=1.0e-8,
        gt=0.0,
        description="Smallest radius on the QE output grid (Bohr).",
    )

    @field_validator("bessel_quad_method", mode="before")
    @classmethod
    def make_lower(cls, v: str) -> str:
        """Lower-case the method name so the input is case-insensitive."""
        return v.lower()


class ElectronsInput(BaseModel):
    """Electron configuration for the all-electron solver."""

    Z: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Nuclear charge. ``0`` (the default) means 'infer from ``sysparams.element``'."
        ),
    )
    configuration: list[str] = Field(
        default_factory=lambda: ["1s1"],
        description=(
            "Occupied shells in spectroscopic notation, e.g. "
            "``['1s2', '2s2', '2p6', '3s2', '3p4']`` for sulfur."
        ),
    )


class SolverInput(BaseModel):
    """Discretisation and eigensolver parameters."""

    theory_level: Literal["non-relativistic", "zora", "scalar-relativistic"] = Field(
        default="non-relativistic",
        description="Radial Hamiltonian to solve.",
    )
    eigensolver: Literal["full", "banded"] = Field(
        default="full",
        description="LAPACK eigensolver: dense (``'full'``) or banded.",
    )
    h_min: float = Field(
        default=0.5,
        gt=0,
        description="Smallest radial element width (Bohr) used by the adaptive element generator.",
    )
    h_max: float = Field(
        default=4.0,
        gt=0,
        description="Largest radial element width (Bohr) used by the adaptive element generator.",
    )
    Rmax: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Outer radial cutoff (Bohr). If ``None``, taken from the input potential / UPF file."
        ),
    )
    tol: float = Field(
        default=1.0e-3,
        gt=0,
        description="Residual tolerance for the radial Schrödinger solver.",
    )
    ng: int = Field(
        default=8,
        ge=1,
        description="Number of Gauss-Lobatto quadrature points per element.",
    )
    elem_tol: float = Field(
        default=1.0e-2,
        gt=0,
        description="Tolerance used by the adaptive element generator when subdividing.",
    )

    @field_validator("theory_level", "eigensolver", mode="before")
    @classmethod
    def make_lower(cls, v: str) -> str:
        """Lower-case the field value so the input is case-insensitive."""
        return v.lower()


def solver_input_factory(default_hmin: float, default_hmax: float) -> type[SolverInput]:
    """Return a :class:`SolverInput` subclass with custom default element widths.

    Used by the all-electron and pseudo-atomic solvers, which want different
    default ``h_min`` / ``h_max`` values without repeating the rest of the
    schema.

    Parameters
    ----------
    default_hmin
        Default value of ``h_min`` for the returned model.
    default_hmax
        Default value of ``h_max`` for the returned model.

    Returns
    -------
    type[SolverInput]
        A dynamically-created subclass of :class:`SolverInput`.
    """
    model = create_model(
        "SolverInput",
        __base__=SolverInput,
        h_min=(float, Field(default=default_hmin, gt=0)),
        h_max=(float, Field(default=default_hmax, gt=0)),
    )
    return cast(type[SolverInput], model)


class DFTInput(BaseModel):
    """DFT driver, exchange-correlation functional, and SCF mixing parameters."""

    driver: str = Field(
        default="internal",
        description=(
            "XC backend: ``'internal'`` (a hand-coded GGA implementation) or "
            "``'pylibxc'`` (libxc via its Python bindings)."
        ),
    )
    xc_functional: str = Field(
        default="PBE",
        description=(
            "Combined exchange-correlation functional name. Set to ``''`` to use "
            "``x_functional`` and ``c_functional`` separately."
        ),
    )
    x_functional: str | None = Field(
        default="gga_x_pbe",
        description="libxc identifier for the exchange functional.",
    )
    c_functional: str | None = Field(
        default="gga_c_pbe",
        description="libxc identifier for the correlation functional.",
    )
    mixing_scheme: Literal["linear", "diis", "anderson"] = Field(
        default="linear",
        description="SCF mixing strategy.",
    )
    diis_history: int = Field(
        default=5,
        ge=2,
        description="Number of past iterates kept by the DIIS / Anderson mixer.",
    )
    alpha_mix: float = Field(
        default=0.6,
        description="Linear mixing weight (and seed weight for DIIS / Anderson).",
    )
    alpha_x: float = Field(
        default=1.0,
        description="Mixing factor of the exchange part (used by hybrid functionals).",
    )
    max_iter: int = Field(
        default=100,
        description="Maximum SCF iterations.",
    )
    conv_tol: float = Field(
        default=1.0e-6,
        description="Density-difference threshold below which SCF is considered converged.",
    )

    @field_validator("mixing_scheme", mode="before")
    @classmethod
    def make_lower(cls, v: str) -> str:
        """Lower-case the mixing-scheme name so the input is case-insensitive."""
        return v.lower()


class PseudoConfigInput(BaseModel):
    """Pseudo-atomic-only configuration overrides."""

    storage_dir: Path = Field(
        default=Path(),
        description="Output directory for pseudo-atomic-specific cache files.",
    )


class ConfinementType(str, Enum):
    """Functional form of the confining potential applied to virtual orbitals."""

    SOFTSTEP = "softstep"
    HARMONIC = "harmonic"
    NONE = None


class ConfinementInput(BaseModel):
    """Confining potential applied when computing virtual / unbound orbitals.

    Used during the ``"nscf"`` step to localise high-energy states into a
    finite radial box (mimicking the role of the unit cell in a periodic code).
    """

    type: ConfinementType = Field(
        default=ConfinementType.NONE,
        description="Confinement functional form. See :class:`ConfinementType`.",
    )
    rc: float = Field(
        default=20.0,
        gt=0,
        description=(
            "Outer cutoff radius (Bohr) at which the confining wall reaches its full height."
        ),
    )
    ri_factor: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Inner radius (Bohr) of the confining region as a fraction of ``rc``; "
            "the confinement is zero for :math:`r < r_i`."
        ),
    )
    Vbarrier: float = Field(
        default=1.0,
        gt=0.0,
        description="Height of the confining potential at :math:`r = r_c` (Hartree).",
    )
    polarization_mode: Literal[None, "softcoul"] = Field(
        default=None,
        description=(
            "If ``'softcoul'``, a soft-Coulomb tail is added to model an external dipole field."
        ),
    )
    softcoul_delta: float = Field(
        default=0.1,
        description=(
            "Smoothing parameter :math:`\\delta` in the soft-Coulomb potential "
            ":math:`-Q / \\sqrt{r^2 + \\delta^2}`."
        ),
    )
    softcoul_charge: float = Field(
        default=1.0,
        description="Effective charge :math:`Q` of the soft-Coulomb tail (atomic units).",
    )

    @field_validator("type", "polarization_mode", mode="before")
    @classmethod
    def make_lower(cls, v: str) -> str:
        """Lower-case the field value so the input is case-insensitive."""
        return v.lower()
