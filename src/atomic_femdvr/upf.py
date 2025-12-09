"""Replacement for UPFInterface using the upf-tools python library."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Self
from upf_tools import UPFDict


class UPFInterface(BaseModel):
    zp: float
    etotps: float
    ecutrho: float
    lmax: int
    nwfc: int
    nbeta: int
    mesh: int
    xmin: float | None
    rmax: float | None
    dx: float | None
    r: npt.NDArray[np.float64]
    nchi: npt.NDArray[np.int32]
    lchi: npt.NDArray[np.int32]
    oc: npt.NDArray[np.float64]
    chi: npt.NDArray[np.float64]
    lll: npt.NDArray[np.int32]
    dion: npt.NDArray[np.float64]
    vloc: npt.NDArray[np.float64]
    kbeta: npt.NDArray[np.int32]
    beta: npt.NDArray[np.float64]
    rho_nlcc: npt.NDArray[np.float64] | None = None
    rho_atom: npt.NDArray[np.float64] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_array_dimensions(self) -> Self:
        """Check that array dimensions are consistent."""
        desired_shapes: dict[str, tuple[int, ...]] = {
            "r": (self.mesh,),
            "nchi": (self.nwfc,),
            "lchi": (self.nwfc,),
            "oc": (self.nwfc,),
            "chi": (self.mesh, self.nwfc),
            "lll": (self.nbeta,),
            "dion": (self.nbeta, self.nbeta),
            "vloc": (self.mesh,),
            "kbeta": (self.nbeta,),
            "beta": (self.mesh, self.nbeta),
            "rho_nlcc": (self.mesh,) if self.rho_nlcc is not None else None,
            "rho_atom": (self.mesh,) if self.rho_atom is not None else None
        }

        for array_name, desired_shape in desired_shapes.items():
            array_value = getattr(self, array_name)
            if array_value.shape != desired_shape:
                raise ValueError(f"Array '{array_name}' has incorrect shape: {array_value.shape}, expected {desired_shape}")
        return self

    @field_validator("r", "oc", "chi", "dion", "vloc", "beta", mode="before")
    @classmethod
    def ensure_numpy_array_float64(cls, v) -> npt.NDArray[np.float64]:
        """Ensure that the input is a numpy array."""
        return np.array(v, dtype=np.float64)

    @field_validator("nchi", "lchi", "lll", "kbeta", mode="before")
    @classmethod
    def ensure_numpy_array_int32(cls, v) -> npt.NDArray[np.int32]:
        """Ensure that the input is a numpy array."""
        return np.array(v, dtype=np.int32)

    @classmethod
    def from_upf(cls, filename: Path) -> Self:

        upf_dict = UPFDict.from_upf(filename)

        dij_1d = upf_dict['nonlocal']['dij']
        nbeta = upf_dict['header']['number_of_proj']
        assert dij_1d.size == nbeta**2
        dij = dij_1d.reshape((nbeta, nbeta))

        return cls(
            zp = upf_dict['header']['z_valence'],
            etotps = upf_dict['header']['total_psenergy'],
            ecutrho = upf_dict['header']['rho_cutoff'],
            lmax = upf_dict['header']['l_max'],
            nwfc = upf_dict['header']['number_of_wfc'],
            nbeta = nbeta,
            mesh = upf_dict['header']['mesh_size'],
            xmin = upf_dict['mesh'].get('xmin', None),
            rmax = upf_dict['mesh'].get('rmax', None),
            dx = upf_dict['mesh'].get('dx', None),
            r = upf_dict['mesh']['r'],
            nchi = [chi['n'] for chi in upf_dict['pswfc']['chi']],
            lchi = [chi['l'] for chi in upf_dict['pswfc']['chi']],
            oc = [chi['occupation'] for chi in upf_dict['pswfc']['chi']],
            chi = np.transpose([chi['content'] for chi in upf_dict['pswfc']['chi']]),
            lll = [beta['angular_momentum'] for beta in upf_dict['nonlocal']['beta']],
            dion = dij,
            vloc = upf_dict['local'],
            kbeta = [beta['cutoff_radius_index'] for beta in upf_dict['nonlocal']['beta']],
            beta = np.transpose([beta['content'] for beta in upf_dict['nonlocal']['beta']]),
            rho_nlcc = upf_dict.get('nlcc', None),
            rho_atom = upf_dict.get('rhoatom', None)
        )

    @property
    def nnodes_chi(self) -> npt.NDArray[np.int32]:
        return np.array([np.sum(self.lchi[:i] == l) for i, l in enumerate(self.lchi)])

    def get_charge_density(self) -> npt.NDArray[np.float64]:
        """Compute the charge density from the wavefunctions."""

        if self.rho_atom is not None:
            return self.rho_atom
        else:
            rho = np.zeros(self.mesh, dtype=np.float64)
            for iwf in range(self.nwfc):
                rho[1:] += self.oc[iwf] * np.abs(self.chi[1:, iwf])**2 / self.r[1:]**2
            rho[0] = rho[1]
            return rho
