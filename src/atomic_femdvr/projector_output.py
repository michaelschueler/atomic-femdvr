"""Writers for radial wavefunctions / projectors in various downstream formats.

Three output formats are supported:

* ``"qe"`` -- Quantum ESPRESSO ``.dat`` text format on a logarithmic radial
  grid; converts :math:`u(r)` to :math:`R(r) = u(r)/r`.
* ``"hdf5"`` -- raw FEM-DVR coefficients with the basis breakpoints, useful
  for re-loading into another atomic-femdvr run.
* ``"bessel"`` -- HDF5 file of spherical-Bessel-transformed wavefunctions
  on a momentum grid; used by kapaow for openmx-style projector setups.
"""

import os

import h5py
import numpy as np

from atomic_femdvr.bessel_transform import bessel_integral
from atomic_femdvr.femdvr import FEDVR_Basis


# ----------------------------------------------------------
def write_projector_file(
    basis: FEDVR_Basis,
    phi: np.ndarray,
    elem: str,
    tag: str,
    nr: int = 1001,
    rmin: float = 1.0e-8,
    bessel_method: str = "simpson",
    bessel_npoints: int = 41,
    qgrid: np.ndarray | None = None,
    rpow: int = 1,
    out_dir: str = "./",
    output_format: str = "qe",
) -> None:
    """Write radial wavefunctions / projectors in the requested format.

    Parameters
    ----------
    basis
        FEM-DVR basis on which ``phi`` is represented.
    phi
        Wavefunctions, shape ``(lmax + 1, nmax + 1, ne*ng + 1)``. Stored as
        :math:`u(r) = r\\, R(r)`.
    elem
        Element symbol used to build output filenames.
    tag
        Free-form label embedded in output filenames (e.g. ``"DZP"``).
    nr
        Number of radial points on the QE log grid (forced to odd).
    rmin
        Smallest radius on the QE log grid (Bohr).
    bessel_method, bessel_npoints
        Quadrature options forwarded to :func:`bessel_integral` when
        ``output_format == "bessel"``.
    qgrid
        Momentum grid required when ``output_format == "bessel"``.
    rpow
        Power of :math:`r` in the Bessel-transform integrand.
    out_dir
        Destination directory.
    output_format
        One of ``"qe"``, ``"hdf5"``, ``"bessel"``.

    Raises
    ------
    ValueError
        If ``output_format`` is not recognised.
    """
    if output_format.lower() == "qe":
        # Quantum ESPRESSO format
        lmax = phi.shape[0] - 1
        nmax = phi.shape[1] - 1

        # check if nr is odd
        if nr % 2 == 0:
            nr += 1

        Rmax = basis.xp[-1]
        rs = np.logspace(np.log10(rmin), np.log10(Rmax), nr)

        # interpolate wavefunctions onto rs grid and convert from u(r) to R(r) = u(r)/r
        larr = []
        psi_interp = np.zeros([lmax + 1, nmax + 1, nr])
        for l in range(lmax + 1):
            for n in range(nmax + 1):
                psi_interp[l, n, :] = basis.interpolate(phi[l, n, :], rs) / rs
                larr.append(l)

        nproj = len(larr)
        psi_interp = psi_interp.reshape(nproj, nr)

        write_projector_qe(out_dir, elem, tag, larr, psi_interp, rs)

    elif output_format.lower() == "hdf5":
        write_projector_hdf5(out_dir, elem, tag, phi, basis)

    elif output_format.lower() == "bessel" and qgrid is not None:
        lmax = phi.shape[0] - 1
        nmax = phi.shape[1] - 1
        phi_bessel = np.zeros([lmax + 1, nmax + 1, len(qgrid)])

        for l in range(lmax + 1):
            for n in range(nmax + 1):
                phi_bessel[l, n, :] = bessel_integral(
                    basis,
                    l,
                    rpow,
                    qgrid,
                    phi[l, n, :],
                    npoints=bessel_npoints,
                    method=bessel_method,
                )

        write_bessel_hdf5(out_dir, elem, tag, phi_bessel, qgrid)

    else:
        raise ValueError(
            f"Unknown output format: {output_format}. Supported formats are 'qe', 'hdf5', 'bessel'."
        )


# ----------------------------------------------------------
def write_projector_qe(
    out_dir: str, elem: str, tag: str, proj_l: list, phi: np.ndarray, rs: np.ndarray
) -> None:
    """Write projectors in Quantum ESPRESSO ``.dat`` text format.

    Parameters
    ----------
    out_dir
        Destination directory.
    elem
        Element symbol for the filename.
    tag
        Free-form label for the filename.
    proj_l
        Per-projector angular momentum quantum numbers.
    phi
        Projectors :math:`R(r)`, shape ``(nproj, nr)``.
    rs
        Logarithmic radial grid, shape ``(nr,)``.
    """
    fname = os.path.join(out_dir, f"{elem}_{tag}_qe.dat")

    nproj = phi.shape[0]
    nr = len(rs)

    with open(fname, "w") as f:
        f.write(f"{nr} {nproj} \n")
        for j in range(nproj):
            f.write(f"{proj_l[j]} ")
        f.write("\n")
        for i, r in enumerate(rs):
            x = np.log(r)
            s = f"{x}  {r} "
            for j in range(nproj):
                s += f"{phi[j, i]} "
            f.write(s + "\n")


# ----------------------------------------------------------
def write_projector_hdf5(
    out_dir: str, elem: str, tag: str, phi: np.ndarray, basis: FEDVR_Basis
) -> None:
    """Write FEM-DVR-coefficient wavefunctions to an HDF5 file.

    Stores the basis breakpoints (``xp``) and shape parameters as
    attributes / datasets so the file can be reloaded into a fresh
    :class:`FEDVR_Basis` of matching dimensions.

    Parameters
    ----------
    out_dir
        Destination directory.
    elem
        Element symbol for the filename.
    tag
        Free-form label for the filename.
    phi
        Wavefunctions, shape ``(lmax + 1, nmax + 1, ne*ng + 1)``.
    basis
        Basis whose breakpoints / dimensions are stored alongside ``phi``.
    """
    fname = os.path.join(out_dir, f"{elem}_{tag}_wfc.h5")

    with h5py.File(fname, "w") as f:
        f.attrs["ne"] = basis.ne
        f.attrs["ng"] = basis.ng
        f.attrs["lmax"] = phi.shape[0] - 1
        f.attrs["nmax"] = phi.shape[1] - 1
        f.create_dataset("xp", data=basis.xp)
        f.create_dataset("wf", data=phi)


# ----------------------------------------------------------
def write_bessel_hdf5(
    out_dir: str, elem: str, tag: str, phi_bessel: np.ndarray, qgrid: np.ndarray
) -> None:
    """Write spherical-Bessel-transformed wavefunctions to an HDF5 file.

    Parameters
    ----------
    out_dir
        Destination directory.
    elem
        Element symbol for the filename.
    tag
        Free-form label for the filename.
    phi_bessel
        Bessel-transformed wavefunctions, shape ``(lmax + 1, nmax + 1, nq)``.
    qgrid
        Momentum grid, shape ``(nq,)``.
    """
    fname = os.path.join(out_dir, f"{elem}_{tag}_bessel.h5")

    with h5py.File(fname, "w") as f:
        f.attrs["lmax"] = phi_bessel.shape[0] - 1
        f.attrs["nmax"] = phi_bessel.shape[1] - 1
        f.create_dataset("qgrid", data=qgrid)
        f.create_dataset("wf_bessel", data=phi_bessel)
