"""Spherical Bessel transform of wavefunctions on a FEM-DVR basis.

The transform implemented here is

.. math::

    \\widetilde{\\phi}(q) = \\int_0^{R_{\\max}} r^{p}\\, \\phi(r)\\,
    j_{\\ell}(q\\,r)\\, dr

evaluated element-by-element using either Simpson's rule on a uniform
sub-grid (the default) or Gauss-Lobatto quadrature on the basis grid.
Used by the pseudo-atomic exporter to write reciprocal-space radial
wavefunctions.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn

from atomic_femdvr.femdvr import FEDVR_Basis

__all__ = ["bessel_integral"]


# =================================================================
def bessel_integral(
    basis: FEDVR_Basis,
    l: int,
    rpow: int,
    qgrid: np.ndarray,
    phi: np.ndarray,
    npoints: int = 41,
    method: str = "simpson",
) -> np.ndarray:
    """Spherical Bessel transform of a radial wavefunction.

    Computes :math:`\\int_0^{R_{\\max}} r^{p}\\, \\phi(r)\\, j_{\\ell}(q r)\\, dr`
    on a momentum grid by element-wise quadrature.

    Parameters
    ----------
    basis
        FEM-DVR basis on which ``phi`` is represented.
    l
        Angular momentum quantum number :math:`\\ell` selecting
        :math:`j_{\\ell}`.
    rpow
        Power :math:`p` of the radial coordinate in the integrand.
    qgrid
        Momentum grid on which the transform is evaluated, shape ``(nq,)``.
    phi
        Wavefunction on the FEM-DVR grid, shape ``(ne*ng + 1,)``.
    npoints
        Number of sub-grid points per element when ``method == "simpson"``.
    method
        Quadrature scheme: ``"simpson"`` (uniform sub-grid) or ``"lobatto"``
        (the basis Gauss-Lobatto nodes).

    Returns
    -------
    np.ndarray
        Bessel-transformed wavefunction, shape ``(nq,)``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported quadrature schemes.
    """
    ne = basis.ne
    ng = basis.ng
    xp = basis.xp
    grid = basis.get_gridpoints()

    phiq = np.zeros_like(qgrid)

    for i in range(ne):
        # Get the grid points for the current element
        r_elem = grid[i * ng : i * ng + ng + 1]
        phi_elem = phi[i * ng : i * ng + ng + 1]

        # Convert to Legendre coefficients
        c_elem = basis.leg.to_spectral(phi_elem)

        if method.lower() == "simpson":
            # Perform integration using Simpson's rule
            rs = np.linspace(xp[i], xp[i + 1], npoints)
            xred = np.linspace(-1, 1, npoints)
            psi_elem_lin = np.polynomial.legendre.legval(xred, c_elem)
            integrand = (
                rs[:, None] ** rpow
                * psi_elem_lin[:, None]
                * spherical_jn(l, rs[:, None] * qgrid[None, :])
            )
            phiq = phiq + simpson(integrand, dx=rs[1] - rs[0], axis=0)

        elif method.lower() == "lobatto":
            # Perform integration using Gauss-Lobatto quadrature
            h = 0.5 * (xp[i + 1] - xp[i])
            w_i = basis.leg.w_i * h
            integrand = (
                r_elem[None, :] ** rpow
                * phi_elem[None, :]
                * spherical_jn(l, r_elem[None, :] * qgrid[:, None])
            )
            phiq = phiq + np.dot(integrand, w_i)
        else:
            raise ValueError(f"Unknown integration method: {method}")

    return phiq


# =================================================================
