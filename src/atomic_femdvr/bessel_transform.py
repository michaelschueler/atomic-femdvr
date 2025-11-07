import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn

from atomic_femdvr.femdvr import FEDVR_Basis
#=================================================================
def bessel_integral(basis: FEDVR_Basis, l: int, rpow: int, qgrid: np.ndarray,
                     phi: np.ndarray, npoints: int = 41, 
                     method: str = 'simpson') -> np.ndarray:
    """
    Perform the spherical Bessel transform of the wavefunction.

    Args:
        basis (FEDVR_Basis): The finite element basis.
        l (int): The angular momentum quantum number.
        rpow (int): The power of the radial coordinate.
        phi (np.ndarray): The wavefunction.

    Returns:
        np.ndarray: The Bessel-transformed wavefunction.
    """
    
    ne = basis.ne
    ng = basis.ng
    xp = basis.xp
    grid = basis.get_gridpoints()

    phiq = np.zeros_like(qgrid)

    for i in range(ne):
        # Get the grid points for the current element
        r_elem = grid[i*ng : i*ng + ng + 1]
        phi_elem = phi[i*ng : i*ng + ng + 1]

        # Convert to Legendre coefficients
        c_elem = basis.leg.to_spectral(phi_elem)

        if method.lower() == 'simpson':
            # Perform integration using Simpson's rule
            rs = np.linspace(xp[i], xp[i+1], npoints)
            xred = np.linspace(-1, 1, npoints)
            psi_elem_lin = np.polynomial.legendre.legval(xred, c_elem)
            integrand = rs[:, None]**rpow * psi_elem_lin[:, None] * spherical_jn(l, rs[:, None] * qgrid[None, :])
            phiq = phiq + simpson(integrand, dx=rs[1] - rs[0], axis=0)

        elif method.lower() == 'lobatto':
            # Perform integration using Gauss-Lobatto quadrature
            h = 0.5 * (xp[i+1] - xp[i])
            w_i = basis.leg.w_i * h
            integrand = r_elem[None, :]**rpow * phi_elem[None, :] * spherical_jn(l, r_elem[None, :] * qgrid[:, None])
            phiq = phiq + np.dot(integrand, w_i)
        else:
            raise ValueError(f"Unknown integration method: {method}")

    return phiq
#=================================================================