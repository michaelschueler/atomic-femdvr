import numpy as np
#===================================================================
def initial_density(r:np.ndarray, Z:float, zeta:float) -> np.ndarray:
    """
    Generate an initial guess for the electron density using a simple exponential decay model.
    :param r: Radial grid points
    :param Z: Nuclear charge
    :param zeta: Effective charge for the decay
    :return: Initial electron density on the radial grid
    """
    rho_init = Z * (zeta**3) / (8 * np.pi) * np.exp(-2 * zeta * r)
    return rho_init
#===================================================================
