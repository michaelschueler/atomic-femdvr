import numpy as np
#===================================================================
def linear_mixing(rho_new:np.ndarray, rho_old:np.ndarray, alpha:float) -> np.ndarray:
    """
    Perform Anderson mixing of two density arrays.
    :param rho_new: New density array
    :param rho_old: Old density array
    :param alpha: Mixing parameter (0 < alpha <= 1)
    :return: Mixed density array
    """
    rho_mixed = (1 - alpha) * rho_old + alpha * rho_new
    return rho_mixed
#===================================================================
