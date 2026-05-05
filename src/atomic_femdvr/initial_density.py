import numpy as np
from scipy.special import factorial, genlaguerre


# ==========================================================================
def hydrogenic_orbital(r: np.ndarray, Z: float, n: int, l: int) -> np.ndarray:
    """
    Computes the hydrogenic orbital for given quantum numbers n and l.

    Parameters:
    r : np.ndarray
        Radial grid points.
    Z : float
        Nuclear charge.
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number.

    Returns:
    np.ndarray
        The radial part of the hydrogenic orbital evaluated at r.
    """
    # Normalization constant
    a0 = 1.0  # Bohr radius in atomic units
    rho = 2 * Z * r / (n * a0)
    prefactor = np.sqrt((2 * Z / (n * a0)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))

    # Radial part
    radial_part = prefactor * (rho**l) * np.exp(-rho / 2) * genlaguerre(n - l - 1, 2 * l + 1)(rho)

    return radial_part


# ==========================================================================
def slater_shielding(
    n_vals: np.ndarray,
    l_vals: np.ndarray,
    occ_vals: np.ndarray,
    n: int,
    l: int,
) -> float:
    """
    Compute shielding S for an electron in 'target_orb' (e.g. '3p')
    according to Slater's rules.
    """
    S = 0.0

    for ni, li, count in zip(n_vals, l_vals, occ_vals, strict=False):
        if ni == n and li == l:
            # Same group
            if n == 1:
                S += 0.30 * (count - 1)
            else:
                S += 0.35 * (count - 1)
        else:
            if l in [0, 1]:  # s or p electron
                if ni == n - 1:
                    S += 0.85 * count
                elif ni < n - 1:
                    S += 1.00 * count
            else:  # d or f electron
                if ni < n:
                    S += 1.00 * count
    return S


# ==========================================================================
def get_slater_density(
    r: np.ndarray, Z: float, n_vals: np.ndarray, l_vals: np.ndarray, occ_vals: np.ndarray
) -> np.ndarray:
    density = np.zeros_like(r)

    for n, l, occ in zip(n_vals, l_vals, occ_vals, strict=False):
        S = slater_shielding(n_vals, l_vals, occ_vals, n, l)
        Z_eff = Z - S

        print(f"n = {n}, l = {l}, occ = {occ}, Zeff = {Z_eff}")

        radial_wavefunction = hydrogenic_orbital(r, Z_eff, n, l)
        density += occ * radial_wavefunction**2

    return density


# ==========================================================================
