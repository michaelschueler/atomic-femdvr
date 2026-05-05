import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


# ===================================================================
def fit_inverse_power_potential(r_vals: np.ndarray, V_vals: np.ndarray, N: int) -> np.ndarray:
    """
    Fits V(r) ≈ sum_{n=1}^N A_n / r^n
    """
    r_vals = np.array(r_vals)
    V_vals = np.array(V_vals)

    # Build design matrix X
    X = np.stack([1 / r_vals**n for n in range(1, N + 1)], axis=1)

    # Solve least-squares fit
    coeffs, *_ = np.linalg.lstsq(X, V_vals, rcond=None)

    return coeffs


# ===================================================================
def interpolate_potential(rs: np.ndarray, Vs: np.ndarray, r_new: np.ndarray) -> np.ndarray:
    """
    Interpolates the potential V(r) at a new radius r_new
    using the provided radial points rs and potential values Vs.
    """
    interp = interp1d(rs, Vs, kind="cubic", bounds_error=False, fill_value=0.0)

    V_new = np.zeros_like(r_new)

    (Ir,) = np.where(r_new < rs[-1])
    V_new[Ir] = interp(r_new[Ir])

    # extrapolate for r_new > rs[-1]
    if r_new[-1] > rs[-1]:
        r_vals = rs[-4:]  # last 4 points for extrapolation
        V_vals = Vs[-4:]

        coeffs = fit_inverse_power_potential(r_vals, V_vals, len(r_vals))

        for i, c in enumerate(coeffs):
            V_new[r_new > rs[-1]] += c / r_new[r_new > rs[-1]] ** (i + 1)

    return V_new


# ===================================================================
def interpolate_density(rs: np.ndarray, rho: np.ndarray, r_new: np.ndarray) -> np.ndarray:
    """
    Interpolates the charge density rho at new radial points r_new
    using the provided radial points rs and density values rho.
    """
    interp = UnivariateSpline(rs, rho, k=3, s=0)

    rho_new = np.zeros_like(r_new)

    (Ir,) = np.where(r_new < rs[-1])
    rho_new[Ir] = interp(r_new[Ir])

    # extrapolate for r_new > rs[-1]
    if (r_new[-1] > rs[-2]) and (rho[-2] > 1.0e-10):
        der_interp = interp.derivative()
        lam = -der_interp(rs[-2]) / rho[-2]
        rho_new[r_new > rs[-1]] = rho[-2] * np.exp(-lam * (r_new[r_new > rs[-1]] - rs[-1]))

    return rho_new
