import numpy as np


# ===================================================================
def soft_confinement(r: np.ndarray, ri: float, rc: float, bignumber: float = 1.0e10) -> np.ndarray:
    """
    Soft confinement potential for a given radius r, inner radius ri, and outer radius rc.
    """
    Vc = np.zeros_like(r)

    (Ir,) = np.where(r < ri)
    if len(Ir) > 0:
        Vc[Ir] = 0.0
    (Ir,) = np.where(r >= rc)
    if len(Ir) > 0:
        Vc[Ir] = bignumber

    (Ir,) = np.where((r >= ri) & (r < rc))
    if len(Ir) > 0:
        Vc[Ir] = np.exp(-(rc - ri) / (r[Ir] - ri)) / (rc - r[Ir])

    return Vc


# ===================================================================
def parabolic_confinement(
    r: np.ndarray,
    ri: float,
    rc: float,
    bignumber: float = 1.0e3,
) -> np.ndarray:
    """
    Parabolic confinement potential for a given radius r, inner radius ri, and outer radius rc.
    """
    Vc = np.zeros_like(r)

    (Ir,) = np.where(r < ri)
    if len(Ir) > 0:
        Vc[Ir] = 0.0
    (Ir,) = np.where(r >= rc)
    if len(Ir) > 0:
        Vc[Ir] = bignumber

    (Ir,) = np.where((r >= ri) & (r < rc))
    if len(Ir) > 0:
        Vc[Ir] = (r[Ir] - ri) ** 2 / (rc - ri) ** 2

    return Vc


# ===================================================================
def soft_step(r: np.ndarray, ri: float, rc: float, Vbarrier: float = 1.0e1) -> np.ndarray:
    """
    Soft step potential for a given radius r, inner radius ri, and outer radius rc.
    """
    Vc = np.zeros_like(r)

    (Ir,) = np.where(r < ri)
    if len(Ir) > 0:
        Vc[Ir] = 0.0
    (Ir,) = np.where(r >= rc)
    if len(Ir) > 0:
        Vc[Ir] = Vbarrier

    (Ir,) = np.where((r >= ri) & (r < rc))
    if len(Ir) > 0:
        Vc[Ir] = Vbarrier * np.sin(0.5 * np.pi * (r[Ir] - ri) / (rc - ri)) ** 2

    return Vc


# ===================================================================
def soft_coulomb_potential(r: np.ndarray, Q: float, delta: float, lam: float = 0.0) -> np.ndarray:
    return -Q / np.sqrt(r**2 + delta**2) * np.exp(-lam * r)
