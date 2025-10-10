import numpy as np


#===================================================================
def SoftConfinement(r, ri, rc, bignumber=1.0e10):
    """
    Soft confinement potential for a given radius r, inner radius ri, and outer radius rc.
    """
    Vc = np.zeros_like(r)

    Ir, = np.where(r < ri)
    if len(Ir) > 0:
        Vc[Ir] = 0.0
    Ir, = np.where(r >= rc)
    if len(Ir) > 0:
        Vc[Ir] = bignumber

    Ir, = np.where((r >= ri) & (r < rc))
    if len(Ir) > 0:
        Vc[Ir] = np.exp(-(rc - ri) / (r[Ir] - ri)) / (rc - r[Ir])

    return Vc
#===================================================================
def ParabolicConfinement(r, ri, rc, bignumber=1.0e3):
    """
    Parabolic confinement potential for a given radius r, inner radius ri, and outer radius rc.
    """
    Vc = np.zeros_like(r)

    Ir, = np.where(r < ri)
    if len(Ir) > 0:
        Vc[Ir] = 0.0
    Ir, = np.where(r >= rc)
    if len(Ir) > 0:
        Vc[Ir] = bignumber

    Ir, = np.where((r >= ri) & (r < rc))
    if len(Ir) > 0:
        Vc[Ir] = (r[Ir] - ri)**2 / (rc - ri)**2

    return Vc
#===================================================================
def SoftStep(r, ri, rc, Vbarrier=1.0e1):
    """
    Soft step potential for a given radius r, inner radius ri, and outer radius rc.
    """
    Vc = np.zeros_like(r)

    Ir, = np.where(r < ri)
    if len(Ir) > 0:
        Vc[Ir] = 0.0
    Ir, = np.where(r >= rc)
    if len(Ir) > 0:
        Vc[Ir] = Vbarrier

    Ir, = np.where((r >= ri) & (r < rc))
    if len(Ir) > 0:
        Vc[Ir] = Vbarrier * np.sin(0.5*np.pi * (r[Ir] - ri) / (rc - ri))**2

    return Vc
#===================================================================
def SoftCoulombPotential(r, Q, delta, lam=0.0):
    return -Q / np.sqrt(r**2 + delta**2) * np.exp(-lam * r)
