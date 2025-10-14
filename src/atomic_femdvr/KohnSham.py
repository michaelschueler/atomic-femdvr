import numpy as np
import scipy.linalg as la
from typing import Optional

from atomic_femdvr.femdvr import FEDVR_Basis


#===================================================================
def SetPhase(psi):
    """
    Set the phase of the wavefunction to ensure it is positive at the maximum point.
    """
    for i in range(psi.shape[1]):
        idx_max = np.argmax(np.abs(psi[:, i]))
        if psi[idx_max, i] < 0.0:
            psi[:, i] *= -1.0
    return psi
#===================================================================
def SolveSchrodinger(basis:FEDVR_Basis, Veff_grid:np.ndarray, lll:np.ndarray, Dion:np.ndarray,
                     beta_grid:np.ndarray, lmax:int, nmax:int, Vconf: Optional[np.ndarray] = None, lmin=0):
    """
    Solve the radial Schrödinger equation using finite element method
    """
    ne = basis.ne  # Number of elements
    ng = basis.ng  # Number of grid points per element
    nb = ne * ng - 1  # Total number of grid points
    r_grid = basis.GetGridpoints()

    lchannels = np.arange(lmin, lmax + 1, step=1, dtype=int)
    num_channels = len(lchannels)

    psi = np.zeros([num_channels, nmax+1, len(r_grid)], dtype=np.float64)
    eps = np.zeros([num_channels, nmax+1], dtype=np.float64)

    Tmat = basis.KinEn_Matrix_zerobound()
    Veff_mat = np.diag(basis.PotEn_Matrix_grid(Veff_grid))

    if Vconf is not None:
        Vconf_mat = np.diag(basis.PotEn_Matrix_grid(Vconf))
        Veff_mat += Vconf_mat

    for il, l in enumerate(lchannels):
        Vl_grid = np.zeros_like(r_grid)
        if l > 0:
            Vl_grid[1:] = l * (l + 1) / (2. * r_grid[1:]**2)
        Vl_grid[0] = Vl_grid[1]  # Avoid division by zero at r=0
        Vl_mat = np.diag(basis.PotEn_Matrix_grid(Vl_grid))
        H_mat = Tmat + Veff_mat + Vl_mat

        # now add the non-local part
        Ib, = np.where(lll == l)
        nbeta = len(Ib)
        beta_vecs = np.zeros([nbeta, nb], dtype=np.float64)
        for ibeta in range(nbeta):
            beta_vecs[ibeta, :] = basis.GetCoeffs(beta_grid[Ib[ibeta], :], cplx=False)

        Dion_Hr = np.zeros([nbeta, nbeta], dtype=np.float64)
        for i in range(nbeta):
            for j in range(nbeta):
                Dion_Hr[i, j] = 0.5 * Dion[Ib[i], Ib[j]] # Convert to Hartree units

        for ibeta in range(nbeta):
            for jbeta in range(nbeta):
                ket_bra = np.outer(beta_vecs[ibeta], beta_vecs[jbeta])
                H_mat[:, :] += Dion_Hr[ibeta, jbeta] * ket_bra


        eps_l, vect = la.eigh(H_mat, subset_by_index=[0, nmax])
        psi_l = basis.GetPsi_All(vect, cplx=False)
        psi_l = SetPhase(psi_l)

        psi[il, :nmax+1, :] = psi_l.T
        eps[il, :nmax+1] = eps_l[:nmax+1]

    return eps, psi
