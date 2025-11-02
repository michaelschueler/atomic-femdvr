
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, lobpcg

from atomic_femdvr.femdvr import FEDVR_Basis


#========================================================================================================
def set_phase(psi):
    """
    Set the phase of the wavefunction to ensure it is positive at the maximum point.
    """
    for i in range(psi.shape[1]):
        idx_max = np.argmax(np.abs(psi[:, i]))
        if psi[idx_max, i] < 0.0:
            psi[:, i] *= -1.0
    return psi
#========================================================================================================
def solve_schrodinger_pseudo(basis:FEDVR_Basis, Veff_grid:np.ndarray, lll:np.ndarray, Dion:np.ndarray,
                     beta_grid:np.ndarray, lmax:int, nmax:int, Vconf: np.ndarray | None = None, lmin:int=0):
    """
    Solve the radial Schrödinger equation using finite element method
    """
    ne = basis.ne  # Number of elements
    ng = basis.ng  # Number of grid points per element
    nb = ne * ng - 1  # Total number of grid points
    r_grid = basis.get_gridpoints()

    lchannels = np.arange(lmin, lmax + 1, step=1, dtype=int)
    num_channels = len(lchannels)

    psi = np.zeros([num_channels, nmax+1, len(r_grid)], dtype=np.float64)
    eps = np.zeros([num_channels, nmax+1], dtype=np.float64)

    Tmat = basis.get_kinetic_matrix()
    Veff_mat = np.diag(basis.get_potential_from_grid(Veff_grid))

    if Vconf is not None:
        Vconf_mat = np.diag(basis.get_potential_from_grid(Vconf))
        Veff_mat += Vconf_mat

    for il, l in enumerate(lchannels):

        # construct potential including centrifugal term
        Vl_grid = get_centrifugal_potential

        # construct Hamiltonian matrix
        Vl_mat = np.diag(basis.get_potential_from_grid(Vl_grid))
        H_mat = Tmat + Veff_mat + Vl_mat

        # now add the non-local part
        Ib, = np.where(lll == l)
        nbeta = len(Ib)

        # construct FEDVR representations of beta functions
        beta_vecs = np.zeros([nbeta, nb], dtype=np.float64)
        for ibeta in range(nbeta):
            beta_vecs[ibeta, :] = basis.get_coeffs(beta_grid[Ib[ibeta], :], cplx=False)


        # construct Dion in Hartree units for the current l-channel
        Dion_Hr = np.zeros([nbeta, nbeta], dtype=np.float64)
        for i in range(nbeta):
            for j in range(nbeta):
                Dion_Hr[i, j] = 0.5 * Dion[Ib[i], Ib[j]] # Convert to Hartree units

        # add the non-local part
        for ibeta in range(nbeta):
            for jbeta in range(nbeta):
                ket_bra = np.outer(beta_vecs[ibeta], beta_vecs[jbeta])
                H_mat[:, :] += Dion_Hr[ibeta, jbeta] * ket_bra

        eps_l, vect = la.eigh(H_mat, subset_by_index=[0, nmax])
        psi_l = basis.get_psi_all(vect, cplx=False)
        psi_l = set_phase(psi_l)

        psi[il, :nmax+1, :] = psi_l.T
        eps[il, :nmax+1] = eps_l[:nmax+1]

    return eps, psi
#========================================================================================================
def solve_schrodinger_local(basis:FEDVR_Basis, Veff_grid:np.ndarray, lmax:int, nmax:int, 
                           Vconf: np.ndarray | None = None, lmin:int=0, 
                           solver: str = 'full') -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the radial Schrödinger equation using finite element method
    """
    ne = basis.ne  # Number of elements
    ng = basis.ng  # Number of grid points per element
    nb = ne * ng - 1  # Total number of grid points
    r_grid = basis.get_gridpoints()

    lchannels = np.arange(lmin, lmax + 1, step=1, dtype=int)
    num_channels = len(lchannels)

    psi = np.zeros([num_channels, nmax+1, len(r_grid)], dtype=np.float64)
    eps = np.zeros([num_channels, nmax+1], dtype=np.float64)

    Veff_diag = basis.get_potential_from_grid(Veff_grid)

    if Vconf is not None:
        Vconf_diag = basis.get_potential_from_grid(Vconf)
        Veff_diag += Vconf_diag

    if solver.lower() == 'full':
        Tmat = basis.get_kinetic_energy_matrix()
    else:
        Tmat_banded = basis.get_kinetic_energy_banded()

    for il, l in enumerate(lchannels):

        # construct potential including centrifugal term
        Vl_grid = get_centrifugal_potential(r_grid, l)
        Vl_diag = basis.get_potential_from_grid(Vl_grid)

        # construct Hamiltonian matrix
        if solver.lower() == 'full':
            H_mat = Tmat + np.diag(Veff_diag + Vl_diag)
            eps_l, vect = la.eigh(H_mat, subset_by_index=[0, nmax])
        else:
            H_mat_banded = Tmat_banded.copy()
            H_mat_banded[-1, :] += Veff_diag + Vl_diag
            eps_l, vect = la.eig_banded(H_mat_banded, lower=False, select='i',
                                         select_range=[0, nmax])


        psi_l = basis.get_psi_all(vect, cplx=False)
        psi_l = set_phase(psi_l)

        psi[il, :nmax+1, :] = psi_l.T
        eps[il, :nmax+1] = eps_l[:nmax+1]

    return eps, psi
#========================================================================================================
# def solve_scalar_relativistic(basis:FEDVR_Basis, Vpot_fnc:callable, gradVpot_fnc:callable, lmax:int, nmax:int,
#                               eps_guess:np.ndarray, lmin:int=0, maxiter:int=100, tol:float=1.0e-6) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Solve the scalar-relativistic radial Schrödinger equation using finite element method
#     """
#     ne = basis.ne  # Number of elements
#     ng = basis.ng  # Number of grid points per element
#     nb = ne * ng - 1  # Total number of grid points
#     r_grid = basis.get_gridpoints()

#     lchannels = np.arange(lmin, lmax + 1, step=1, dtype=int)
#     num_channels = len(lchannels)

#     psi = np.zeros([num_channels, nmax, len(r_grid)], dtype=np.float64)
#     eps_SR = np.zeros([num_channels, nmax], dtype=np.float64)

#     Tmat = basis.get_kinetic_matrix()
#     Vvec = basis.get_potential_from_grid(Vpot_fnc(r_grid))
#     V_mat = np.diag(Vvec)

#     r_fnc = lambda r: 1. / r
#     r_vec = basis.get_potential_from_grid(r_fnc)
    
#     Dmat = basis.get_derivative_matrix()

#     for il, l in enumerate(lchannels):

#         kappa = l * (l + 1)

#         for istate in range(nmax):
#             eps = eps_guess[il, istate]

#             for it in range(maxiter):
#                 eps_old = eps

#                 M_inv_vec = basis.get_potential_from_grid(lambda r: 1.0 / (1.0 + (eps + Vpot_fnc(r)) / (2.0 * c**2)))
#                 M_inv_mat = np.diag(M_inv_vec)
#                 lM_vec = basis.get_potential_from_grid(lambda r: (eps + Vpot_fnc(r)) / (2.0 * c**2))
#                 V_mat_current = np.diag(lM_vec + Vvec)

#                 B_vec = basis.get_potential_from_grid(lambda r: (dVdr_fnc(r)) / (4.0 * c**2))
#                 B_mat = np.diag(B_vec)

#                 H_mat = M_inv_mat @ Tmat + V_mat_current -

#========================================================================================================\
def get_centrifugal_potential(r_grid:np.ndarray, l:int) -> np.ndarray:
    """
    Get the centrifugal potential for a given angular momentum quantum number l.
    """
    Vl_grid = np.zeros_like(r_grid)
    if l > 0:
        Vl_grid[1:] = l * (l + 1) / (2. * r_grid[1:]**2)
    Vl_grid[0] = Vl_grid[1]  # Avoid division by zero at r=0
    return Vl_grid
#========================================================================================================