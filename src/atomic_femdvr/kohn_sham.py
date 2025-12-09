
import numpy as np
import scipy.linalg as la

from atomic_femdvr.femdvr import FEDVR_Basis


#========================================================================================================
def set_phase(psi):
    """
    Set the phase of the wavefunction to ensure it is positive at the maximum point.
    """
    for i in range(psi.shape[0]):
        idx_max = np.argmax(np.abs(psi[i, :]))
        if psi[i, idx_max] < 0.0:
            psi[i, :] *= -1.0
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

    Tmat = basis.get_kinetic_energy_matrix()
    Veff_mat = np.diag(basis.get_potential_from_grid(Veff_grid))

    if Vconf is not None:
        Vconf_mat = np.diag(basis.get_potential_from_grid(Vconf))
        Veff_mat += Vconf_mat

    for il, l in enumerate(lchannels):

        # construct potential including centrifugal term
        Vl_grid = get_centrifugal_potential(r_grid, l)

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
        vect_T = np.ascontiguousarray(vect.T)
        psi_l = basis.get_psi(vect_T, cplx=False)
        psi_l = set_phase(psi_l)

        psi[il, :nmax+1, :] = psi_l
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
            

        vect_T = np.ascontiguousarray(vect.T)
        psi_l = basis.get_psi(vect_T, cplx=False)
        psi_l = set_phase(psi_l)

        psi[il, :nmax+1, :] = psi_l
        eps[il, :nmax+1] = eps_l[:nmax+1]

    return eps, psi
#========================================================================================================
def solve_scalar_relativistic(basis:FEDVR_Basis, Veff_grid:np.ndarray, lmax:int, nmax:int,
                              Vconf: np.ndarray | None = None, lmin:int=0, 
                              maxiter:int=100, tol:float=1.0e-6) -> tuple[np.ndarray, np.ndarray]:

    c = 137.035999074  # Fine structure constant
    alpha = 1. / c  # Fine structure constant
    kappa = -1. # Kappa value for the radial equation

    ne = basis.ne  # Number of elements
    ng = basis.ng  # Number of grid points per element
    nb = ne * ng - 1  # Total number of grid points
    r_grid = basis.get_gridpoints()
    r_grid[0] = 1.0e-10  # avoid division by zero

    lchannels = np.arange(lmin, lmax + 1, step=1, dtype=int)
    num_channels = len(lchannels)

    Veff_diag = basis.get_potential_from_grid(Veff_grid)

    if Vconf is not None:
        Vconf_diag = basis.get_potential_from_grid(Vconf)
        Veff_diag += Vconf_diag

    # derivative of potential
    dVeff_dr = basis.get_grid_derivative(Veff_grid)

    Tmat = basis.get_kinetic_energy_matrix()
    Dmat = basis.get_deriv_matrix()

    one_over_r = 1. / r_grid
    one_over_r_mat = np.diag(basis.get_potential_from_grid(one_over_r))
    Dterm_mat = Dmat + kappa * one_over_r_mat

    psi = np.zeros([num_channels, nmax+1, len(r_grid)], dtype=np.float64)
    eps = np.zeros([num_channels, nmax+1], dtype=np.float64)

    for il, l in enumerate(lchannels):

        # non-relativistic Hamiltonian matrix
        Vl_grid = get_centrifugal_potential(r_grid, l)
        Vl_diag = basis.get_potential_from_grid(Vl_grid)
        H0_mat = Tmat + np.diag(Veff_diag + Vl_diag)

        eps_l, vect = la.eigh(H0_mat, subset_by_index=[0, nmax])
        vect_T = np.ascontiguousarray(vect.T)

        for istate in range(nmax+1):
            eps_curr = eps_l[istate]
            y0 = vect_T[istate, :].copy()
            y = y0.copy()
            dlt = np.zeros(nb)

            err = 1.0
            it = 0

            # while err > tol and it < maxiter:
            #     it += 1

            #     eps_old = eps_curr

            #     M_inv_grid = 1./(1. - 0.5 * alpha**2 * (Veff_grid - eps_curr))
            #     Bterm_grid = 0.5 * alpha**2 * M_inv_grid**2 * dVeff_dr
            #     lM_grid = l * (l + 1) / (2. * r_grid**2) * M_inv_grid

            #     M_inv_diag = basis.get_potential_from_grid(M_inv_grid)
            #     lM_diag = basis.get_potential_from_grid(lM_grid)
            #     Bterm_diag = basis.get_potential_from_grid(Bterm_grid)

            #     H_mat = np.einsum('i, ij -> ij', M_inv_diag, Tmat)
            #     H_mat += np.diag(Veff_diag + lM_diag)
            #     H_mat -= 0.5 * np.einsum('i, ij -> ij', Bterm_diag, Dterm_mat)

            #     deltaH = H_mat - H0_mat
            #     deltaH *= 0.

            #     # right-hand side
            #     rhs = np.dot(deltaH, y) + (eps_l[istate] - eps_curr) * y0

            #     # coefficient matrix
            #     coeff_mat = eps_curr * np.eye(nb) - H0_mat
            #     # G = np.linalg.inv(coeff_mat)

            #     # solve for the correction to the wavefunction
            #     # dlt = np.dot(G, rhs)
            #     dlt = la.solve(coeff_mat, rhs)
            #     y = y0 + dlt
            #     norm = np.sqrt( np.dot(y, y) )
            #     y /= norm
            #     # vect_new = la.solve(coeff_mat, rhs)
            #     # vect_new = np.dot(G, rhs)
            #     # norm = np.sqrt( np.dot(vect_new, vect_new) )
            #     # vect_new /= norm

            #     # eps_curr = np.dot(vect_new, np.dot(H_mat, vect_new))
            #     eps_curr = np.dot(y, np.dot(H_mat, y))
            #     err = np.abs(eps_curr - eps_old)

            #     print(f"l={l} state={istate} iter={it} eps={eps_curr:.8f} err={err:.2e}")

            #     vect_T[istate, :] = y

            eps[il, istate] = eps_curr

        psi_l = basis.get_psi(vect_T, cplx=False)
        psi[il, :, :] = set_phase(psi_l)

        exit()

    return eps, psi

#========================================================================================================
def get_green_function(basis:FEDVR_Basis, Tmat:np.ndarray, Dmat:np.ndarray, Veff_grid:np.ndarray,
                       l:int, eps:float) -> np.ndarray:
    
        c = 137.035999074  # Fine structure constant
        alpha = 1. / c  # Fine structure constant
        kappa = -1. # Kappa value for the radial equation


        M_inv_grid = 1./(1. - 0.5 * alpha**2 * (Veff_grid - eps))
        Bterm_grid = 0.5 * alpha**2 * M_inv_grid**2 * dVeff_dr
        lM_grid = l * (l + 1) / (2. * r_grid**2) * M_inv_grid

        M_inv_diag = basis.get_potential_from_grid(M_inv_grid)
        lM_diag = basis.get_potential_from_grid(lM_grid)
        Bterm_diag = basis.get_potential_from_grid(Bterm_grid)

        H_mat = np.einsum('i, ij -> ij', M_inv_diag, Tmat)
        H_mat += np.diag(Veff_diag + lM_diag)
        H_mat -= 0.5 * np.einsum('i, ij -> ij', Bterm_diag, Dterm_mat)


#========================================================================================================



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
