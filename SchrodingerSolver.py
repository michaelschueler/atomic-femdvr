import numpy as np
import scipy.linalg as la
from femdvr import FEDVR_Basis
#------------------------------------------------------------
def SetPhase(psi):
    """
    Set the phase of the wavefunction to ensure it is positive at the maximum point.
    """
    for i in range(psi.shape[1]):
        idx_max = np.argmax(np.abs(psi[:, i]))
        if psi[idx_max, i] < 0.0:
            psi[:, i] *= -1.0
    return psi
#------------------------------------------------------------
def SolveNR(r_elements, Vpot_fnc, l, nr, ng):
    """
    Solve the radial Schrödinger equation using finite element method
    """

    ne = len(r_elements) - 1  # Number of elements
    fe = FEDVR_Basis(ne, ng, r_elements)

    r_grid = fe.GetGridpoints()

    Vl_fnc = lambda r: Vpot_fnc(r) + l * (l + 1) / (2. * r**2)

    Vvec = fe.PotEn_Matrix(Vl_fnc)

    Tmat = fe.KinEn_Matrix_zerobound()
    Hmat = Tmat + np.diag(Vvec)

    eps, vect = la.eigh(Hmat, subset_by_index=[0, nr - 1])

    psi = fe.GetPsi_All(vect, cplx=False)
    psi = SetPhase(psi)

    return eps, r_grid, psi.T
#------------------------------------------------------------
def SolveZORA(r_elements, Vpot_fnc, gradVpot_fnc, l, nr, ng):
    """
    Solve the radial Schrödinger equation using ZORA (Zeroth Order Regular Approximation)
    """

    c = 137.035999074  # Fine structure constant

    ne = len(r_elements) - 1  # Number of elements
    fe = FEDVR_Basis(ne, ng, r_elements)

    r_grid = fe.GetGridpoints()

    al = 1. / (2 * c**2)  # Fine structure constant squared
    # al *= 0.1

    kp_fnc = lambda r: 1. / (1 - al * Vpot_fnc(r))
    kp_der_fnc = lambda r: al * gradVpot_fnc(r) / (1 - al * Vpot_fnc(r))**2
    kp_l_fnc = lambda r: kp_fnc(r) * l * (l + 1) / (2. * r**2)
    Vl_fnc = lambda r: Vpot_fnc(r)
    r_fnc = lambda r: 1. / r

    V_vec = fe.PotEn_Matrix(Vl_fnc)
    kp_vec = fe.PotEn_Matrix(kp_fnc)
    kp_mat = np.diag(kp_vec)
    kp_der_vec = fe.PotEn_Matrix(kp_der_fnc)
    kp_der_mat = np.diag(kp_der_vec)
    kp_l_vec = fe.PotEn_Matrix(kp_l_fnc)
    kp_l_mat = np.diag(kp_l_vec)
    r_vec = fe.PotEn_Matrix(r_fnc)
    r_mat = np.diag(r_vec)

    # non-relativistic kinetic energy matrix
    Kmat = fe.KinEn_Matrix_zerobound() 

    # first derivative matrix
    Dmat = fe.GetDeriv_Matrix_zerobound() 
    # Kmat = -0.5 * Dmat @ Dmat

    # relativistic kinetic energy matrix
    Tmat = kp_mat @ Kmat + kp_l_mat - 0.5 * kp_der_mat @ (Dmat - r_mat)
    # Tmat = -0.5 * Dmat @ kp_mat @ Dmat
    Tmat = 0.5 * (Tmat + Tmat.T)  # Ensure symmetry
    # Tmat = -0.5 * Dmat @ kp_mat @ Dmat

    Hmat = Tmat + np.diag(V_vec)

    eps, vect = la.eigh(Hmat, subset_by_index=[0, nr - 1])

    psi = fe.GetPsi_All(vect, cplx=False)
    psi = SetPhase(psi)

    return eps, r_grid, psi.T
#------------------------------------------------------------
def SolveSR(r_elements, Vpot_fnc, gradVpot_fnc, l, eps_guess, ng, maxiter=10, tol=1.0e-6):

    # We solve the radial Schrödinger equation in Rydberg atomic units. 
    # The potential is assumed to be in Hartree energy units.

    c = 137.035999074  # Fine structure constant
    alpha = 1. / c  # Fine structure constant
    kappa = -1. # Kappa value for the radial equation

    nr = len(eps_guess)  # Number of radial functions to solve for

    ne = len(r_elements) - 1  # Number of elements
    fe = FEDVR_Basis(ne, ng, r_elements)
    r_grid = fe.GetGridpoints()

    nb = ne*ng+1
    psi = np.zeros((nb, nr), dtype=np.float64)
    
    M_inv_fnc = lambda r, E: 1./(1. - 0.5 * alpha**2 * (Vpot_fnc(r) - E))
    Bterm_fnc = lambda r, E: 0.5 * alpha**2 * M_inv_fnc(r, E)**2 * gradVpot_fnc(r)
    lM_fnc = lambda r, E: l * (l + 1) / (2. * r**2) * M_inv_fnc(r, E)
    r_fnc = lambda r: 1. / r

    Kmat = fe.KinEn_Matrix_zerobound()
    Dmat = fe.GetDeriv_Matrix_zerobound()
    r_vec = fe.PotEn_Matrix(r_fnc)
    V_vec = fe.PotEn_Matrix(Vpot_fnc)

    eps_SR = np.zeros(nr, dtype=np.float64)

    for istate in range(nr):
        eps = eps_guess[istate]

        for it in range(maxiter):
            eps_old = eps

            M_inv_vec = fe.PotEn_Matrix(lambda r: M_inv_fnc(r, eps))
            M_inv_mat = np.diag(M_inv_vec)
            lM_vec = fe.PotEn_Matrix(lambda r: lM_fnc(r, eps))
            V_mat = np.diag(lM_vec + V_vec)

            B_vec = fe.PotEn_Matrix(lambda r: Bterm_fnc(r, eps))
            B_mat = np.diag(B_vec)

            H_mat = M_inv_mat @ Kmat + V_mat - 0.5 * B_mat @ (Dmat + kappa * np.diag(r_vec))
            
            evals, vect = la.eigh(H_mat, subset_by_index=[0, istate])
            eps = evals[istate]

            if np.abs(eps - eps_old) < tol:
                break

        if it == maxiter - 1:
            print(f"Warning: Maximum iterations reached for state {istate+1} with energy {eps:.6f}")

        eps_SR[istate] = eps
        psi[:, istate] = fe.GetPsi(vect[:, istate], cplx=False)

    psi = SetPhase(psi)

    return eps_SR, r_grid, psi.T
#------------------------------------------------------------

#------------------------------------------------------------
def SolvePseudo(r_elements, Vpot_fnc, Dion, beta_fnc, l, nr, ng):
    """
    Solve the radial Schrödinger equation using finite element method
    """

    ne = len(r_elements) - 1  # Number of elements
    nb = ne * ng - 1  # Total number of grid points
    fe = FEDVR_Basis(ne, ng, r_elements)

    r_grid = fe.GetGridpoints()

    Vl_fnc = lambda r: Vpot_fnc(r) + l * (l + 1) / (2. * r**2)

    Vvec = fe.PotEn_Matrix(Vl_fnc)

    Tmat = fe.KinEn_Matrix_zerobound()
    Hmat = Tmat + np.diag(Vvec)

    # now add the non-local part
    nbeta = Dion.shape[0]

    # beta_vecs = fe.Get_coeffs_batch(nbeta, beta_fnc)
    beta_grid = beta_fnc(r_grid)
    beta_vecs = np.zeros([nbeta, nb], dtype=np.float64)
    for ibeta in range(nbeta):
        beta_vecs[ibeta, :] = fe.GetCoeffs(beta_grid[ibeta, :], cplx=False)

    Vnl_mat = np.zeros([nb, nb], dtype=np.float64)
    for ibeta in range(nbeta):
        for jbeta in range(nbeta):
            ket_bra = np.outer(beta_vecs[ibeta], beta_vecs[jbeta])
            Vnl_mat[:, :] += Dion[ibeta, jbeta] * ket_bra

    Hmat += Vnl_mat

    eps, vect = la.eigh(Hmat, subset_by_index=[0, nr - 1])

    psi = fe.GetPsi_All(vect, cplx=False)
    psi = SetPhase(psi)

    return eps, r_grid, psi.T
#------------------------------------------------------------