import numpy as np

from atomic_femdvr.femdvr import FEDVR_Basis

#========================================================================================================
def gram_schmidt( phi:np.ndarray, nst:int, psi:np.ndarray, normalize:bool) -> np.ndarray:
    """
    Orthogonalize wave-function psi against the wavefunctions phi using the Gram-Schmidt process
    """

    if nst == 0:
        return psi

    psi_ortho = psi.copy()

    for i in range(nst):
        overlap = np.dot(phi[i, :], psi)
        psi_ortho -= overlap * phi[i, :]

    if normalize:
        overlap = np.dot(psi_ortho, psi_ortho)
        norm = np.sqrt(overlap)
        if norm > 1e-14:
            psi_ortho /= norm

    return psi_ortho
#========================================================================================================
def pcg_eigensolver(matvec_H: callable, matvec_P: callable, nst:int, eps0:np.ndarray, 
                    vect0:np.ndarray, maxiter: int = 1000, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the eigenvalue problem H c = e c using the Preconditioned Conjugate Gradient method.
    Allows for energy-dependent Hamiltonians by providing matvec_H and matvec_P functions.
    """

    nb = vect0.shape[1]
    eps_all = np.zeros(nst)
    h_psi = np.zeros(nb)
    cg = np.zeros(nb)
    sd = np.zeros(nb)
    sd_prev = np.zeros(nb)
    sd_precond = np.zeros(nb)
    h_cg = np.zeros(nb)

    vect = vect0.copy()

    for istate in range(nst):
        sd_product_mixed = 0.0

        psi = vect[istate, :].copy()
        eps = eps0[istate]

        # Orthogonalize starting eigenfunctions to those already calculated...
        if istate > 0:
            psi = gram_schmidt(vect[:istate, :], istate, psi, normalize=True)

        # Calculate starting gradient: |hpsi> = H|psi>
        h_psi = matvec_H(psi, eps)

        # Calculates starting eigenvalue: e(p) = <psi(p)|H|psi>
        eps = np.dot(psi, h_psi)

        err = 1.0
        it = 0
        while err > tol and it < maxiter:
            it += 1

            if it > 1:
                sd_prev = sd.copy()
            else:
                sd_prev = np.zeros_like(sd)

            # compute the residual
            sd = -h_psi + eps * psi

            # apply preconditioner
            sd_precond = matvec_P(sd, eps)

            overlap = np.dot(psi, sd_precond)
            sd_precond -= overlap * psi

            if istate > 0:
                sd_precond = gram_schmidt(vect[:istate, :], istate, sd_precond, normalize=False)

            sd_norm = np.sqrt( np.dot(sd_precond, sd_precond) )
            if sd_norm < 1e-14:
                print(f"Warning: Residual norm too small for state {istate+1} at iteration {it}")
                break
            sd_precond /= sd_norm

            sd_product = np.dot(sd_precond, sd)
            if it > 1:
                sd_product_mixed = np.dot(sd_precond, sd_prev)

            if it == 1:
                sd_product_previous = sd_product
            else:
                gmm = (sd_product - sd_product_mixed) / sd_product_previous
                sd_product_previous = sd_product
                cg = sd_precond + gmm * cg

            overlap = np.dot(psi, cg)
            cg = cg - overlap * psi

            # normalize cg 
            cg_norm = np.sqrt( np.dot(cg, cg) )
            if cg_norm < 1e-14:
                print(f"Warning: Conjugate gradient norm too small for state {istate+1} at iteration {it}")
                break
            cg /= cg_norm

            # cg contains now the conjugate gradient
            #  compute H|cg>
            h_cg = matvec_H(cg, eps)
            
            # Line minimization
            b0 = np.dot(cg, h_cg)
            alpha = - eps + b0
            beta = 2.0 * np.dot(cg, h_psi)
            theta = 0.5 * np.arctan2(-beta, alpha)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # This checks whether we are picking the maximum or the minimum.
            theta2 = theta + 0.5 * np.pi
            cos_theta2 = np.cos(theta2)
            sin_theta2 = np.sin(theta2)
            sol1 = eps + sin_theta**2 * alpha + sin_theta * cos_theta * beta
            sol2 = eps + sin_theta2**2 * alpha + sin_theta2 * cos_theta2 * beta

            if sol2 < sol1:
                theta = theta2
                cos_theta = cos_theta2
                sin_theta = sin_theta2

            # Update wavefunction: |psi> = cos(theta) * |psi> + sin(theta) * |cg>
            psi_new = cos_theta * psi + sin_theta * cg

            # update H|psi>: |h_psi> = ctheta * |h_psi> + stheta * |h_cg>
            h_psi_new = cos_theta * h_psi + sin_theta * h_cg

            psi = psi_new.copy()
            h_psi = h_psi_new.copy()

            # new eigenvalue: e = <psi|H|psi>
            eps_new = np.dot(psi, h_psi)

            err = np.abs(eps_new - eps)
            eps = eps_new

        if it == maxiter:
            print(f"Warning: Maximum iterations reached for state {istate+1} with energy {eps:.6f}")

        vect[istate, :] = psi.copy()
        eps_all[istate] = eps

    return eps_all, vect

#========================================================================================================