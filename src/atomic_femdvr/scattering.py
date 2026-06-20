"""
Radial scattering solvers and phase-shift extraction for photoemission calculations.

Asymptotic BC consistency
--------------------------
The FEM-DVR asymptotic BC imposes, at the last two grid points r_{N-1} and r_N = r_{N-1}+h:

    u_l(r_N) = alpha * u_l(r_{N-1}) + beta
    alpha = exp(ikh),  beta = 2i*cn*exp(-ikr_{N-1})*sin(kh)

At large r, the outgoing Hankel function h^+_l(kr) ~ (-i)^{l+1}*exp(ikr)/(kr), so
r*h^+_l(kr) ~ (-i)^{l+1}/k * exp(ikr). The ratio for consecutive grid points is:

    r_N*h^+_l(kr_N) / [r_{N-1}*h^+_l(kr_{N-1})] -> exp(ikh)   (for r >> l/k)

The outgoing component is therefore transparent to alpha = exp(ikh). For the incoming
component r*h^-_l ~ (i)^{l+1}/k * exp(-ikr):

    r_N*h^-_l(kr_N) - alpha*r_{N-1}*h^-_l(kr_{N-1})
    ≈ (i)^{l+1}/k * exp(-ikr_{N-1}) * (-2i*sin(kh))

So beta encodes the INCOMING wave amplitude only. For V=0 with cn=1, the solution is
u_l(r) = cn*(exp(ikr) - exp(-ikr)), which for l=0 is 2i*sin(kr). For general l with
potential V, the solution asymptotically satisfies:

    psi_l(r) = h^+_l(kr) + S_l * h^-_l(kr),  S_l = exp(2i*delta_l)

and delta_l is extracted via extract_phase().
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.kohn_sham import get_centrifugal_potential


def solve_scattering_local(basis: FEDVR_Basis, Veff_grid: np.ndarray, k: float, l: int,
                            cn: float = 1.0, Vconf: np.ndarray | None = None) -> np.ndarray:
    """
    Solve the radial scattering equation for a local potential.

    Parameters
    ----------
    basis : FEDVR_Basis
    Veff_grid : np.ndarray
        Effective local potential on the FEM-DVR grid (Hartree).
    k : float
        Wave vector, k = sqrt(2*Ek) in atomic units.
    l : int
        Angular momentum quantum number of the final state.
    cn : float
        Amplitude of the driving incoming wave at the boundary.
    Vconf : np.ndarray | None
        Optional confinement potential.

    Returns
    -------
    psi : np.ndarray, complex, shape (ngrid,)
        Reduced radial wavefunction u_l(r) on the full FEM-DVR grid.
    """
    Ek = 0.5 * k**2
    r_grid = basis.get_gridpoints()
    h = r_grid[-1] - r_grid[-2]
    r_last = r_grid[-2]

    alpha = np.exp(1j * k * h)
    beta = 2j * cn * np.exp(-1j * k * r_last) * np.sin(k * h)

    Tmat, Rvec = basis.get_kinetic_energy_matrix(alpha, beta)

    Veff_diag = basis.get_potential_from_grid(Veff_grid)
    if Vconf is not None:
        Veff_diag += basis.get_potential_from_grid(Vconf)

    Vl_diag = basis.get_potential_from_grid(get_centrifugal_potential(r_grid, l))

    nb = Tmat.shape[0]
    H_mat = (Tmat
             + np.diag((Veff_diag + Vl_diag).astype(np.complex128))
             - Ek * np.eye(nb, dtype=np.complex128))

    cff = np.linalg.solve(H_mat, Rvec)

    psi = basis.get_psi(cff, cplx=True)
    psi[-1] = alpha * psi[-2] + beta
    return psi


def solve_scattering_nonlocal(basis: FEDVR_Basis, Veff_grid: np.ndarray, k: float, l: int,
                               lll: np.ndarray, Dion: np.ndarray, beta_pp: np.ndarray,
                               cn: float = 1.0, Vconf: np.ndarray | None = None) -> np.ndarray:
    """
    Solve the radial scattering equation with a non-local pseudopotential.

    Adds Kleinman-Bylander non-local projectors for angular momentum channel l.
    All other parameters are the same as solve_scattering_local.

    Parameters
    ----------
    lll : np.ndarray
        Angular momentum indices of PP projectors (length nbeta).
    Dion : np.ndarray
        D_ion matrix (KB coefficients) in Rydberg units, shape (nbeta, nbeta).
    beta_pp : np.ndarray
        Beta projector functions on the FEM-DVR grid, shape (nbeta, ngrid).
    """
    Ek = 0.5 * k**2
    ne = basis.ne
    ng = basis.ng
    nb = ne * ng - 1

    r_grid = basis.get_gridpoints()
    h = r_grid[-1] - r_grid[-2]
    r_last = r_grid[-2]

    alpha = np.exp(1j * k * h)
    beta_bc = 2j * cn * np.exp(-1j * k * r_last) * np.sin(k * h)

    Tmat, Rvec = basis.get_kinetic_energy_matrix(alpha, beta_bc)

    Veff_diag = basis.get_potential_from_grid(Veff_grid)
    if Vconf is not None:
        Veff_diag += basis.get_potential_from_grid(Vconf)

    Vl_diag = basis.get_potential_from_grid(get_centrifugal_potential(r_grid, l))

    H_mat = (Tmat
             + np.diag((Veff_diag + Vl_diag).astype(np.complex128))
             - Ek * np.eye(nb, dtype=np.complex128))

    Ib, = np.where(lll == l)
    if len(Ib) > 0:
        beta_vecs = np.zeros([len(Ib), nb], dtype=np.float64)
        for ibeta, ib in enumerate(Ib):
            beta_vecs[ibeta, :] = basis.get_coeffs(beta_pp[ib, :], cplx=False)

        Dion_Hr = 0.5 * Dion[np.ix_(Ib, Ib)]  # convert Ry -> Ha

        for ibeta in range(len(Ib)):
            for jbeta in range(len(Ib)):
                H_mat[:nb, :nb] += Dion_Hr[ibeta, jbeta] * np.outer(beta_vecs[ibeta],
                                                                       beta_vecs[jbeta])

    cff = np.linalg.solve(H_mat, Rvec)

    psi = basis.get_psi(cff, cplx=True)
    psi[-1] = alpha * psi[-2] + beta_bc
    return psi


def extract_phase(psi: np.ndarray, r_grid: np.ndarray, k: float,
                  l: int) -> tuple[complex, float]:
    """
    Extract S_l = exp(2i*delta_l) and phase shift delta_l from a scattering wavefunction.

    Uses the last two interior grid points (r_a = r_grid[-3], r_b = r_grid[-2]) and
    expands the reduced radial wavefunction in spherical Bessel functions:

        u_l(r) = c_j * kr*j_l(kr) + c_y * kr*y_l(kr)

    Decomposing into Hankel functions h^± = j_l ± i*y_l:

        c_+ = (c_j - i*c_y) / 2  [amplitude of h^+_l]
        c_- = (c_j + i*c_y) / 2  [amplitude of h^-_l]
        S_l = c_- / c_+

    Parameters
    ----------
    psi : np.ndarray (complex)
        Reduced radial wavefunction on the FEM-DVR grid (including boundary point).
    r_grid : np.ndarray
        FEM-DVR grid points.
    k : float
        Wave vector.
    l : int
        Angular momentum quantum number.

    Returns
    -------
    S_l : complex
        S-matrix element exp(2i*delta_l).
    delta_l : float
        Phase shift in radians, delta_l = arg(S_l) / 2.
    """
    r_a, r_b = r_grid[-3], r_grid[-2]
    u_a, u_b = psi[-3], psi[-2]

    u_j = np.array([k * r_a * spherical_jn(l, k * r_a),
                    k * r_b * spherical_jn(l, k * r_b)])
    u_y = np.array([k * r_a * spherical_yn(l, k * r_a),
                    k * r_b * spherical_yn(l, k * r_b)])

    mat = np.column_stack([u_j, u_y]).astype(np.complex128)
    c_j, c_y = np.linalg.solve(mat, np.array([u_a, u_b]))

    c_plus = c_j - 1j * c_y
    c_minus = c_j + 1j * c_y
    S_l = c_minus / c_plus
    delta_l = 0.5 * np.angle(S_l)

    return S_l, delta_l
