"""
Radial scattering solvers and phase-shift extraction for photoemission calculations.

Asymptotic BC
-------------
The FEM-DVR asymptotic BC encodes the condition

    u_l(r_N) = alpha * u_l(r_{N-1}) + beta

at the last two grid points r_{N-1} and r_N.  The asymptotic solution is

    u_l(r) = r * [h^+_l(kr) + S_l * h^-_l(kr)],   S_l = exp(2i*delta_l)

where h^+_l = j_l + i*y_l (outgoing) and h^-_l = j_l - i*y_l (incoming).

alpha is chosen to be transparent to the outgoing Hankel component:

    alpha = u_out(r_N) / u_out(r_{N-1}),   u_out(r) = r * h^+_l(kr)

beta drives the incoming Hankel component with amplitude cn:

    beta = cn * [u_in(r_N) - alpha * u_in(r_{N-1})],   u_in(r) = r * h^-_l(kr)

Using the exact Hankel functions (via _hankel_bc) makes the BC correct at any r,
not just in the large-r plane-wave limit.  For V=0 with cn=1 the interior solution
converges to u_l(r) = 2*r*j_l(kr) (the regular free-particle reduced wavefunction).
"""

import numpy as np
from scipy.special import erf, spherical_jn, spherical_yn

from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.kohn_sham import get_centrifugal_potential

_C_LIGHT = 137.035999084  # speed of light in atomic units


def _hankel_bc(l: int, k: float, r_last: float, r_next: float,
               cn: float = 1.0) -> tuple[complex, complex]:
    """
    Compute exact Hankel boundary-condition parameters (alpha, beta).

    alpha is the ratio of the outgoing reduced Hankel wavefunction at r_next
    to that at r_last; it is transparent to the outgoing component:

        u_out(r) = r * h^+_l(kr) = r * (j_l(kr) + i*y_l(kr))
        alpha = u_out(r_next) / u_out(r_last)

    beta drives the incoming Hankel component with amplitude cn:

        u_in(r) = r * h^-_l(kr) = r * (j_l(kr) - i*y_l(kr))
        beta = cn * (u_in(r_next) - alpha * u_in(r_last))

    For r >> l/k this reduces to alpha -> exp(ik*h) and
    beta -> cn*(i)^l * 2*sin(kh)*exp(-ikr_last)/k.
    """
    u_out_last = r_last * (spherical_jn(l, k * r_last) + 1j * spherical_yn(l, k * r_last))
    u_out_next = r_next * (spherical_jn(l, k * r_next) + 1j * spherical_yn(l, k * r_next))
    u_in_last  = r_last * (spherical_jn(l, k * r_last) - 1j * spherical_yn(l, k * r_last))
    u_in_next  = r_next * (spherical_jn(l, k * r_next) - 1j * spherical_yn(l, k * r_next))

    alpha = u_out_next / u_out_last
    beta  = cn * (u_in_next - alpha * u_in_last)
    return alpha, beta


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
    alpha, beta = _hankel_bc(l, k, r_grid[-2], r_grid[-1], cn)

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
    alpha, beta_bc = _hankel_bc(l, k, r_grid[-2], r_grid[-1], cn)

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


def solve_scattering_zora(basis: FEDVR_Basis, Veff_grid: np.ndarray, k: float, l: int,
                          Z: float, nuclear_sigma: float,
                          cn: float = 1.0) -> np.ndarray:
    """
    Solve the radial scattering equation under scalar ZORA.

    Uses model ZORA: M(r) = 1/(1 - V_nuc_smooth/(2c²)) with a Gaussian-smoothed
    nuclear potential. Since M(Rmax) ≈ 1, the outgoing-wave BC is grafted from the
    NR kinetic energy matrix (T_rel_bc = T_rel_0 + T_NR_bc - T_NR_0).

    The centrifugal term is M-weighted: l(l+1)*M(r)/(2r²).

    Parameters
    ----------
    Z : float
        Nuclear charge.
    nuclear_sigma : float
        Gaussian smearing width for the nuclear potential (Bohr).
    """
    Ek = 0.5 * k**2
    r_grid = basis.get_gridpoints()

    sig2 = np.sqrt(2.0) * nuclear_sigma
    V_nuc_smooth = np.empty_like(r_grid)
    V_nuc_smooth[1:] = -Z * erf(r_grid[1:] / sig2) / r_grid[1:]
    V_nuc_smooth[0] = -Z * np.sqrt(2.0 / np.pi) / nuclear_sigma
    M_grid = 1.0 / (1.0 - V_nuc_smooth / (2.0 * _C_LIGHT**2))

    alpha, beta = _hankel_bc(l, k, r_grid[-2], r_grid[-1], cn)

    T_rel = basis.get_p_kinetic_matrix(M_grid)
    T_NR_bc, Rvec = basis.get_kinetic_energy_matrix(alpha, beta)
    T_NR_0 = basis.get_kinetic_energy_matrix(0.0, 0.0)
    T_mat = T_rel.astype(np.complex128) + (T_NR_bc - T_NR_0)

    Veff_diag = basis.get_potential_from_grid(Veff_grid)
    Vl_grid = np.zeros_like(r_grid)
    if l > 0:
        Vl_grid[1:] = l * (l + 1) * M_grid[1:] / (2.0 * r_grid[1:]**2)
    Vl_diag = basis.get_potential_from_grid(Vl_grid)

    nb = T_mat.shape[0]
    H_mat = (T_mat
             + np.diag((Veff_diag + Vl_diag).astype(np.complex128))
             - Ek * np.eye(nb, dtype=np.complex128))

    cff = np.linalg.solve(H_mat, Rvec)
    psi = basis.get_psi(cff, cplx=True)
    psi[-1] = alpha * psi[-2] + beta
    return psi


def solve_scattering_kh(basis: FEDVR_Basis, Veff_grid: np.ndarray, k: float, l: int,
                         Z: float, nuclear_sigma: float,
                         cn: float = 1.0) -> np.ndarray:
    """
    Solve the radial scattering equation under scalar Koelling-Harmon.

    The KH mass at scattering energy Ek is M_inv(r) = 1/(1 + (Ek - V_eff_smooth)/(2c²)).
    For the asymptotic BC the NR wave vector k = sqrt(2*Ek) is used; the correction
    k_KH/k_NR - 1 = Ek/(4c²) is negligible for typical photoemission energies.

    The Veff used for M_inv is regularised near the nucleus identically to the
    bound-state KH solver: Veff_for_M = Veff_grid + (V_nuc_smooth - V_nuc_point).

    The centrifugal term is M_inv-weighted: l(l+1)*M_inv(r)/(2r²).
    """
    Ek = 0.5 * k**2
    r_grid = basis.get_gridpoints()

    sig2 = np.sqrt(2.0) * nuclear_sigma
    V_nuc_point = np.empty_like(r_grid)
    V_nuc_point[1:] = -Z / r_grid[1:]
    V_nuc_point[0] = V_nuc_point[1]
    V_nuc_smooth = np.empty_like(r_grid)
    V_nuc_smooth[1:] = -Z * erf(r_grid[1:] / sig2) / r_grid[1:]
    V_nuc_smooth[0] = -Z * np.sqrt(2.0 / np.pi) / nuclear_sigma
    Veff_for_M = Veff_grid + (V_nuc_smooth - V_nuc_point)
    M_inv_grid = 1.0 / (1.0 + (Ek - Veff_for_M) / (2.0 * _C_LIGHT**2))

    alpha, beta = _hankel_bc(l, k, r_grid[-2], r_grid[-1], cn)

    T_rel = basis.get_p_kinetic_matrix(M_inv_grid)
    T_NR_bc, Rvec = basis.get_kinetic_energy_matrix(alpha, beta)
    T_NR_0 = basis.get_kinetic_energy_matrix(0.0, 0.0)
    T_mat = T_rel.astype(np.complex128) + (T_NR_bc - T_NR_0)

    Veff_diag = basis.get_potential_from_grid(Veff_grid)
    Vl_grid = np.zeros_like(r_grid)
    if l > 0:
        Vl_grid[1:] = l * (l + 1) * M_inv_grid[1:] / (2.0 * r_grid[1:]**2)
    Vl_diag = basis.get_potential_from_grid(Vl_grid)

    nb = T_mat.shape[0]
    H_mat = (T_mat
             + np.diag((Veff_diag + Vl_diag).astype(np.complex128))
             - Ek * np.eye(nb, dtype=np.complex128))

    cff = np.linalg.solve(H_mat, Rvec)
    psi = basis.get_psi(cff, cplx=True)
    psi[-1] = alpha * psi[-2] + beta
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
