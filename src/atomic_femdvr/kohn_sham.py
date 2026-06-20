
import numpy as np
import scipy.linalg as la
from scipy.special import erf

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

    T_mat = basis.get_kinetic_energy_matrix()
    Veff_diag = basis.get_potential_from_grid(Veff_grid)

    if Vconf is not None:
        Vconf_diag = basis.get_potential_from_grid(Vconf)
        Veff_diag += Vconf_diag

    for il, l in enumerate(lchannels):

        # construct potential including centrifugal term
        Vl_grid = get_centrifugal_potential(r_grid, l)

        # construct Hamiltonian matrix
        Vl_diag = basis.get_potential_from_grid(Vl_grid)

        V_mat = np.diag(Veff_diag + Vl_diag)
        H_mat = T_mat + V_mat

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
def solve_schrodinger_zora(basis: FEDVR_Basis, Veff_grid: np.ndarray, lmax: int, nmax: int,
                           Z: float = 1.0, nuclear_sigma: float = 1.0e-3,
                           Vconf: np.ndarray | None = None, lmin: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Scalar-relativistic solver using ZORA (Zeroth Order Regular Approximation).

    The ZORA mass factor M(r) = 1/(1 - V(r)/(2c^2)) depends only on the potential,
    so H_ZORA is energy-independent and Hermitian. This is a standard eigenvalue
    problem; eigenstates are exactly orthogonal.

    The kinetic energy -(1/2)d/dr[M d/dr] is assembled element-by-element using
    GL quadrature with M(r) as a weight, exactly as T_NR but with M inserted.
    This correctly handles bridge-point contributions that are missed by the
    naive D^T diag(M) D / 2 formula.

    The centrifugal term picks up the same M factor: l(l+1)*M(r)/(2r^2).

    Model ZORA with Gaussian nuclear model: the ZORA mass uses a Gaussian-smoothed
    nuclear potential V_nuc(r) = -Z/r * erf(r / (sqrt(2) * nuclear_sigma)) instead
    of the singular -Z/r. This keeps M(r) finite and nonzero at r=0, removing the
    pathological loss of kinetic repulsion that otherwise causes spurious core-state
    behaviour. nuclear_sigma is a numerical parameter; any value << 1/Z gives
    negligible error in physical observables.
    Vconf is excluded from M and included only in the diagonal potential.
    """
    c = 137.035999074

    r_grid = basis.get_gridpoints()

    lchannels = np.arange(lmin, lmax + 1, step=1, dtype=int)
    num_channels = len(lchannels)

    psi = np.zeros([num_channels, nmax + 1, len(r_grid)], dtype=np.float64)
    eps = np.zeros([num_channels, nmax + 1], dtype=np.float64)

    # Gaussian-smoothed nuclear potential for the ZORA mass only.
    # V_nuc(r) = -Z/r * erf(r / (sqrt(2)*sigma)); at r=0 this has the analytic limit -Z*sqrt(2/pi)/sigma.
    # Using this in M(r) keeps M finite and nonzero at the origin, regularising
    # the otherwise pathological ZORA behaviour near the nucleus.
    sig2 = np.sqrt(2.0) * nuclear_sigma
    V_nuc_smooth = np.empty_like(r_grid)
    V_nuc_smooth[1:] = -Z * erf(r_grid[1:] / sig2) / r_grid[1:]
    V_nuc_smooth[0] = -Z * np.sqrt(2.0 / np.pi) / nuclear_sigma  # analytic r->0 limit
    M_grid = 1.0 / (1.0 - V_nuc_smooth / (2.0 * c**2))
   
    Veff_diag = basis.get_potential_from_grid(Veff_grid)
    if Vconf is not None:
        Veff_diag += basis.get_potential_from_grid(Vconf)

    # T_ZORA = -(1/2) d/dr[M d/dr], built element-by-element (same as T_NR but M-weighted)
    T_ZORA = basis.get_p_kinetic_matrix(M_grid)

    for il, l in enumerate(lchannels):

        # Centrifugal term: l(l+1) M(r) / (2r^2)
        Vl_grid = np.zeros_like(r_grid)
        if l > 0:
            Vl_grid[1:] = l * (l + 1) * M_grid[1:] / (2.0 * r_grid[1:]**2)
        Vl_grid[0] = Vl_grid[1]
        Vl_diag = basis.get_potential_from_grid(Vl_grid)

        H_mat = T_ZORA + np.diag(Veff_diag + Vl_diag)
        eps_l, vect = la.eigh(H_mat, subset_by_index=[0, nmax])

        vect_T = np.ascontiguousarray(vect.T)
        psi_l = basis.get_psi(vect_T, cplx=False)
        psi_l = set_phase(psi_l)

        psi[il, :nmax + 1, :] = psi_l
        eps[il, :nmax + 1] = eps_l[:nmax + 1]

    return eps, psi
#========================================================================================================
def solve_schrodinger_kh(basis: FEDVR_Basis, Veff_grid: np.ndarray, lmax: int, nmax: int,
                          Z: float = 1.0, nuclear_sigma: float = 1.0e-3,
                          Vconf: np.ndarray | None = None, lmin: int = 0,
                          maxiter: int = 50, tol: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Scalar-relativistic solver using the Koelling-Harmon (KH) fixed-point method.

    The KH mass factor M_inv(eps_ref, r) = 1/(1 + (eps_ref - V(r))/(2c^2)) is
    energy-dependent, making H_KH a nonlinear eigenvalue problem. The key to
    preserving orthogonality is to use a single reference energy eps_ref for the
    entire l-channel rather than a per-state energy:

      - All nmax+1 states are simultaneously solved from one Hermitian H_KH(eps_ref).
      - Eigenstates of a common Hermitian operator are exactly orthogonal.
      - eps_ref is updated (highest eigenvalue from previous iteration) and the
        channel is re-diagonalized until convergence. Typically 3-10 iterations.

    The kinetic energy D^T diag(M_inv) D / 2 is the exact DVR representation of
    -d/dr[M_inv d/dr]/2. This automatically includes the Darwin correction that
    arises from the spatial variation of M_inv (no separate Darwin term needed).

    The scalar-relativistic kappa average is -1 for all l-channels (the spin-orbit
    contributions from j=l+/-1/2 cancel exactly when averaged by degeneracy 2j+1).
    This is already captured by the radial-only D matrix.

    Gaussian nuclear model: the nuclear part of Veff used inside the KH mass is
    replaced by a Gaussian-smoothed version V_nuc(r) = -Z/r * erf(r/(sqrt(2)*sigma)),
    keeping M_inv finite and nonzero at r=0 (same regularisation as in ZORA).
    The diagonal potential Veff_diag is unchanged (full Coulomb).
    """
    c = 137.035999074

    r_grid = basis.get_gridpoints()
    r_grid[0] = 1.0e-10  # avoid division by zero

    lchannels = np.arange(lmin, lmax + 1, step=1, dtype=int)
    num_channels = len(lchannels)

    psi = np.zeros([num_channels, nmax + 1, len(r_grid)], dtype=np.float64)
    eps = np.zeros([num_channels, nmax + 1], dtype=np.float64)

    Veff_diag = basis.get_potential_from_grid(Veff_grid)
    if Vconf is not None:
        Veff_diag += basis.get_potential_from_grid(Vconf)

    # Reconstruct V_nuc_point with the same convention as V0_grid in full_atom_dft
    # (V0_grid[0] = V0_grid[1]), then compute the Gaussian correction once.
    V_nuc_point = np.empty_like(r_grid)
    V_nuc_point[1:] = -Z / r_grid[1:]
    V_nuc_point[0] = V_nuc_point[1]

    sig2 = np.sqrt(2.0) * nuclear_sigma
    V_nuc_smooth = np.empty_like(r_grid)
    V_nuc_smooth[1:] = -Z * erf(r_grid[1:] / sig2) / r_grid[1:]
    V_nuc_smooth[0] = -Z * np.sqrt(2.0 / np.pi) / nuclear_sigma  # analytic r->0 limit

    # V_eff with nuclear part replaced by its Gaussian-smoothed version
    Veff_for_M = Veff_grid + (V_nuc_smooth - V_nuc_point)

    # Initial eigenvalues from NR to seed the fixed-point iteration
    Tmat_NR = basis.get_kinetic_energy_matrix()

    for il, l in enumerate(lchannels):

        Vl_diag_NR = basis.get_potential_from_grid(get_centrifugal_potential(r_grid, l))
        H0 = Tmat_NR + np.diag(Veff_diag + Vl_diag_NR)
        eps_l, vect = la.eigh(H0, subset_by_index=[0, nmax])

        for it in range(maxiter):
            eps_old = eps_l.copy()

            # Single reference energy for the whole channel → one Hermitian H → exact orthogonality
            eps_ref = eps_l[nmax]

            # KH mass factor using smoothed nuclear potential to regularise M_inv near origin
            M_inv_grid = 1.0 / (1.0 + (eps_ref - Veff_for_M) / (2.0 * c**2))

            # T_KH = -(1/2) d/dr[M_inv d/dr], built element-by-element
            T_KH = basis.get_p_kinetic_matrix(M_inv_grid)

            # Centrifugal term: l(l+1) M_inv(r) / (2r^2)
            Vl_grid = np.zeros_like(r_grid)
            if l > 0:
                Vl_grid[1:] = l * (l + 1) * M_inv_grid[1:] / (2.0 * r_grid[1:]**2)
            Vl_grid[0] = Vl_grid[1]
            Vl_diag = basis.get_potential_from_grid(Vl_grid)

            H_mat = T_KH + np.diag(Veff_diag + Vl_diag)
            eps_l, vect = la.eigh(H_mat, subset_by_index=[0, nmax])

            if np.max(np.abs(eps_l - eps_old)) < tol:
                break

        vect_T = np.ascontiguousarray(vect.T)
        psi_l = basis.get_psi(vect_T, cplx=False)
        psi_l = set_phase(psi_l)

        psi[il, :nmax + 1, :] = psi_l
        eps[il, :nmax + 1] = eps_l

    return eps, psi
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
