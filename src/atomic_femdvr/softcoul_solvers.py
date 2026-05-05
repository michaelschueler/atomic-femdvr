import numpy as np
import primme
import scipy.linalg as la
from scipy.interpolate import interp1d
from scipy.sparse.linalg import LinearOperator, lobpcg

from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.kohn_sham import solve_schrodinger_local


# ====================================================================
def solve_direct(
    basis: FEDVR_Basis,
    Z: float,
    lmax: int,
    num_states: int,
    a0: float = 1.0e-2,
    solver: str = "full",
):
    """
    Solve the radial Schrödinger equation for a soft-Coulomb potential using direct diagonalization.
    """
    # construct soft-Coulomb potential
    r_grid = basis.get_gridpoints()
    Vsc_grid = -Z / np.sqrt(r_grid**2 + a0**2)

    eps_l, psi_l = solve_schrodinger_local(basis, Vsc_grid, lmax, num_states, solver=solver)

    return eps_l, psi_l


# ====================================================================
def get_derivative_matrix(rs: np.ndarray) -> np.ndarray:
    """
    Returns the second-order derivative matrix for given grid
    """
    # compute diagonal and off-diagonal elements of the derivative matrix
    n = len(rs)
    D_diag = np.zeros(n)
    D_offdiag = np.zeros(n - 1)

    for i in range(1, n - 1):
        h1 = rs[i] - rs[i - 1]
        h2 = rs[i + 1] - rs[i]
        D_diag[i] = -2.0 / (h1 * h2)
        D_offdiag[i - 1] = 1.0 / (h1 * (h1 + h2))
        D_offdiag[i] = 1.0 / (h2 * (h1 + h2))

    return D_diag, D_offdiag


# ====================================================================
def get_guess(rgrid: np.ndarray, Z: float, l: int, num_states: int, a0: float) -> np.ndarray:
    nr = len(rgrid)
    rs = np.linspace(rgrid[1], rgrid[-2], nr)
    h = rs[1] - rs[0]

    H_diag = -Z / np.sqrt(rs**2 + a0**2) + l * (l + 1) / (2.0 * rs**2)
    H_diag += 1.0 / h**2
    H_offdiag = -0.5 / h**2 * np.ones(nr - 1)

    lam, phi0 = la.eigh_tridiagonal(H_diag, H_offdiag, select="i", select_range=(0, num_states - 1))

    intp = interp1d(rs, phi0, kind="cubic", axis=0, fill_value=0.0, bounds_error=False)
    phi0 = intp(rgrid[1:-1])

    return lam, phi0


# ====================================================================
def solve_iterative(
    basis: FEDVR_Basis,
    Z: float,
    l: int,
    num_states: int,
    a0: float = 1.0e-2,
    driver: str = "lobpcg",
    preconditioner: str | None = None,
    maxiter: int = 5000,
    tol: float = 1e-8,
):
    """
    Solve the radial Schrödinger equation for a soft-Coulomb potential using LOBPCG method.
    """
    ne = basis.ne  # Number of elements
    ng = basis.ng  # Number of grid points per element
    nb = ne * ng - 1  # Total number of grid points
    r_grid = basis.get_gridpoints()

    # construct kinetic energy matrix
    Tmat = basis.get_kinetic_energy_matrix()

    # construct soft-Coulomb potential
    Vsc_grid = -Z / np.sqrt(r_grid**2 + a0**2)

    # add centrifugal term
    if l > 0:
        Vsc_grid[1:] += l * (l + 1) / (2.0 * r_grid[1:] ** 2)
        Vsc_grid[0] = Vsc_grid[1]  # Avoid division by zero

    Vsc_mat = np.diag(basis.get_potential_from_grid(Vsc_grid))

    # construct Hamiltonian matrix
    H_mat = Tmat + Vsc_mat

    D_diag, D_offdiag = get_derivative_matrix(r_grid[1:-1])
    T_diag = -0.5 * D_diag
    -0.5 * D_offdiag
    H_diag = T_diag + np.diag(Vsc_mat)

    # estimation of lowest eigenvalues for preconditioning
    eps0 = -0.5 * Z**2 / ((l + 1) ** 2)

    # arrays for preconditioning
    if preconditioner == "diag":
        P_1 = 1.0 / np.diag(H_mat)

        def prec(x):
            if x.ndim == 1:
                return x * P_1
            else:
                return x[:, :] * P_1[:, None]

    elif preconditioner == "inv":
        # P_2 = np.linalg.inv(H_mat)
        P_2 = la.inv(Tmat) * np.eye(nb)

        def prec(x):
            return P_2 @ x

    elif preconditioner == "tri":
        h = (r_grid[-2] - r_grid[1]) / (len(r_grid) - 3)

        H_diag = (1.0 / h**2) * np.ones(nb)
        H_offdiag = -0.5 / h**2 * np.ones(nb - 1)

        def prec(x):
            _du2, _d, _du, y, _info = la.lapack.dgtsv(H_offdiag, H_diag, H_offdiag, x)
            return y
    else:

        def prec(x):
            return x

    # define linear operator for Hamiltonian
    def matvec(x):
        return H_mat @ x

    H_op = LinearOperator((nb, nb), matvec=matvec)
    P_op = LinearOperator((nb, nb), matvec=prec)

    # initial guess for wavefunctions
    _lam, X = get_guess(r_grid, Z, l, num_states, a0)

    if driver == "primme":
        # solve eigenvalue problem using PRIMME

        eigvals, eigvecs = primme.eigsh(
            H_op, num_states, M=P_op, sigma=eps0, which="SM", maxiter=10000, tol=tol
        )

        return eigvals, eigvecs.T  # shape (num_states, nb)

    elif driver == "lobpcg":
        # solve eigenvalue problem using LOBPCG
        eigvals, eigvecs = lobpcg(H_op, X, M=P_op, largest=False, maxiter=maxiter, tol=tol)
    else:
        raise ValueError(f"Unknown driver: {driver}")

    return eigvals, eigvecs.T  # shape (num_states, nb)
