from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial, genlaguerre

from atomic_femdvr.adaptive_elements import optimize_elements
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.softcoul_solvers import solve_direct
from atomic_femdvr.density_potential import charge_density
from atomic_femdvr.utils import print_time
#==================================================================
def hydrogenic_orbital(r: np.ndarray, Z: float, n: int, l: int) -> np.ndarray:
    """
    Computes the hydrogenic orbital for given quantum numbers n and l.

    Parameters:
    r : np.ndarray
        Radial grid points.
    Z : float
        Nuclear charge.
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number.

    Returns:
    np.ndarray
        The radial part of the hydrogenic orbital evaluated at r.
    """
    # Normalization constant
    a0 = 1.0  # Bohr radius in atomic units
    rho = 2 * Z * r / (n * a0)
    prefactor = np.sqrt((2 * Z / (n * a0))**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))

    # Radial part
    radial_part = prefactor * (rho ** l) * np.exp(-rho / 2) * genlaguerre(n - l - 1, 2 * l + 1)(rho)

    return radial_part
#========================================================================================================
def set_phase(psi):
    """
    Set the phase of the wavefunction to ensure it is positive at the maximum point.
    """
    idx_max = np.argmax(np.abs(psi))
    if psi[idx_max] < 0.0:
        psi *= -1.0
    return psi
#==================================================================
def wfc_test(plot: bool) -> None:

    Z = 1.0
    a0 = 1.0e-14
    lmax = 2
    num_states = 4
    ng = 8  # Number of grid points per element

    elem_method = 'exponential'
    elem_tol = 1e-1

    grid, energies, wavefunctions, rho = solve_schrodinger(Z, a0, ng, elem_method, elem_tol, lmax, num_states)


    if plot:

        fig, ax = plt.subplots(2, lmax + 1, sharex=True, figsize=(15, 8))

        for l in range(lmax + 1):
            ax[0,l].set_title(f"l = {l}")
            for n in range(num_states):
                ax[0, l].plot(grid, wavefunctions[l, n, :], label=f"n={n}")
            ax[0,l].legend()

            for n in range(num_states):
                wfc_exact = grid * hydrogenic_orbital(grid, Z, n + l + 1, l)
                wfc_exact = set_phase(wfc_exact)

                ax[1, l].plot(grid, np.abs(wavefunctions[l, n, :] - wfc_exact), label=f"n={n}")
                # ax[1, l].plot(grid, wfc_exact, '--', label=f"n={n} Exact")
            ax[1,l].legend()

        plt.show()



    # compute charge density    
    rho_exact = np.zeros_like(grid)
    for l in range(lmax + 1):
        for n in range(num_states):
            Rr_exact = hydrogenic_orbital(grid, Z, n + l + 1, l)
            rho_exact += Rr_exact**2


    if plot:
        fig, ax = plt.subplots()
        ax.plot(grid, np.abs(rho - rho_exact), label='error in density')
        ax.set_xlabel('r (a.u.)')
        ax.set_ylabel(r'$\rho(r)$')
        ax.legend()
        plt.show()

#==================================================================
def solve_schrodinger(Z, a0, ng, elem_method, elem_tol, lmax, num_states):

    Rmax = 200.0 / Z**(1/3)
    h_min = 0.25 / Z**(1/3)
    h_max = 20.0 / Z**(1/3)
    r_elements = optimize_elements(Z, h_min, h_max, Rmax, tol=elem_tol, Za=1.0,
                                  method=elem_method)

    ne = len(r_elements) - 1
    basis = FEDVR_Basis(ne, ng, r_elements, build_derivatives=True)
    grid = basis.get_gridpoints()


    energies, wavefunctions = solve_direct(basis, Z, lmax, num_states - 1,
                                          a0=a0, solver='banded')

    lchi = np.arange(0, lmax + 1).repeat(num_states)
    nnodes_chi = np.tile(np.arange(0, num_states), lmax + 1)
    occ = np.ones_like(lchi)

    rho = charge_density(basis, nnodes_chi, lchi, occ, wavefunctions)


    return grid, energies, wavefunctions, rho
#==================================================================

    #     print(energies_primme)
#========================================================================================================


