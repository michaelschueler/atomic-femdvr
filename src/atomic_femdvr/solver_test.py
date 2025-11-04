from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from atomic_femdvr.adaptive_elements import optimize_elements
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.softcoul_solvers import solve_direct
from atomic_femdvr.utils import print_time


#==================================================================
def solver_test(plot: bool) -> None:

    Z = 100.0
    a0 = 1.0e-14
    lmax = 2
    num_states = 4

    ng = 8  # Number of grid points per element

    elem_method = 'exponential'
    elem_tols = np.logspace(-1, -4, 4)
    npoints_exp = []
    errors_exp = []

    for tol in elem_tols:
        npoints, error = solve_with_error(Z, a0, ng, elem_method, tol, lmax, num_states)
        npoints_exp.append(npoints)
        errors_exp.append(error)
        print(f"Method: {elem_method}, Tol: {tol:.1e}, Points: {npoints}, Max Error: {error:.2e}")

    elem_method = 'wkb'
    elem_tols = np.logspace(0, -3, 4)
    npoints_wkb = []
    errors_wkb = []

    for tol in elem_tols:
        npoints, error = solve_with_error(Z, a0, ng, elem_method, tol, lmax, num_states)
        npoints_wkb.append(npoints)
        errors_wkb.append(error)
        print(f"Method: {elem_method}, Tol: {tol:.1e}, Points: {npoints}, Max Error: {error:.2e}")

    if plot:
        fig, ax = plt.subplots()
        ax.plot(npoints_exp, errors_exp, 'o-', label='Exponential')
        ax.plot(npoints_wkb, errors_wkb, 's--', label='WKB')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Grid Points')
        ax.set_ylabel('Maximum Energy Error (a.u.)')
        ax.set_title('Convergence of Soft-Coulomb Solver')
        ax.legend()
        plt.show()

#==================================================================
def solve_with_error(Z, a0, ng, elem_method, elem_tol, lmax, num_states):

    Rmax = 200.0 / Z
    h_min = 0.5 / Z
    h_max = 20.0 / Z
    r_elements = optimize_elements(Z, h_min, h_max, Rmax, tol=elem_tol, Za=1.0,
                                  method=elem_method)

    ne = len(r_elements) - 1
    basis = FEDVR_Basis(ne, ng, r_elements, build_derivatives=True)
    grid = basis.get_gridpoints()
    num_points = len(grid)

    energies, wavefunctions = solve_direct(basis, Z, lmax, num_states - 1,
                                          a0=a0, solver='banded')

    errors = np.zeros((lmax + 1, num_states))
    for l in range(lmax + 1):
        energies_exact = -Z**2 / (2.0 * (np.arange(1, num_states + 1) + l)**2)
        errors[l, :] = energies[l, :] - energies_exact

    return num_points,np.amax(np.abs(errors))
#==================================================================
def solver_suite(plot: bool) -> None:
    """Run a suite of solvers on the soft-Coulomb potential and compare results.
    """
    Z = 100.0  # Nuclear charge
    a0 = 1.0e-14  # Softening parameter

    ng = 16  # Number of grid points per element

    tic = perf_counter()
    Rmax = 200.0 / Z
    h_min = 0.5 / Z
    h_max = 20.0 / Z
    elem_tol = 1e-1
    r_elements = optimize_elements(Z, h_min, h_max, Rmax, tol=elem_tol, Za=1.0,
                                  method='exponential')

    ne = len(r_elements) - 1
    basis = FEDVR_Basis(ne, ng, r_elements, build_derivatives=True)
    grid = basis.get_gridpoints()
    nb = len(grid)

    toc = perf_counter()
    print_time(tic, toc, "Optimizing finite elements")

    print(f"Number of elements: {len(r_elements) - 1}")
    print(f"Number of grid points: {nb}\n")


    fig, ax = plt.subplots()
    ax.scatter(r_elements, np.zeros_like(r_elements), s=10.0, edgecolors='r',
                facecolors='none', label='Element Boundaries')
    plt.show()


    lmax = 2  # Maximum angular momentum quantum number
    num_states = 4  # Number of bound states to compute

    # Solve using direct diagonalization
    tic = perf_counter()
    energies_direct, wavefunctions_direct = solve_direct(basis, Z, lmax, num_states,
                                                         a0=a0, solver='banded')
    toc = perf_counter()
    print_time(tic, toc, "Direct diagonalization")

    for l in range(lmax + 1):
        print(f"\nEnergies from direct diagonalization for l={l}:")
        energies_exact = -Z**2 / (2.0 * (np.arange(1, num_states + 1) + l)**2)
        for n in range(num_states):
            print(f"  n={n}, E_direct = {energies_direct[l, n]:.8f}, "
                  f"E_exact = {energies_exact[n]:.8f}, "
                  f"Diff = {energies_direct[l, n] - energies_exact[n]:.2e}")


    fig, ax = plt.subplots(1, lmax + 1, figsize=(15, 4))
    for l in range(lmax + 1):
        ax[l].set_title(f"l = {l} (Direct Diagonalization)")
        for n in range(num_states):
            ax[l].plot(grid, wavefunctions_direct[l, n, :], label=f"n={n}")
        ax[l].legend()
    plt.show()

    exit()

    l = 2  # Angular momentum quantum number
    num_states = 4  # Number of bound states to compute

    # Solve using direct diagonalization
    tic = perf_counter()
    energies_direct, wavefunctions_direct = solve_direct(basis, Z, l, num_states, a0=a0)
    toc = perf_counter()
    print_time(tic, toc, "Direct diagonalization")

    print("\nEnergies from direct diagonalization:")
    print(energies_direct)

    # Solve using iterative method (LOBPCG)
    prec_options = ['diag', 'tri', 'inv']

    for prec in prec_options:
        print(f"\nSolving using LOBPCG with preconditioner: {prec}")
        tic = perf_counter()
        energies_iterative, wavefunctions_iterative = solve_iterative(basis, Z, l, num_states, a0=a0,
                                                                       preconditioner=prec, maxiter=5000, tol=1e-6)
        toc = perf_counter()
        print_time(tic, toc, f"LOBPCG with preconditioner: {prec}")

        print("\nEnergies:")
        print(energies_iterative)

    # Solve using iterative method (PRIMME)

    # for prec in prec_options:
    #     print(f"\nSolving using PRIMME with preconditioner: {prec}")
    #     tic = perf_counter()
    #     energies_primme, wavefunctions_primme = solve_iterative(basis, Z, l, num_states, a0=a0,
    #                                                              preconditioner=prec, driver='primme',
    #                                                              maxiter=1000, tol=1e-8)
    #     toc = perf_counter()
    #     print_time(tic, toc, f"PRIMME with preconditioner: {prec}")

    #     print("\nEnergies:")
    #     print(energies_primme)
#========================================================================================================


