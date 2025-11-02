import numpy as np
from time import perf_counter

from atomic_femdvr.adaptive_elements import optimize_elements
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.utils import plot_wavefunctions, print_eigenvalues, print_time

from atomic_femdvr.softcoul_solvers import solve_direct, solve_iterative
#==================================================================
def solver_suite(plot: bool) -> None:
    """Run a suite of solvers on the soft-Coulomb potential and compare results.
    """

    Z = 1.0  # Nuclear charge
    a0 = 0.5  # Softening parameter

    ng = 16

    tic = perf_counter()
    Rmax = 80.0
    h_min = 0.5
    h_max = 5.0
    elem_tol = 1e-2
    r_elements = optimize_elements(Z, h_min, h_max, Rmax, elem_tol)

    ne = len(r_elements) - 1
    basis = FEDVR_Basis(ne, ng, r_elements)
    grid = basis.get_gridpoints()
    nb = len(grid)

    toc = perf_counter()
    print_time(tic, toc, "Optimizing finite elements")

    print(f"Number of elements: {len(r_elements) - 1}")
    print(f"Number of grid points: {nb}\n")


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


