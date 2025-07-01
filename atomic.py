import sys
import os
import getopt
from time import perf_counter
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import json

from adaptive_elements import OptimizeElements
from femdvr import FEDVR_Basis
from SchrodingerSolver import SolveNR, SolveZORA, SolveSR
#==================================================================
def PrintTime(tic, toc, msg):
    """
    Print the elapsed time for a given operation.
    """
    elapsed = toc - tic
    if elapsed < 1:
        print(f"Time[{msg}] : {elapsed * 1000:.2f} ms")
    elif elapsed > 300:
        print(f"Time[{msg}] : {elapsed / 60:.2f} m")
    else:
        print(f"Time[{msg}] : {elapsed:.2f} s")
#==================================================================
def ReadInput(fname):
    """
    Read input parameters from a JSON file.
    """
    with open(fname, 'r') as f:
        data = json.load(f)

    sysparams = data.get('sysparams', {})
    if not sysparams:
        raise ValueError("No 'sysparams' found in the input file.")
    solver = data.get('solver', {})
    if not solver:
        raise ValueError("No 'solver' parameters found in the input file.")

    return sysparams, solver
#==================================================================
def SolveAtomic(sysparams, solver):
    """
    Solve the atomic system based on the provided parameters.
    """

    file_pot = sysparams.get('file_pot', '')
    if not os.path.isfile(file_pot):
        raise FileNotFoundError(f"Potential file '{file_pot}' does not exist.")

    pot_columns = sysparams.get('pot_columns', [0, 4])
    if len(pot_columns) != 2:
        raise ValueError("Expected 'pot_columns' to have exactly two elements.")

    # Read potential from file
    rs, Vpot = np.loadtxt(file_pot, usecols=pot_columns, unpack=True)
    Rmax_ = rs[-1]

    pot_energy_unit = sysparams.get('pot_energy_unit', 'Rydberg')
    if pot_energy_unit.lower() == 'rydberg':
        Vpot *= 0.5
    elif pot_energy_unit.lower() == 'ev':
        Vpot *= 1. / (2 * 13.605693009 )

    spl = UnivariateSpline(rs, Vpot, s=0, k=3)
    Vpot_fnc = lambda r: spl(r)
    gradVpot_fnc = spl.derivative()

    # Effective charge
    Zc = - Vpot[0] * rs[0]  # Effective charge

    # Optimize radial elements
    h_min = solver.get('h_min', 0.01)
    h_max = solver.get('h_max', 4.0)
    Rmax = solver.get('Rmax', Rmax_)
    tol = solver.get('tol', 1.0e-3)
    ng = solver.get('ng', 8)

    method = solver.get('method', 'non-relativistic')
    allowed_methods = ['non-relativistic', 'zora', 'scalar-relativistic']
    if method.lower() not in allowed_methods:
        raise ValueError(f"Method '{method}' is not supported. Choose from {allowed_methods}.")

    tic = perf_counter()
    r_elements = OptimizeElements(Zc, h_min, h_max, Rmax, tol)
    toc = perf_counter()
    PrintTime(tic, toc, "Optimize radial elements")
    print("\n")

    ne = len(r_elements) - 1  # Number of elements
    nb = ne * ng + 1  # Total number of grid points
    print(f"Number of optimized radial elements: {ne}")
    print("Total number of grid points:", nb, "\n")

    lmax = sysparams.get('lmax', 0)
    nmax = sysparams.get('nmax', 4)

    psi = np.zeros([lmax + 1, nmax, nb], dtype=np.float64)
    eps = np.zeros([lmax + 1, nmax], dtype=np.float64)

    tic = perf_counter()
    print(f"Solving Schrödinger equation for lmax = {lmax}, nmax = {nmax}")
    print(f"method = {method}")
    for l in range(lmax + 1):
        if method.lower() == 'non-relativistic':
            eps[l, :nmax], r_grid, psi[l, :nmax, :] = SolveNR(r_elements, Vpot_fnc, l, nmax, ng)
        elif method.lower() == 'zora':
            eps[l, :nmax], r_grid, psi[l, :nmax, :] = SolveZORA(r_elements, Vpot_fnc, gradVpot_fnc, l, nmax, ng)
        elif method.lower() == 'scalar-relativistic':
            eps_guess, r_grid, psi[l, :nmax, :] = SolveNR(r_elements, Vpot_fnc, l, nmax, ng)
            eps[l, :nmax], r_grid, psi[l, :nmax, :] = SolveSR(r_elements, Vpot_fnc, gradVpot_fnc, l,
                                                              eps_guess, ng)
    toc = perf_counter()
    PrintTime(tic, toc, "Schrödinger equation")
    print("\n")

    eigenvalues = {}
    for l in range(lmax + 1):
        Ie, = np.where(eps[l, :nmax] < 0)
        eps_bound = eps[l, Ie]
        tag = f'{l}'
        eigenvalues[tag] = eps_bound.tolist()

    return eigenvalues, r_grid, psi
#==================================================================
def GetOrbitalLabel(n, l):
    """
    Get the orbital label for a given principal quantum number n and angular momentum quantum number l.
    """
    l_labels = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j']

    nq = n + l + 1  # Principal quantum number

    if l < len(l_labels):
        return f"{nq}{l_labels[l]}"
    else:
        return f"{nq}l{l}"  # Fallback for higher angular momentum states
#==================================================================
def PrintEigenvalues(lmax, eigenvalues):
    """
    Print the eigenvalues for each angular momentum quantum number.
    """

    Hr_to_eV = 2. * 13.605693009  # Hartree to eV conversion factor


    print(40 * '-')
    print("eigenvalues (in eV)".center(40))
    print(40 * '-')
    for l in range(lmax + 1):
        print(f"l = {l}")
        eps_bound = eigenvalues.get(f'{l}', [])
        n_bound = len(eps_bound)
        if n_bound == 0:
            print("  No bound states found.")
        else:
            for n in range(n_bound):
                orb = GetOrbitalLabel(n, l)
                print(f"  E({orb}) = {Hr_to_eV * eps_bound[n]:.6f} eV")
    print(40 * '-')
#==================================================================
def PlotWavefunctions(r_grid, psi, lmax, eigenvalues):

    """
    Plot the wavefunctions for each angular momentum quantum number.
    """

    fig, ax = plt.subplots(1, lmax + 1, figsize=(4*(lmax + 1), 6))

    for l in range(lmax + 1):
        ax[l].set_title(rf"$\ell$ = {l}")
        ax[l].set_xlabel("r (a.u.)")
        ax[l].set_ylabel("wave-function")

        eps_bound = eigenvalues.get(f'{l}', [])
        n_bound = len(eps_bound)

        for n in range(n_bound):
            orb = GetOrbitalLabel(n, l)
            ax[l].plot(r_grid, psi[l, n, :], label=orb)

        ax[l].legend()
        ax[l].set_xlim([0, r_grid[-1]])
        # ax[l].set_ylim([0, np.max(psi**2) * 1.1])

    plt.tight_layout()
    plt.show()

#==================================================================
def main(argv):

    short_options = "hpi:o:"
    long_options = ["help", "plot", "input=", "output="]

    print(60 * '*')
    print("Atomic Schrödinger Equation Solver".center(60))
    print(60 * '*')
    tic = perf_counter()

    try:
        opts, args = getopt.getopt(argv, short_options, long_options)
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2) 

    # get input and output file names
    input_file = ''
    output_file = ''
    plot_results = False    

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: python atomic.py -i <input_file> [-o <output_file>] [--plot]")
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-o", "--output"):
            output_file = arg
        elif opt == "--plot":
            plot_results = True

    if not input_file:
        print("Error: Input file is required. Use -i <input_file> to specify it.")
        sys.exit(2)

    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(2)

    # Read input parameters
    sysparams, solver = ReadInput(input_file)

    eigenvalues, r_grid, psi = SolveAtomic(sysparams, solver)

    lmax = sysparams.get('lmax', 0)
    PrintEigenvalues(lmax, eigenvalues)

    if plot_results:
        PlotWavefunctions(r_grid, psi, lmax, eigenvalues)

    toc = perf_counter()
    PrintTime(tic, toc, "Total")
    print(60 * '*')

#==================================================================
if __name__ == "__main__":  
    main(sys.argv[1:])