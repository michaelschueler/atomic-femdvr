import sys
import os
import getopt
from time import perf_counter
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
import json

from utils import PrintTime, GetOrbitalLabel, PrintEigenvalues, PlotWavefunctions
from adaptive_elements import OptimizeElements
from SchrodingerSolver import SolveNR, SolvePseudo
from upf_interface import upf_class
#==================================================================
def ReadInput(fname):
    """
    Read input parameters from a JSON file.
    """
    with open(fname, 'r') as f:
        data = json.load(f)

    pseudo_config = data.get('pseudo_config', {})
    if not pseudo_config:
        raise ValueError("No 'pseudo_config' found in the input file.")

    sysparams = data.get('sysparams', {})
    if not sysparams:
        raise ValueError("No 'sysparams' found in the input file.")
    solver = data.get('solver', {})
    if not solver:
        raise ValueError("No 'solver' parameters found in the input file.")

    return pseudo_config, sysparams, solver
#==================================================================
def SolveAtomic(pseudo_config, sysparams, solver):
    """
    Solve the atomic system based on the provided parameters.
    """

    upflib_dir = pseudo_config.get('upflib_dir', '')
    lib_ext = pseudo_config.get('lib_ext', 'so')
    pseudo_dir = pseudo_config.get('pseudo_dir', '')

    file_upf = sysparams.get('file_upf', '')
    file_upf = os.path.join(pseudo_dir, file_upf)
    if not os.path.isfile(file_upf):
        raise FileNotFoundError(f"UPF file '{file_upf}' does not exist.")

    # Read UPF file
    tic = perf_counter()
    upf = upf_class(upflib_dir, lib_ext)
    upf.Read_UPF(file_upf)
    upf.Read_PP()
    toc = perf_counter()
    PrintTime(tic, toc, "Reading UPF file")
    print("\n")

    rs = upf.r
    Vloc = np.ascontiguousarray(upf.vloc)
    # convert to Hartree
    Vloc *= 0.5  # Convert to Hartree (1 Hartree = 2 Rydberg)
    Rmax_ = rs[-1]

    spl = UnivariateSpline(rs, Vloc, s=0, k=3)
    Vpot_fnc = lambda r: spl(r)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(rs, Vloc, label='Local potential Vloc')
    ax.set_xlabel('r (a.u.)')
    ax.set_ylabel('Vloc (a.u.)')

    plt.show()
    exit()

    # Effective charge
    Zc = - Vpot_fnc(1.0)  # Effective charge

    print(f"Effective charge Zc = {Zc:.6f} a.u.\n")


    # Optimize radial elements
    h_min = solver.get('h_min', 0.01)
    h_max = solver.get('h_max', 4.0)
    Rmax = solver.get('Rmax', Rmax_)
    tol = solver.get('tol', 1.0e-3)
    ng = solver.get('ng', 8)


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
    for l in range(lmax + 1):

        Ib, = np.where(upf.lll == l)

        if len(Ib) == 0:
            # No beta functions for this l, use local potential
            eps[l, :nmax], r_grid, psi[l, :nmax, :] = SolveNR(r_elements, Vpot_fnc, l, nmax, ng)

        else:
            # prepare beta functions
            nbeta = len(Ib)
            irmax = upf.kbeta_max
            beta_l = np.zeros([nbeta, irmax], dtype=np.float64)
            for ibeta in range(nbeta):
                beta_l[ibeta, :] = upf.beta[0:irmax, Ib[ibeta]]


            beta_fnc = interp1d(rs[0:irmax], beta_l, axis=1, kind='cubic', 
                                bounds_error=False, fill_value=0)

            Dion_Hr = np.zeros([nbeta, nbeta], dtype=np.float64)
            for i in range(nbeta):
                for j in range(nbeta):
                    Dion_Hr[i, j] = 0.5 * upf.dion[Ib[i], Ib[j]]

            eps[l, :nmax], r_grid, psi[l, :nmax, :] = SolvePseudo(r_elements, Vpot_fnc, 
                                                                  Dion_Hr, beta_fnc, l, nmax, ng)

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


#==================================================================
def main(argv):

    short_options = "hpi:o:"
    long_options = ["help", "plot", "input=", "output="]

    print(60 * '*')
    print("Pseudo-atomic Schrödinger Equation Solver".center(60))
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
            print("Usage: python pseudo_atomic.py -i <input_file> [-o <output_file>] [--plot]")
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
    pseudo_config, sysparams, solver = ReadInput(input_file)

    eigenvalues, r_grid, psi = SolveAtomic(pseudo_config, sysparams, solver)

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