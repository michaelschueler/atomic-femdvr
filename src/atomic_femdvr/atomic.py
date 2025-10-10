import json
import os
from time import perf_counter

import numpy as np
from scipy.interpolate import UnivariateSpline

from atomic_femdvr.adaptive_elements import OptimizeElements
from atomic_femdvr.SchrodingerSolver import SolveNR, SolveSR, SolveZORA
from atomic_femdvr.utils import PrintTime


#==================================================================
def ReadInput(fname):
    """
    Read input parameters from a JSON file.
    """
    with open(fname) as f:
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
