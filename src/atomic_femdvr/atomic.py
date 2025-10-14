from time import perf_counter

import numpy as np
from scipy.interpolate import UnivariateSpline

from atomic_femdvr.adaptive_elements import OptimizeElements
from atomic_femdvr.SchrodingerSolver import SolveNR, SolveSR, SolveZORA
from atomic_femdvr.utils import PrintTime

from atomic_femdvr.input import SysParamsInput, BaseModel, solver_input_factory, SolverInput

class AtomicInput(BaseModel):
    sysparams: SysParamsInput
    solver: solver_input_factory(default_hmin=0.01, default_hmax=4.0)

#==================================================================
def solve_atomic(sysparams: SysParamsInput, solver: SolverInput):
    """
    Solve the atomic system based on the provided parameters.
    """

    # Read potential from file
    rs, Vpot = np.loadtxt(sysparams.file_pot, usecols=sysparams.pot_columns, unpack=True)
    Rmax_ = rs[-1]

    pot_energy_unit = sysparams.pot_energy_unit
    if pot_energy_unit == "Ry":
        Vpot *= 0.5
    elif pot_energy_unit == "eV":
        Vpot *= 1. / (2 * 13.605693009 )

    spl = UnivariateSpline(rs, Vpot, s=0, k=3)
    Vpot_fnc = lambda r: spl(r)
    gradVpot_fnc = spl.derivative()

    # Effective charge
    Zc = - Vpot[0] * rs[0]  # Effective charge

    # Optimize radial elements
    solver.Rmax = solver.Rmax or Rmax_

    tic = perf_counter()
    r_elements = OptimizeElements(Zc, solver.h_min, solver.h_max, solver.Rmax, solver.tol)
    toc = perf_counter()
    PrintTime(tic, toc, "Optimize radial elements")
    print("\n")

    ne = len(r_elements) - 1  # Number of elements
    nb = ne * solver.ng + 1  # Total number of grid points
    print(f"Number of optimized radial elements: {ne}")
    print("Total number of grid points:", nb, "\n")

    lmax = sysparams.lmax
    nmax = sysparams.nmax

    psi = np.zeros([lmax + 1, nmax, nb], dtype=np.float64)
    eps = np.zeros([lmax + 1, nmax], dtype=np.float64)

    tic = perf_counter()
    print(f"Solving Schrödinger equation for lmax = {sysparams.lmax}, nmax = {sysparams.nmax}")
    print(f"method = {solver.method}")
    for l in range(lmax + 1):
        if solver.method == 'non-relativistic':
            eps[l, :nmax], r_grid, psi[l, :nmax, :] = SolveNR(r_elements, Vpot_fnc, l, sysparams.nmax, solver.ng)
        elif solver.method == 'zora':
            eps[l, :nmax], r_grid, psi[l, :nmax, :] = SolveZORA(r_elements, Vpot_fnc, gradVpot_fnc, l, sysparams.nmax, solver.ng)
        elif solver.method == 'scalar-relativistic':
            eps_guess, r_grid, psi[l, :nmax, :] = SolveNR(r_elements, Vpot_fnc, l, sysparams.nmax, solver.ng)
            eps[l, :nmax], r_grid, psi[l, :nmax, :] = SolveSR(r_elements, Vpot_fnc, gradVpot_fnc, l,
                                                              eps_guess, solver.ng)
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
