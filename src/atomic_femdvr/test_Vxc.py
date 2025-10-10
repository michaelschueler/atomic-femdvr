import json
import os
import sys
from time import perf_counter

import DensityPotential as denpot
import KohnSham as ks
import matplotlib.pyplot as plt
import numpy as np
from adaptive_elements import OptimizeElements
from femdvr import FEDVR_Basis
from scipy.interpolate import interp1d
from upf_interface import upf_class
from utils import PrintTime


#==================================================================
def main(argv):

    # read json input file
    if len(argv) < 1:
        print("Usage: python test_Vxc.py <input_file>")
        return

    input_file = argv[0]
    try:
        with open(input_file) as f:
            params = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    pseudo_config = params.get('pseudo_config', {})
    sysparams = params.get('sysparams', {})
    solver = params.get('solver', {})
    dft = params.get('dft', {})

    upflib_dir = pseudo_config.get('upflib_dir', '')
    lib_ext = pseudo_config.get('lib_ext', 'so')

    tic = perf_counter()
    upf = upf_class(upflib_dir, lib_ext)
    toc = perf_counter()
    PrintTime(tic, toc, "Loading UPF library")

    tic = perf_counter()
    file_upf = sysparams.get('file_upf', '')
    file_upf = os.path.join(pseudo_config.get('pseudo_dir', ''), file_upf)
    if not os.path.isfile(file_upf):
        raise FileNotFoundError(f"UPF file '{file_upf}' does not exist.")
    upf.Read_UPF(file_upf)
    toc = perf_counter()
    PrintTime(tic, toc, "Reading UPF file")

    tic = perf_counter()
    upf.ReadWavefunctions()
    toc = perf_counter()
    PrintTime(tic, toc, "Reading UPF wavefunctions")

    tic = perf_counter()
    upf.Read_PP()
    toc = perf_counter()
    PrintTime(tic, toc, "Reading UPF pseudo")

    Zval = upf.zp
    print("valence charge:", Zval)

    # print occupations
    for iwfc in range(upf.nwfc):
        print(f"Wavefunction {iwfc+1}: n={upf.nnodes_chi[iwfc]}, l={upf.lchi[iwfc]}, occupation={upf.oc[iwfc]}")

    # contruct elements
    tic = perf_counter()
    Rmax = solver.get('Rmax', 30.0)
    h_min = solver.get('h_min', 0.5)
    h_max = solver.get('h_max', 4.0)
    elem_tol = solver.get('elem_tol', 1.0e-2)
    r_elements = OptimizeElements(Zval, h_min, h_max, Rmax, elem_tol)
    toc = perf_counter()
    PrintTime(tic, toc, "Optimizing elements")

    print("number of elements:", len(r_elements))

    # create basis
    tic = perf_counter()
    ne = len(r_elements) - 1
    ng = solver.get('ng', 8)
    basis = FEDVR_Basis(ne, ng, r_elements, build_integrals=True)
    toc = perf_counter()
    PrintTime(tic, toc, "Creating FEDVR basis")

    # get charge density
    rho_upf = upf.GetChargeDensity()
    interp = interp1d(upf.r, rho_upf, kind='cubic', bounds_error=False, fill_value=0.0)
    grid = basis.GetGridpoints()
    rho_grid = interp(grid)

    # compute Hartree potential
    tic = perf_counter()
    V_Ha = denpot.HartreePotential(basis, rho_grid)
    toc = perf_counter()
    PrintTime(tic, toc, "Computing Hartree potential")

    # compute exchange-correlation potential`
    tic = perf_counter()
    x_functional = dft.get('xc_functional', 'gga_x_pbe')
    c_functional = dft.get('c_functional', 'gga_c_pbe')
    V_xc = denpot.ExchangeCorrelationPotential(basis, rho_grid, x_functional, c_functional)
    toc = perf_counter()
    PrintTime(tic, toc, "Computing exchange-correlation potential")

    interp = interp1d(upf.r, upf.vloc, kind='cubic', bounds_error=False, fill_value=0.0)
    vloc_grid = interp(grid)

    Vhxc = V_Ha + 0.5 * V_xc

    # load potential from file
    file_pot = 'data/Si_Vhxc.plot'
    rs, Vpot = np.loadtxt(file_pot, usecols=(0, 1), unpack=True)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(grid, rho_grid, label='Charge Density (computed)', c='black')
    ax[1].plot(grid, Vhxc, label='V_Hxc (computed)', c='red')
    ax[1].plot(rs, Vpot, label='V_Hxc (from file)', c='blue')
    ax[2].set_xlabel('r (a.u.)')
    ax[1].set_ylabel(r"$V_{Hxc}$ (a.u.)")
    ax[2].plot(grid, V_Ha, label='V_H (computed)', c='green')
    ax[2].plot(grid, V_xc, label='V_xc (computed)', c='orange')
    ax[2].set_ylabel(r"$V_{xc}$ (a.u.)")
    ax[1].legend()
    ax[2].legend()
    plt.show()

    exit()

    # fig, ax = plt.subplots(3, 1, sharex=True)
    # ax[0].plot(grid, V_Ha, label='Hartree Potential')
    # ax[0].set_ylabel(r"$V_{H}$ (a.u.)")
    # ax[0].legend()

    # ax[1].plot(grid, V_xc, label='Exchange-Correlation Potential')
    # ax[1].set_ylabel(r"$V_{xc}$ (a.u.)")
    # ax[1].legend()

    # ax[2].plot(grid, Veff, label='Effective Potential')
    # ax[2].set_xlabel('r (a.u.)')
    # ax[2].set_ylabel(r"$V_{eff}$ (a.u.)")
    # ax[2].legend()

    # plt.show()



    # prepare non-local projectors
    nbeta = upf.nbeta
    beta = np.ascontiguousarray(upf.beta.T)
    interp = interp1d(upf.r, beta, axis=1, kind='cubic', bounds_error=False, fill_value=0.0)
    beta_grid = interp(grid)

    tic = perf_counter()
    lmax = np.amax(upf.lchi)
    nmax = np.amax(upf.nnodes_chi)
    eps, psi = ks.SolveSchrodinger(basis, Veff, upf.lll, upf.dion, beta_grid, lmax, nmax)
    toc = perf_counter()
    PrintTime(tic, toc, "Solving Schrödinger equation")

    # run DFT self-consistency
    alpha = 0.6
    tol = 1.0e-8
    maxiter = 100

    err = 1.0e8
    it = 0
    V_Ha_old = V_Ha.copy()
    V_xc_old = V_xc.copy()
    rho_old = rho_grid.copy()
    while (err > tol) and (it < maxiter):
        rho_new = denpot.ChargeDensity(basis, upf.nnodes_chi, upf.lchi,
                                       upf.oc, psi)
        V_Ha_new = denpot.HartreePotential(basis, rho_old)
        V_xc_new = denpot.ExchangeCorrelationPotential(basis, rho_old, x_functional, c_functional)

        V_Ha_mix = alpha * V_Ha_new + (1. - alpha) * V_Ha_old
        V_xc_mix = alpha * V_xc_new + (1. - alpha) * V_xc_old

        Veff = V_Ha_mix + 0.5 * V_xc_mix + 0.5 * vloc_grid

        eps, psi = ks.SolveSchrodinger(basis, Veff, upf.lll, upf.dion, beta_grid, lmax, nmax)

        rho_diff = rho_new - rho_old
        err = np.sum(np.abs(rho_diff))

        rho_old = rho_new.copy()
        it += 1

        print(f"iter = {it}, error = {err:.2e}")

    fig, ax = plt.subplots(lmax + 1, 1, figsize=(8, 6), sharex=True)

    for iwf in range(upf.nwfc):
        l = upf.lchi[iwf]
        ax[l].plot(upf.r, upf.chi[:,iwf], ls='--', label=f"UPF Wavefunction {iwf+1} (l={l})")


    for l in range(lmax + 1):
        for n in range(nmax + 1):
            ax[l].plot(grid, psi[l, n, :], label=f"l={l}, n={n}, E={eps[l, n]:.4f} Ha")
        ax[l].set_ylabel(r"$\psi_l(r)$")
        ax[l].legend()
    ax[-1].set_xlabel('r (a.u.)')
    plt.tight_layout()
    plt.show()

#==================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
