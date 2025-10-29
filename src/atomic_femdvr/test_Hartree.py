import json
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline

from atomic_femdvr.adaptive_elements import OptimizeElements
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.legendre_integrals import GetLegendreIntegrals
from atomic_femdvr.upf_interface import UPFInterface
from atomic_femdvr.utils import PrintTime


#==================================================================
def HartreePotential_Spline(rs, rho):

    Ar_integrand = rs * rho
    Br_integrand = rs**2 * rho

    A_spl = UnivariateSpline(rs, Ar_integrand, s=0, k=3)
    B_spl = UnivariateSpline(rs, Br_integrand, s=0, k=3)

    A_integral_spl = A_spl.antiderivative()
    B_integral_spl = B_spl.antiderivative()
    A_integral = A_integral_spl(rs[-1]) - A_integral_spl(rs)

    B_term = np.zeros_like(rs)
    Ir, = np.where(rs > 1.0e-8)
    B_term[Ir] = B_integral_spl(rs[Ir]) / rs[Ir]
    Ir, = np.where(rs <= 1.0e-8)

    V_Ha = (A_integral + B_term)

    return V_Ha
#==================================================================
def HartreePotential_FEMDVR(rs, rho, Zc, h_min, h_max, elem_tol, ng):

    Rmax = rs[-1]
    r_elements = OptimizeElements(Zc, h_min, h_max, Rmax, elem_tol)
    # r_elements = np.linspace(0.0, Rmax, 10)

    ne = len(r_elements) - 1

    basis = FEDVR_Basis(ne, ng, r_elements)
    leg_integ = GetLegendreIntegrals(basis.leg, r_elements)
    grid = basis.GetGridpoints()

    rho_spl = UnivariateSpline(rs, rho, s=0, k=3)
    rho_grid = rho_spl(grid)

    rho_elem_prev = np.zeros(ng + 1)
    rho_elem_curr = np.zeros(ng + 1)

    A_integ = np.zeros_like(grid)
    B_integ = np.zeros_like(grid)
    A_tot = np.zeros(ne)
    B_tot = np.zeros(ne)

    A_tot_integ = 0.
    B_tot_integ = 0.

    for i in range(ne):
        rho_elem_curr[:] = rho_grid[i*ng : i*ng + ng + 1]
        r_curr = grid[i*ng : i*ng + ng + 1]

        A_integrand = rho_elem_curr * r_curr
        B_integrand = rho_elem_curr * r_curr**2
        A_elem_integ = np.dot(leg_integ[i, :, :], A_integrand)
        B_elem_integ = np.dot(leg_integ[i, :, :], B_integrand)

        A_integ[i*ng : i*ng + ng + 1] = A_elem_integ + A_tot_integ
        B_integ[i*ng : i*ng + ng + 1] = B_elem_integ + B_tot_integ

        A_tot_integ += A_elem_integ[-1]
        B_tot_integ += B_elem_integ[-1]


    V_Ha = np.zeros_like(grid)
    V_Ha[1:] = A_integ[-1] - A_integ[1:] + B_integ[1:] / grid[1:]
    V_Ha[0] = A_integ[-1] - A_integ[0]

    return r_elements, grid, V_Ha

#==================================================================
def main(argv):

    # read json input file
    if len(argv) < 2:
        print("Usage: python test_upf.py <input_file> <upf file>")
        return

    input_file = argv[0]
    try:
        with open(input_file) as f:
            params = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    upf_file = argv[1]


    upflib_dir = params.get('upflib_dir', '')
    lib_ext = params.get('lib_ext', 'so')

    tic = perf_counter()
    upf = UPFInterface(upflib_dir, lib_ext)
    toc = perf_counter()
    PrintTime(tic, toc, "Loading UPF library")

    tic = perf_counter()
    upf.Read_UPF(upf_file)
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

    # print occupations
    for iwfc in range(upf.nwfc):
        print(f"Wavefunction {iwfc+1}: n={upf.nnodes_chi[iwfc]}, l={upf.lchi[iwfc]}, occupation={upf.oc[iwfc]}")

    rho = np.zeros(upf.mesh, dtype=np.float64)
    for iwfc in range(upf.nwfc):
        rho[1:] += upf.oc[iwfc] * np.abs(upf.chi[1:, iwfc])**2 / upf.r[1:]**2
    rho[0] = rho[1]  # Set the first element to zero to avoid division by zero

    Zc = simpson(upf.r**2 * rho, x=upf.r)

    V_Ha_spl = HartreePotential_Spline(upf.r, rho)

    h_min = 1.0
    h_max = 4.0
    elem_tol = 1.0e-2
    ng = 8
    r_elements, grid, V_Ha_femdvr = HartreePotential_FEMDVR(upf.r, rho, Zc, h_min, h_max, elem_tol, ng)


    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(upf.r, rho, label='Charge Density')
    ax[1].plot(upf.r, V_Ha_spl, label='spline')
    ax[1].scatter(grid, V_Ha_femdvr, edgecolor='crimson', facecolor='none', label='FEMDVR')

    for i in range(len(r_elements)):
        ax[1].axvline(x=r_elements[i], c='k', ls='--')
    ax[-1].set_xlabel('r (a.u.)')
    ax[0].set_ylabel(r"$\rho$")
    ax[1].set_ylabel(r"$V_H $ (a.u.)")
    ax[1].legend()
    plt.show()

    # test Hartree potential
    spline = UnivariateSpline(upf.r, V_Ha_spl, s=0, k=3)
    der1_tmp = spline.derivative(1)
    der2_tmp = spline.derivative(2)
    rho_test_spl = -(2 / upf.r * der1_tmp(upf.r) + der2_tmp(upf.r))

    fig, ax = plt.subplots()
    ax.plot(upf.r, rho_test_spl, label='Spline')
    ax.plot(upf.r, rho, label='density')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('r (a.u.)')
    ax.set_ylabel(r"$\rho$ (a.u.)")
    ax.legend()
    plt.show()

#==================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
