from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson

from atomic_femdvr.adaptive_elements import optimize_elements
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.density_potential import exchange_correlation_potential

#==================================================================
def vxc_benchmark(file_rho:str, file_vxc: str) -> None:

    rho_data = np.loadtxt(file_rho)
    vxc_data = np.loadtxt(file_vxc)

    rs_rho = rho_data[:, 0]
    rho = rho_data[:, 1]
    rho_spl = UnivariateSpline(rs_rho, rho / rs_rho**2, s=0, k=3, ext=0)


    rs_vxc = vxc_data[:, 0]
    V_xc_ref = vxc_data[:, 1]

    Zc = simpson(rho, x=rs_rho)
    print(f"Total charge from rho: {Zc:.6f} a.u.")

    Z = 1.0
    ng = 16  # Number of grid points per element

    elem_method = 'exponential'
    elem_tol = 1e-1

    Rmax = 80.0 / Z**(1/3)
    h_min = 0.25 / Z**(1/3)
    h_max = 20.0 / Z**(1/3)
    r_elements = optimize_elements(Z, h_min, h_max, Rmax, tol=elem_tol, Za=1.0,
                                  method=elem_method)

    ne = len(r_elements) - 1
    basis = FEDVR_Basis(ne, ng, r_elements, build_derivatives=True,
                        build_integrals=True)
    grid = basis.get_gridpoints()

    rho_grid = rho_spl(grid)
    Zc = simpson(grid**2 * rho_grid, x=grid)
    print(f"Total charge from rho: {Zc:.6f} a.u.")

    
    # rho_grid *= 2.0

    # # fac = 1.0 / 13.605693122994
    # fac = 1.0 / (4 * np.pi)
    # print("Scaling factor applied to rho_grid:", fac)
    # print(" 1 / fac =", 1 / fac)

    # rho_grid *= fac

    # rs_lin = np.linspace(0.0, Rmax, 1000)
    # rho_lin = rho_spl(rs_lin)
    # rho_lin *= 2.0

    Vxc_femdvr = exchange_correlation_potential(basis, rho_grid, 
                                                   x_functional='lda_x',
                                                   c_functional='lda_c_pz',
                                                   alpha_x=1.0, driver='pylibxc')
    
    # fac = Vxc_femdvr[0] / V_xc_ref[0]
    # print(f"Scaling factor applied to Vxc_femdvr: {fac:.6f}")
    # Vxc_femdvr /= fac

    # Ry = 13.605693122994
    # print(4 * np.pi / Ry)
    # exit()

    # Vxc_lin = exchange_correlation_potential(basis, rho_lin, 
    #                                                x_functional='lda_x',
    #                                                c_functional='LDA_C_PZ',
    #                                                alpha_x=1.0, driver='pylibxc')

    # Vxc_femdvr = exchange_correlation_potential(basis, rho_grid, 
    #                                                xc_functional='LDA_XC_ZLP',
    #                                                alpha_x=1.0, driver='pylibxc')

    # Vxc_femdvr = exchange_correlation_potential(basis, rho_grid, 
    #                                             xc_functional='PBE',
    #                                             alpha_x=1.0, driver='internal')

    error = np.max(np.abs(Vxc_femdvr - np.interp(grid, rs_vxc, V_xc_ref)))
    print(f"Maximum error in exchange-correlation potential: {error:.6e} a.u.")

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(rs_rho, rho, label='Charge Density')
    ax[1].plot(rs_vxc, V_xc_ref, label='Reference Exchange-Correlation Potential', color='black')
    ax[1].plot(grid, Vxc_femdvr, c='crimson', marker='o', markeredgecolor='crimson',
                markerfacecolor='none', label='FEMDVR Exchange-Correlation Potential')
    # ax[1].plot(rs_lin, Vxc_lin, c='blue', ls='--', label='FEMDVR Exchange-Correlation Potential (linear grid)')

    for i in range(len(r_elements)):
        ax[1].axvline(x=r_elements[i], c='k', ls='--')
    ax[-1].set_xlabel('r (a.u.)')
    ax[0].set_ylabel(r"$\rho$")
    ax[1].set_ylabel(r"$V_{xc} $ (a.u.)")
    ax[1].legend()
    plt.show()