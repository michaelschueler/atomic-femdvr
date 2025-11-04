from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp, simpson, quad

from atomic_femdvr.adaptive_elements import optimize_elements
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.density_potential import hartree_potential
from atomic_femdvr.utils import print_time
#==================================================================
def model_potential(rs):
    V0 = -1.0
    rc = 4.0
    a0 = 1.0
    V = V0 / np.sqrt(rs**2 + a0**2)
    return V
#==================================================================
def get_rho_model(rs):
    return np.exp(-rs)
#==================================================================
def rho_from_potential(basis, V):
    """Generate a model charge density from a given potential using Poisson's equation"""

    ne = basis.ne
    ng = basis.ng
    grid = basis.get_gridpoints()

    dV_dr = np.zeros_like(V)
    d2V_dr2 = np.zeros_like(V)

    for i in range(ne):
        h = 0.5 * (basis.xp[i+1] - basis.xp[i])
        V_elem = V[i*ng : i*ng + ng + 1]
        dV_dr_elem = np.dot(basis.leg.D_ii, V_elem) / h
        d2V_dr2_elem = np.dot(basis.leg.D_ii, np.dot(basis.leg.D_ii, V_elem)) / (h**2)
        dV_dr[i*ng : i*ng + ng + 1] = dV_dr_elem
        d2V_dr2[i*ng : i*ng + ng + 1] = d2V_dr2_elem

    rho = np.zeros_like(V)
    rho[1:] = - (d2V_dr2[1:] + 2.0 / grid[1:] * dV_dr[1:]) / (4.0 * np.pi)
    rho[0] = - (3.0 * d2V_dr2[0]) / (4.0 * np.pi)

    return rho
#==================================================================
def HartreePotential_Spline(rs, rho):

    Ar_integrand = rs**2 * rho
    Br_integrand = rs * rho

    # B_inf = simpson(Br_integrand, rs)

    A_spl = UnivariateSpline(rs, Ar_integrand, s=0, k=3)
    B_spl = UnivariateSpline(rs, Br_integrand, s=0, k=3)

    A_integral_spl = A_spl.antiderivative()
    B_integral_spl = B_spl.antiderivative()
    # A_integral = A_integral_spl(rs[-1]) - A_integral_spl(rs)

    B_inf = B_integral_spl(rs[-1])

    V_Ha = - B_integral_spl(rs) + B_inf
    V_Ha[1:] += A_integral_spl(rs[1:]) / rs[1:]


    # B_term = np.zeros_like(rs)
    # Ir, = np.where(rs > 1.0e-8)
    # B_term[Ir] = B_integral_spl(rs[Ir]) / rs[Ir]
    # Ir, = np.where(rs <= 1.0e-8)

    # V_Ha = (A_integral + B_term)

    return V_Ha
#==================================================================
def HartreePotential_ODE(rs, rho):

    rho_spl = UnivariateSpline(rs, rho, s=0, k=3)

    integrand = lambda r: rho_spl(r) * r
    B_inf = quad(integrand, 0, rs[-1], epsabs=1.0e-12, epsrel=1.0e-12)[0]


    # V0 = -4 * np.pi * simpson(rho * rs, rs)
    V0 = - 4*np.pi * B_inf


    def rhs_fnc(r, w):
        return np.array([w[1], -2.0 / (r + 1e-12) * w[1] + 4.0 * np.pi * rho_spl(r)])

    sol = solve_ivp(rhs_fnc, (rs[0], rs[-1]), [V0, 0.0], t_eval=rs, 
                    method='RK45', rtol=1.0e-12, atol=1.0e-12)
    V_Ha = sol.y[0, :]
    return V_Ha

#==================================================================
def hartree_test(plot: bool) -> None:

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

    # V_model = model_potential(grid)
    # rho_model = rho_from_potential(grid, V_model)

    rho_model = get_rho_model(grid)

    V_Ha_fedvr = hartree_potential(basis, rho_model)
    V_Ha_spline = HartreePotential_Spline(grid, rho_model)
    V_Ha_ode = HartreePotential_ODE(grid, rho_model)

    rho_fedvr = rho_from_potential(basis, V_Ha_fedvr)
    rho_spline = rho_from_potential(basis, V_Ha_spline)
    rho_ode = rho_from_potential(basis, V_Ha_ode)

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        # ax[0].plot(grid, V_model, marker='o', label='Model Potential')
        ax[0].plot(grid, 4*np.pi * V_Ha_fedvr, marker='x', label='Computed Hartree Potential (FEDVR)')
        ax[0].plot(grid, 4*np.pi * V_Ha_spline, marker='^', label='Computed Hartree Potential (Spline)')
        ax[0].plot(grid, - V_Ha_ode, marker='s', label='Computed Hartree Potential (ODE)')
        ax[0].set_xlabel('r (a.u.)')
        ax[0].set_ylabel('V(r) (a.u.)')
        ax[0].set_title('Model Potential')
        ax[0].legend()
        ax[1].plot(grid, rho_model, marker='o', label='Model Charge Density', color='orange')
        ax[1].plot(grid, 4*np.pi * rho_fedvr, marker='x', label='Computed Charge Density (FEDVR)', color='green')
        ax[1].plot(grid, 4*np.pi * rho_spline, marker='^', label='Computed Charge Density (Spline)', color='red')
        ax[1].plot(grid, - rho_ode, marker='s', label='Computed Charge Density (ODE)', color='purple')
        ax[1].set_xlabel('r (a.u.)')
        ax[1].set_ylabel('rho(r) (a.u.)')
        ax[1].set_title('Model Charge Density')
        ax[1].legend()
        plt.tight_layout()
        plt.show()

#==================================================================
def hartree_benchmark(file_rho:str, file_vh: str) -> None:

    rho_data = np.loadtxt(file_rho)
    vh_data = np.loadtxt(file_vh)

    rs_rho = rho_data[:, 0]
    rho = rho_data[:, 1]
    rho_spl = UnivariateSpline(rs_rho, rho / rs_rho**2, s=0, k=3, ext=0)

    rs_vh = vh_data[:, 0]
    V_Ha_ref = vh_data[:, 1]
    V_Ha_ref *= 0.5  # Convert from Ry to Hartree

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

    V_Ha_femdvr = hartree_potential(basis, rho_grid)

    error = np.max(np.abs(V_Ha_femdvr - np.interp(grid, rs_vh, V_Ha_ref)))
    print(f"Maximum error in Hartree potential: {error:.6e} a.u.")

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(rs_rho, rho, label='Charge Density')
    ax[1].plot(rs_vh, V_Ha_ref, label='Reference Hartree Potential', color='black')
    ax[1].scatter(grid, V_Ha_femdvr, edgecolor='crimson', facecolor='none', label='FEMDVR Hartree Potential')

    for i in range(len(r_elements)):
        ax[1].axvline(x=r_elements[i], c='k', ls='--')
    ax[-1].set_xlabel('r (a.u.)')
    ax[0].set_ylabel(r"$\rho$")
    ax[1].set_ylabel(r"$V_H $ (a.u.)")
    ax[1].legend()
    plt.show()





