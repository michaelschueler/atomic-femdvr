import numpy as np
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.XCFunctionals import gga_functional


#===================================================================
def ChargeDensity(basis:FEDVR_Basis, nnodes_chi:np.ndarray, lchi:np.ndarray,
                  occ:np.ndarray, psi:np.ndarray):
    """Computes the charge density from given wave-functions"""
    lmax = np.amax(lchi)
    nmax = np.amax(nnodes_chi)
    nwf = len(occ)

    grid = basis.GetGridpoints()
    rho = np.zeros_like(grid)

    # get derivative at r=0
    h = 0.5 * (basis.xp[1] - basis.xp[0])
    psi_elem = psi[:, :, :basis.ng + 1]
    dpsi_dr_elem = np.einsum('ij,lnj->lni', basis.leg.D_ii, psi_elem) / h

    for iwf in range(nwf):
        l = lchi[iwf]
        n = nnodes_chi[iwf]
        rho[1:] += occ[iwf] * np.abs(psi[l, n, 1:])**2 / grid[1:]**2
        rho[0] += occ[iwf] * dpsi_dr_elem[l, n, 0]**2


    return rho
#===================================================================
def HartreePotential(basis:FEDVR_Basis, rho:np.ndarray):

    ne = basis.ne
    ng = basis.ng
    grid = basis.GetGridpoints()

    A_integ = np.zeros_like(grid)
    B_integ = np.zeros_like(grid)

    A_tot_integ = 0.
    B_tot_integ = 0.

    for i in range(ne):
        rho_elem = rho[i*ng : i*ng + ng + 1]
        r_elem = grid[i*ng : i*ng + ng + 1]

        A_integrand = rho_elem * r_elem
        B_integrand = rho_elem * r_elem**2
        A_elem_integ = np.dot(basis.leg_integ[i, :, :], A_integrand)
        B_elem_integ = np.dot(basis.leg_integ[i, :, :], B_integrand)

        A_integ[i*ng : i*ng + ng + 1] = A_elem_integ + A_tot_integ
        B_integ[i*ng : i*ng + ng + 1] = B_elem_integ + B_tot_integ

        A_tot_integ += A_elem_integ[-1]
        B_tot_integ += B_elem_integ[-1]

    V_Ha = np.zeros_like(grid)
    V_Ha[1:] = A_integ[-1] - A_integ[1:] + B_integ[1:] / grid[1:]
    V_Ha[0] = A_integ[-1] - A_integ[0]

    return V_Ha
#===================================================================
def ExchangeCorrelationPotential(basis:FEDVR_Basis, rho:np.ndarray,
                                 xc_functional:str='',
                                 x_functional:str='gga_x_pbe', c_functional:str='gga_c_pbe',
                                 alpha_x:float=1.0, driver='internal'):
    """Computes the exchange-correlation potential using libxc"""
    grid = basis.GetGridpoints()
    ne = basis.ne
    ng = basis.ng

    drho_dr = np.zeros_like(rho)
    for i in range(ne):
        h = 0.5 * (basis.xp[i+1] - basis.xp[i])
        rho_elem = rho[i*ng : i*ng + ng + 1]
        drho_dr_elem = np.dot(basis.leg.D_ii, rho_elem) / h
        drho_dr[i*ng : i*ng + ng + 1] = drho_dr_elem


    sigma = drho_dr**2

    if driver == 'internal':
        available_functionals = ['PBE', 'PBE0', 'B3LYP']

        if xc_functional not in available_functionals:
            raise ValueError(f"Unsupported exchange-correlation functional: {xc_functional}. "
                             f"Available functionals: {available_functionals}")

        # Use internal GGA functional implementation
        exc, xc_data = gga_functional(xc_functional, rho, drho_dr, alpha_x)
        V_xc_grid = xc_data[0]
    elif driver == 'pylibxc':
        # try importing libxc
        try:
            import pylibxc
        except ImportError:
            raise ImportError("pylibxc is not installed. Please install it to use the libxc driver.")

        input_data = {"rho": rho, "sigma": sigma}

        if len(xc_functional) > 0:
            xc_func = pylibxc.LibXCFunctional(xc_functional.lower(), "unpolarized")
            output = xc_func.compute(input_data)
            V_xc_grid = np.array(output["vrho"]).flatten()
        else:
            x_func = pylibxc.LibXCFunctional(x_functional.lower(), "unpolarized")
            output = x_func.compute(input_data)
            V_x = np.array(output["vrho"]).flatten()

            c_func = pylibxc.LibXCFunctional(c_functional.lower(), "unpolarized")
            output = c_func.compute(input_data)
            V_c = np.array(output["vrho"]).flatten()

            V_xc_grid = alpha_x * V_x + V_c
    else:
        raise ValueError(f"Unsupported driver: {driver}. Available drivers: 'internal', 'pylibxc'.")

    return V_xc_grid
#===================================================================
def GetFullPotential_GGA(basis:FEDVR_Basis, rho:np.ndarray, v_rho:np.ndarray, v_sigma:np.ndarray):
    """Computes the full potential from the density and exchange-correlation potentials"""
    grid = basis.GetGridpoints()
    ne = basis.ne
    ng = basis.ng

    # Compute the gradient of the charge density
    drho_dr = np.zeros_like(rho)
    for i in range(ne):
        h = 0.5 * (basis.xp[i+1] - basis.xp[i])
        rho_elem = rho[i*ng : i*ng + ng + 1]
        drho_dr_elem = np.dot(basis.leg.D_ii, rho_elem) / h
        drho_dr[i*ng : i*ng + ng + 1] = drho_dr_elem

    div_sigma_term = np.zeros_like(grid)
    temp = grid**2 * v_sigma * drho_dr

    dtemp_dr = np.zeros_like(grid)
    for i in range(ne):
        h = 0.5 * (basis.xp[i+1] - basis.xp[i])
        temp_elem = temp[i*ng : i*ng + ng + 1]
        dtemp_dr[i*ng : i*ng + ng + 1] = np.dot(basis.leg.D_ii, temp_elem) / h

    div_sigma_term[1:] = dtemp_dr[1:] / grid[1:]**2
    div_sigma_term[0] = div_sigma_term[1]  # Avoid division by zero at r=0

    # V_full = v_rho - div_sigma_term  # full GGA XC potential
    V_full = v_rho

    return V_full
#===================================================================
def lda_x(n):
    """LDA exchange potential and energy density for spin-unpolarized system."""
    Cx = (3/4) * (3/np.pi)**(1/3)
    eps_x = -Cx * n**(1/3)
    v_x = -(4/3) * Cx * n**(1/3)
    return eps_x, v_x
