import numpy as np

from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.xc_functionals import gga_functional


#===================================================================
def charge_density(basis:FEDVR_Basis, nnodes_chi:np.ndarray, lchi:np.ndarray,
                  occ:np.ndarray, psi:np.ndarray) -> np.ndarray:
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

#===================================================================
def hartree_potential(basis:FEDVR_Basis, rho:np.ndarray) -> np.ndarray:

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
def exchange_correlation_potential(basis:FEDVR_Basis, rho:np.ndarray,
                                   rho_nlcc:np.ndarray=np.array([]),
                                   xc_functional:str='', x_functional:str='gga_x_pbe', 
                                   c_functional:str='gga_c_pbe',
                                   alpha_x:float=1.0, driver='internal') -> np.ndarray:
    """Computes the exchange-correlation potential using libxc"""
    grid = basis.GetGridpoints()
    ne = basis.ne
    ng = basis.ng

    if len(rho_nlcc) == len(rho):
        rho_ = rho + rho_nlcc
    else:
        rho_ = rho

    drho_dr = np.zeros_like(rho_)
    for i in range(ne):
        h = 0.5 * (basis.xp[i+1] - basis.xp[i])
        rho_elem = rho_[i*ng : i*ng + ng + 1]
        drho_dr_elem = np.dot(basis.leg.D_ii, rho_elem) / h
        drho_dr[i*ng : i*ng + ng + 1] = drho_dr_elem


    sigma = drho_dr**2

    if driver == 'internal':
        available_functionals = ['PBE', 'PBE0', 'B3LYP']

        if xc_functional not in available_functionals:
            raise ValueError(f"Unsupported exchange-correlation functional: {xc_functional}. "
                             f"Available functionals: {available_functionals}")

        # Use internal GGA functional implementation
        exc, xc_data = gga_functional(xc_functional, rho_, drho_dr, alpha_x)
        V_xc_grid = xc_data[0]
    elif driver == 'pylibxc':
        # try importing libxc
        try:
            import pylibxc
        except ImportError:
            raise ImportError("pylibxc is not installed. Please install it to use the libxc driver.")

        input_data = {"rho": rho_, "sigma": sigma}

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
