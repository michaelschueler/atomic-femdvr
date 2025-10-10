import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from femdvr import FEDVR_Basis
from adaptive_elements import OptimizeElements
from SchrodingerSolver import SolveNR
#------------------------------------------------------------
def Solve_FEMDVR(Rmax, h_min, h_max, elem_tol, ng, a0=0.1, nevals=3, lmax=2):
    Zval = 1.0

    r_elements = OptimizeElements(Zval, h_min, h_max, Rmax, elem_tol)

    ne = len(r_elements) - 1  # Number of elements
    fe = FEDVR_Basis(ne, ng, r_elements)

    r_grid = fe.GetGridpoints()

    eps_vals = np.zeros([lmax+1, nevals])
    psi = np.zeros([lmax+ 1, nevals, len(r_grid)])

    for l in range(lmax + 1):

        Vpot_fnc = lambda r: -Zval / np.sqrt(r**2 + a0**2)

        eps, r_grid, psi_l = SolveNR(r_elements, Vpot_fnc, l, nevals, ng)

        eps_vals[l] = eps
        psi[l, :, :] = psi_l

    return r_grid, eps_vals, psi
#------------------------------------------------------------
def main():

    Rmax = 80.0
    h_min = 0.5
    h_max = 4.0
    elem_tol = 1.0e-2
    ng = 16
    lmax = 2

    r_grid, eps_vals, psi = Solve_FEMDVR(Rmax, h_min, h_max, elem_tol, ng, lmax=lmax)

    # import reference data
    psi_ref = []
    for l in range(lmax + 1):
        file_ref = f'data/wavefunctions_N1000_neval3_l{l}.dat'
        if not os.path.isfile(file_ref):
            raise FileNotFoundError(f"Reference file '{file_ref}' does not exist.")
        
        data = np.loadtxt(file_ref, unpack=True)
        rs = data[0, :]
        psi_ref_l = data[1:, :]
        psi_ref.append(psi_ref_l)

    psi_ref = np.array(psi_ref)

    # Plot results
    fig, ax = plt.subplots(lmax+1, sharex=True)
    for l in range(eps_vals.shape[0]):
        for n in range(psi.shape[1]):
            ax[l].plot(r_grid, psi[l, n, :], label=f'l={l}, E={eps_vals[l, n]:.3f}')
            ax[l].plot(rs, psi_ref[l, n, :], 'k--', label='Reference')

        ax[l].legend()


    ax[-1].set_xlabel('r')
    plt.show()
#------------------------------------------------------------
if __name__ == "__main__":
    main()
