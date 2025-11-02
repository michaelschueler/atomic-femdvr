import os
import pickle

import numpy as np
from scipy.interpolate import interp1d


import atomic_femdvr.density_potential as density_potential
import atomic_femdvr.kohn_sham as kohn_sham
from atomic_femdvr.adaptive_elements import optimize_elements
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.input import (
    DFTInput,
    SolverInput,
    SysParamsInput,
    ElectronsInput
)
from atomic_femdvr.interp_tools import interpolate_density, interpolate_potential
from atomic_femdvr.diis import DIIS

#==========================================================================
class FullAtomDFT:
    #.......................................................
    def __init__(self, sysparams: SysParamsInput, electrons: ElectronsInput, solver: SolverInput, dft: DFTInput):
        self.electrons = electrons
        self.sysparams = sysparams
        self.solver = solver
        self.dft = dft

        self.validate_configuration()

        self.rho_grid = None

        self.Z = self.electrons.Z

        # optimize elements
        solver.Rmax = solver.Rmax or 50 * (self.lmax + 1)
        # rescale grid limits based on Z
        h_min = solver.h_min / (self.Z ** (1/3))
        h_max = solver.h_max / (self.Z ** (1/3))
        Rmax = solver.Rmax / (self.Z ** (1/3))

        self.r_elements = optimize_elements(self.Z, h_min, h_max, Rmax, solver.elem_tol)

        # set up the basis
        ne = len(self.r_elements) - 1
        self.basis = FEDVR_Basis(ne, solver.ng, self.r_elements, 
                                build_derivatives=True, build_integrals=True)

        self.grid = self.basis.GetGridpoints()
        self.num_grid = len(self.grid)

        self.V0_grid = np.zeros(self.num_grid, dtype=np.float64)
        self.V0_grid[1:] = -self.Z / self.grid[1:]
        self.V0_grid[0] = self.V0_grid[1]  # avoid singularity at r=0

    #.......................................................
    def validate_configuration(self):
        shell_labels = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

        self.lmax = 0

        # check shell labels
        shells = self.electrons.elect_config.keys()
        for shell in shells:
            l_char = shell[0].upper()
            if l_char not in shell_labels:
                raise ValueError(f"Invalid shell label: {shell}")

        # check the occupation of each shell
        self.ll = []
        self.nn = []
        self.occ = []
        for shell in shells:
            l_char = shell[0].upper()
            l = shell_labels.index(l_char)
        
            self.ll.append(l)
            n_list = np.arange(0, len(self.electrons.elect_config[shell]), dtype=int)
            self.nn.extend(n_list.tolist())

            max_occ = 2 * (2 * l + 1)
            occ_list = self.electrons.elect_config[shell]
            self.nrad_vals[l] = len(occ_list)

            for occ in occ_list:
                if occ < 0.0 or occ > max_occ:
                    raise ValueError(f"Invalid occupation {occ} for shell {shell}. Max occupation is {max_occ}.")

        self.ll = np.array(self.ll, dtype=int)
        self.nn = np.array(self.nn, dtype=int)
        self.occ = np.array(self.occ, dtype=float)

        self.nmax = np.amax(self.nn)
        self.lmax = np.amax(self.ll)
        self.num_electrons = np.sum(self.occ)
    #.......................................................
    def get_effective_potential(self, rho_grid:np.ndarray | None=None) -> np.ndarray:
        if rho_grid is None:
            rho_grid = self.rho_grid


        V_Ha = density_potential.hartree_potential(self.basis, rho_grid)

        V_xc = density_potential.exchange_correlation_potential(self.basis, rho_grid,
                                                   xc_functional=self.dft.xc_functional,
                                                   x_functional=self.dft.x_functional,
                                                   c_functional=self.dft.c_functional,
                                                   alpha_x=self.dft.alpha_x,
                                                   driver=self.dft.driver)

        V_xc *= 0.5 # Convert to Hartree units

        V_eff = self.V0_grid + V_Ha + V_xc
        return V_eff
    #.......................................................
    def solve_schrodinger(self, Veff: np.ndarray, lmax: int, nmax: int, 
                          Vconf: np.ndarray | None = None, lmin: int = 0):

        eps, psi = kohn_sham.solve_schrodinger_local(self.basis, Veff, lmax, nmax,
                                                     lmin=lmin, solver=self.solver.eigensolver)

        return eps, psi
    #.......................................................
    def get_bound_states(self):

        V_eff = self.get_effective_potential()
        eps, psi = self.solve_schrodinger(V_eff, self.lmax, self.nmax)

        eigenvalues = {}
        for l in range(self.lmax + 1):
            Ie, = np.where(eps[l, :self.nmax+1] < 0)
            eps_bound = eps[l, Ie]
            tag = f'{l}'
            eigenvalues[tag] = eps_bound.tolist()

        return eigenvalues, psi
    #.......................................................


    #.......................................................
    def ks_self_consistency(self):
        """
        Performs Kohn-Sham self-consistency to find the ground state density.
        """
        max_iter = self.dft.max_iter
        tol = self.dft.conv_tol
        alpha_mix = self.dft.alpha_mix

        mixing_scheme = self.dft.mixing_scheme.lower()
        if mixing_scheme.lower() == 'diis':
            diis_history = self.dft.diis_history
            diis = DIIS(max_history=diis_history)

        V_eff = self.effective_potential()

        # Initial guess for the wavefunctions
        eps, psi = self.solve_schrodinger(V_eff, self.lmax, self.nmax)

        rho = self.rho_grid.copy()

        err = 1.0e8
        iter_count = 0
        while err > tol and iter_count < max_iter:
            iter_count += 1

            # Compute charge density
            rho_out = density_potential.charge_density(self.basis, self.nn, self.ll, self.occ, psi)

            if mixing_scheme.lower() == 'diis':
                r = rho_out - rho
                diis.update(rho, r)

                if iter_count > 1:
                    rho = diis.extrapolate(dot_product=lambda a, b: np.dot(a, b))
                else:
                    rho = rho_out.copy()

                err = np.linalg.norm(r)
            else:
                # linear mixing of the density
                rho = alpha_mix * rho_out + (1 - alpha_mix) * rho
                err = np.linalg.norm(rho - rho_out)

            # Update effective potential
            V_eff = self.effective_potential(rho_grid=rho)

            # Solve Schrödinger equation with new potential
            eps, psi = self.solve_schrodinger(V_eff, self.lmax, self.nmax)

        self.rho_grid = rho.copy()

        return iter_count, err
    #.......................................................

    #.......................................................
    def save_density_potential(self):
        """
        Saves the charge density and potential to a file.
        """
        V_eff = self.effective_potential()

        data = {
            'grid': self.grid,
            'rho': self.rho_grid,
            'Veff': V_eff
        }

        if not os.path.exists(self.pseudo_config.storage_dir):
            os.makedirs(self.pseudo_config.storage_dir)

        filename = 'density_potential.pkl'
        with open(self.pseudo_config.storage_dir / filename, 'wb') as f:
            pickle.dump(data, f)
    #.......................................................
    def read_density_potential(self):
        """
        Reads the charge density and potential from a file.
        """
        storage_dir = self.pseudo_config.storage_dir
        filename = 'density_potential.pkl'
        filepath = os.path.join(storage_dir, filename)

        if not os.path.isfile(filepath):
            return False

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        grid = data['grid']
        rho_grid = data['rho']
        Veff_grid = data['Veff']

        # check if grid matches
        if len(grid) != self.num_grid:
            restart_success = False
        else:
            restart_success = np.allclose(grid, self.grid)

        if restart_success:
            self.rho_grid = rho_grid.copy()

        return restart_success
