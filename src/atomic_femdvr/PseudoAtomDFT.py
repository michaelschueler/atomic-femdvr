import os
import pickle

import atomic_femdvr.DensityPotential as denpot
import atomic_femdvr.KohnSham as ks
import numpy as np
from atomic_femdvr.adaptive_elements import OptimizeElements
from atomic_femdvr.Confinement import ParabolicConfinement, SoftCoulombPotential, SoftStep
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.interp_tools import InterpolateDensity, InterpolatePotential
from atomic_femdvr.Projectors import WriteProjectorFile
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from atomic_femdvr.upf_interface import upf_class

from atomic_femdvr.input import PseudoConfigInput, SysParamsInput, SolverInput, ConfinementInput


#==========================================================================
class PseudoAtomDFT:
    #.......................................................
    def __init__(self, pseudo_config: PseudoConfigInput, sysparams: SysParamsInput, solver: SolverInput, dft: dict):
        self.pseudo_config = pseudo_config
        self.sysparams = sysparams
        self.solver = solver
        self.dft = dft

        self.upf = None
        self.rho_grid = None
        self.Vloc_grid = None

        self.Zval = 1.0  # Default value, can be set later

        # optimize elements
        solver.Rmax = solver.Rmax or 30.0
        self.r_elements = OptimizeElements(self.Zval, solver.h_min, solver.h_max, solver.Rmax, solver.elem_tol)

        # set up the basis
        ne = len(self.r_elements) - 1
        self.basis = FEDVR_Basis(ne, solver.ng, self.r_elements, build_integrals=True)

        self.grid = self.basis.GetGridpoints()

        self.num_grid = len(self.grid)
    #.......................................................
    def ReadUPF(self, read_density: bool = True, read_potential: bool = True):
        self.upf = upf_class(self.pseudo_config.upflib_dir, self.pseudo_config.lib_ext)
        self.upf.Read_UPF(self.sysparams.file_upf)
        self.upf.ReadWavefunctions()
        self.upf.Read_PP()

        self.Zval = self.upf.zp
        self.lmax_pseudo = np.amax(self.upf.lchi)
        self.nmax_pseudo = np.amax(self.upf.nnodes_chi)

        if read_density:
            rho_upf = self.upf.GetChargeDensity()
            self.rho_grid = InterpolateDensity(self.upf.r, rho_upf, self.grid)

        if read_potential:
            Vloc_upf = self.upf.vloc
            self.Vloc_grid = InterpolatePotential(self.upf.r, Vloc_upf, self.grid)
            self.Vloc_grid *= 0.5 # Convert to Hartree units

        # interpolate beta projectors to new grid
        self.nbeta = self.upf.nbeta
        beta = np.ascontiguousarray(self.upf.beta.T)
        kbeta_max = self.upf.kbeta_max
        interp = interp1d(self.upf.r[0:kbeta_max], beta[:, 0:kbeta_max], axis=1,
                          kind='cubic', bounds_error=False, fill_value=0.0)
        self.beta_grid = interp(self.grid)

    #.......................................................
    def EffectivePotential(self, rho_grid=None):
        if rho_grid is None:
            rho_grid = self.rho_grid


        V_Ha = denpot.HartreePotential(self.basis, rho_grid)

        V_xc = denpot.ExchangeCorrelationPotential(self.basis, rho_grid,
                                                   xc_functional=self.dft.xc_functional,
                                                   x_functional=self.dft.x_functional,
                                                   c_functional=self.dft.c_functional,
                                                   alpha_x=self.dft.alpha_x,
                                                   driver=self.dft.driver)

        V_xc *= 0.5 # Convert to Hartree units

        V_eff = self.Vloc_grid + V_Ha + V_xc
        return V_eff
    #.......................................................
    def SolveSchrodinger(self, Veff, lmax, nmax, Vconf=None, lmin=0):

        eps, psi = ks.SolveSchrodinger(self.basis, Veff, self.upf.lll, self.upf.dion,
                                       self.beta_grid, lmax, nmax, Vconf=Vconf, lmin=lmin)

        return eps, psi
    #.......................................................
    def GetBoundStates(self):

        V_eff = self.EffectivePotential()
        eps, psi = self.SolveSchrodinger(V_eff, self.lmax_pseudo, self.nmax_pseudo)

        eigenvalues = {}
        for l in range(self.lmax_pseudo + 1):
            Ie, = np.where(eps[l, :self.nmax_pseudo+1] < 0)
            eps_bound = eps[l, Ie]
            tag = f'{l}'
            eigenvalues[tag] = eps_bound.tolist()

        return eigenvalues, psi
    #.......................................................
    def GetConfinement(self, confinement:dict):
        """
        Returns the confinement potential based on the specified type.
        """
        rc = confinement.get('rc', 20.0)
        ri_factor = confinement.get('ri_factor', 0.9)
        ri = ri_factor * rc

        confinement_type = confinement.get('type', 'SoftStep')
        if confinement_type.lower() == 'softstep':
            Vbarrier = confinement.get('Vbarrier', 1.0)
            return SoftStep(self.grid, ri, rc, Vbarrier=Vbarrier)
        elif confinement_type.lower() == 'parabolic':
            return ParabolicConfinement(self.grid, ri, rc)
        else:
            raise ValueError(f"Unknown confinement type: {confinement_type}")
    #.......................................................
    def GetAllStates(self, lmax:int, nmax:int, confinement:dict=None):
        """
        Returns all bound states including unbound states.
        """
        V_eff = self.EffectivePotential()

        if confinement is not None:
            rc = confinement.get('rc', 20.0)
            ri_factor = confinement.get('ri_factor', 0.9)
            ri = ri_factor * rc
            # Vconf = SoftConfinement(self.grid, ri, rc)

            Vconf = self.GetConfinement(confinement)

            polarization_mode = confinement.get('polarization_mode', 'none')

            if polarization_mode.lower() == 'none':
                eps, psi = self.SolveSchrodinger(V_eff, lmax, nmax, Vconf=Vconf)
            elif polarization_mode.lower() == 'softcoul':

                # solve first for the bound states
                eps_bound, psi_bound = self.SolveSchrodinger(V_eff, self.lmax_pseudo, nmax,
                                                             Vconf=Vconf)

                # now solve remaining l-channels with soft Coulomb potential
                softcoul_delta = confinement.get('softcoul_delta', 0.1)
                softcoul_charge = confinement.get('softcoul_charge', 1.0)

                Vsoftcoul = SoftCoulombPotential(self.grid, softcoul_charge,
                                                   softcoul_delta)

                eps_unbound, psi_unbound = self.SolveSchrodinger(Vsoftcoul, lmax, nmax,
                                                                 Vconf=Vconf, lmin=self.lmax_pseudo + 1)

                # combine bound and unbound states
                eps = np.zeros([lmax + 1, nmax + 1], dtype=np.float64)
                psi = np.zeros([lmax + 1, nmax + 1, self.num_grid], dtype=np.float64)
                eps[:self.lmax_pseudo+1, :] = eps_bound
                psi[:self.lmax_pseudo+1, :, :] = psi_bound
                eps[self.lmax_pseudo+1:lmax+1, :] = eps_unbound
                psi[self.lmax_pseudo+1:lmax+1, :, :] = psi_unbound
            else:
                raise ValueError(f"Unknown polarization mode: {polarization_mode}")

        else:
            eps, psi = self.SolveSchrodinger(V_eff, lmax, nmax)

        eigenvalues = {}
        for l in range(lmax + 1):
            tag = f'{l}'
            eigenvalues[tag] = eps[l, :].tolist()

        return eigenvalues, psi
    #.......................................................
    def OptimizeSoftCoul(self, confinement: ConfinementInput):
        """
        Optimize the soft Coulomb potential parameters for a given lmax and nmax.
        """
        if confinement.polarization_mode != 'softcoul':
            raise ValueError("Polarization mode must be 'softcoul' for this method.")

        eigenvalues_bound, psi_bound = self.GetBoundStates()
        psi_ref = psi_bound[-1, 0, :]  # Use the state with highest l as reference

        Vconf = self.GetConfinement(confinement)

        # wrapper for the optimization function
        def objective_func(Q):

            # Set up the soft Coulomb potential
            Vsoftcoul = SoftCoulombPotential(self.grid, Q, confinement.softcoul_delta)

            eps, psi = self.SolveSchrodinger(Vsoftcoul, self.lmax_pseudo, 1,
                                             Vconf=Vconf, lmin=self.lmax_pseudo)

            ovlp = np.abs(self.basis.GetOverlap(psi_ref, psi[0, 0, :]))**2

            # print(f"Q: {Q:.4f}, Overlap: {ovlp:.4f}")

            return 1.0 - ovlp  # We want to maximize the overlap

        # Optimize the charge Q
        Qmin = 0.2 * self.Zval
        Qmax = 10.0 * self.Zval
        result = minimize_scalar(objective_func, bounds=(Qmin, Qmax), method='bounded')
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        Q_opt = result.x

        return Q_opt
    #.......................................................
    def GetStatesEnergyShift(self, lmax:int, nmax:int, confinement:dict):
        eigenvalues_bounds, psi_bound = self.GetBoundStates()
        eigenvalues_all, psi_all = self.GetAllStates(lmax, nmax, confinement=confinement)

        energy_shifts = np.zeros(self.lmax_pseudo + 1, dtype=np.float64)

        for l in range(self.lmax_pseudo + 1):
            tag = f'{l}'
            epsl_bound = np.array(eigenvalues_bounds[tag])
            epsl_all = np.array(eigenvalues_all[tag])
            n = np.argmax(epsl_bound)
            energy_shifts[l] = epsl_all[n] - epsl_bound[n]

        return energy_shifts, eigenvalues_all, psi_all
    #.......................................................
    def ExportProjector(self, lmax:int, nmax:int, psi:np.ndarray, confinement:dict,
                        out_dir:str, nr:int=1001, rmin=1.0e-8):
        Rmax = self.r_elements[-1]
        rs = np.logspace(np.log10(rmin), np.log10(Rmax), nr)

        larr = []
        psi_interp = np.zeros([lmax + 1, nmax + 1, nr])
        for l in range(lmax + 1):
            for n in range(nmax + 1):
                psi_interp[l, n, :] = self.basis.Interpolate(psi[l, n, :], rs)
                larr.append(l)

        nproj = len(larr)

        # write to file
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        elem = self.sysparams.get('element', 'Mo')

        prefs = ['S', 'D', 'T', 'Q', 'H']
        zeta_tag = f"{prefs[nmax]}Z" if nmax < len(prefs) else f"{nmax}Z"

        lmax_upf = np.amax(self.upf.lchi)
        extra_l = lmax - lmax_upf
        if extra_l > 0:
            p_tag = 'P' * extra_l
        else:
            p_tag = ''

        tag = zeta_tag + p_tag

        rc = confinement.get('rc', 'none')
        if rc != 'none':
            tag += f'_rc{rc}'

        psi_interp = np.reshape(psi_interp, [nproj, nr])
        WriteProjectorFile(out_dir, elem, tag, larr, psi_interp, rs)
    #.......................................................
    def KS_SelfConsistency(self, max_iter=100, tol=1.0e-6, alpha=0.6):
        """
        Performs Kohn-Sham self-consistency to find the ground state density.
        """
        V_eff = self.EffectivePotential()
        lmax = np.amax(self.upf.lchi)
        nmax = np.amax(self.upf.nnodes_chi)

        # Initial guess for the wavefunctions
        eps, psi = self.SolveSchrodinger(V_eff, lmax, nmax)

        rho_old = self.rho_grid.copy()

        err = 1.0e8
        iter_count = 0
        while err > tol and iter_count < max_iter:
            iter_count += 1

            # Compute charge density
            rho_new = denpot.ChargeDensity(self.basis, self.upf.nnodes_chi, self.upf.lchi,
                                       self.upf.oc, psi)

            # linear mixing of the density
            rho_new = alpha * rho_new + (1 - alpha) * rho_old

            # Update effective potential
            V_eff = self.EffectivePotential(rho_grid=rho_new)

            # Compute error
            err = np.linalg.norm(rho_new - rho_old)

            # Solve Schrödinger equation with new potential
            eps, psi = self.SolveSchrodinger(V_eff, lmax, nmax)

            rho_old = rho_new.copy()


        self.rho_grid = rho_new.copy()

        return iter_count, err

    #.......................................................
    def SaveDensityPotential(self):
        """
        Saves the charge density and potential to a file.
        """
        V_eff = self.EffectivePotential()

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
    def ReadDensityPotential(self):
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
