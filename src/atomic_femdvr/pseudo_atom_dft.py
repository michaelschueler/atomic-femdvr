import os
import h5py

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult, minimize_scalar

import atomic_femdvr.density_potential as density_potential
import atomic_femdvr.kohn_sham as kohn_sham
from atomic_femdvr.adaptive_elements import optimize_elements
from atomic_femdvr.confinement import parabolic_confinement, soft_coulomb_potential, soft_step
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.input import (
    ControlInput,
    ConfinementInput,
    ConfinementType,
    DFTInput,
    SolverInput,
    SysParamsInput,
    OutputInput
)
from atomic_femdvr.interp_tools import interpolate_density, interpolate_potential
from atomic_femdvr.projector_output import write_projector_file
from atomic_femdvr.upf import UPFInterface


#==========================================================================
class PseudoAtomDFT:
    #.......................................................
    def __init__(self, control: ControlInput, sysparams: SysParamsInput, 
                 solver: SolverInput, dft: DFTInput):
        self.control = control
        self.sysparams = sysparams
        self.solver = solver
        self.dft = dft

        self._upf: UPFInterface | None = None
        self.rho_grid = None
        self._Vloc_grid: np.ndarray | None = None

        self.Zval = 1.0  # Default value, can be set later
        self.element = self.sysparams.element

        # optimize elements
        solver.Rmax = solver.Rmax or 30.0
        self.r_elements = optimize_elements(self.Zval, solver.h_min, solver.h_max, solver.Rmax, solver.elem_tol)

        # set up the basis
        ne = len(self.r_elements) - 1
        self.basis = FEDVR_Basis(ne, solver.ng, self.r_elements,
                                build_derivatives=True, build_integrals=True)

        self.grid = self.basis.get_gridpoints()

        self.num_grid = len(self.grid)
    #.......................................................

    @property
    def upf(self) -> UPFInterface:
        if self._upf is None:
            raise ValueError("UPF file has not been read yet. Call ReadUPF() first.")
        return self._upf

    @upf.setter
    def upf(self, value: UPFInterface) -> None:
        self._upf = value

    @property
    def Vloc_grid(self) -> np.ndarray:
        if self._Vloc_grid is None:
            raise ValueError("Local potential has not been read yet. Call ReadUPF(read_potential=True) first.")
        return self._Vloc_grid

    @Vloc_grid.setter
    def Vloc_grid(self, value: np.ndarray) -> None:
        self._Vloc_grid = value

    def read_upf(self, read_density: bool = True, read_potential: bool = True):
        assert self.sysparams.file_upf is not None
        self.upf = UPFInterface.from_upf(self.sysparams.file_upf)

        self.Zval = self.upf.zp
        self.lmax_pseudo = np.amax(self.upf.lchi)
        self.nmax_pseudo = np.amax(self.upf.nnodes_chi)

        if read_density:
            rho_upf = self.upf.get_charge_density()
            self.rho_grid = interpolate_density(self.upf.r, rho_upf, self.grid)

        if read_potential:
            Vloc_upf = self.upf.vloc
            self.Vloc_grid = interpolate_potential(self.upf.r, Vloc_upf, self.grid)
            self.Vloc_grid *= 0.5 # Convert to Hartree units

        if read_density and self.upf.rho_nlcc is not None:
            self.rho_nlcc = interpolate_density(self.upf.r, self.upf.rho_nlcc, self.grid)

        # interpolate beta projectors to new grid
        self.nbeta = self.upf.nbeta
        beta = np.ascontiguousarray(self.upf.beta.T)
        kbeta_max = np.max(self.upf.kbeta)
        interp = interp1d(self.upf.r[0:kbeta_max], beta[:, 0:kbeta_max], axis=1,
                          kind='cubic', bounds_error=False, fill_value=0.0)
        self.beta_grid = interp(self.grid)

    #.......................................................
    def get_effective_potential(self, rho_grid:np.ndarray | None=None) -> np.ndarray:
        if rho_grid is None:
            rho_grid = self.rho_grid

        V_Ha = density_potential.hartree_potential(self.basis, rho_grid)

        V_xc = density_potential.exchange_correlation_potential(self.basis, rho_grid,
                                                    rho_nlcc=self.rho_nlcc,
                                                   xc_functional=self.dft.xc_functional,
                                                   x_functional=self.dft.x_functional,
                                                   c_functional=self.dft.c_functional,
                                                   alpha_x=self.dft.alpha_x,
                                                   driver=self.dft.driver)

        V_eff = self.Vloc_grid + V_Ha + V_xc
        return V_eff
    #.......................................................
    def solve_schrodinger(self, Veff: np.ndarray, lmax: int, nmax: int,
                          Vconf: np.ndarray | None = None, lmin: int = 0,
                          non_local: bool = True) -> tuple[np.ndarray, np.ndarray]:

        if non_local:
            eps, psi = kohn_sham.solve_schrodinger_pseudo(self.basis, Veff, self.upf.lll, self.upf.dion,
                                           self.beta_grid, lmax, nmax, Vconf=Vconf, lmin=lmin)
        else:
            eps, psi = kohn_sham.solve_schrodinger_local(self.basis, Veff, lmax, nmax, Vconf=Vconf, lmin=lmin)

        return eps, psi
    #.......................................................
    def get_bound_states(self):

        V_eff = self.get_effective_potential()
        eps, psi = self.solve_schrodinger(V_eff, self.lmax_pseudo, self.nmax_pseudo)

        eigenvalues = {}
        for l in range(self.lmax_pseudo + 1):
            Ie, = np.where(eps[l, :self.nmax_pseudo+1] < 0)
            eps_bound = eps[l, Ie]
            tag = f'{l}'
            eigenvalues[tag] = eps_bound.tolist()

        return eigenvalues, psi
    #.......................................................
    def get_confinement(self, confinement: ConfinementInput):
        """
        Returns the confinement potential based on the specified type.
        """
        ri = confinement.ri_factor * confinement.rc

        if confinement.type == ConfinementType.SOFTSTEP:
            return soft_step(self.grid, ri, confinement.rc, Vbarrier=confinement.Vbarrier)
        elif confinement.type == ConfinementType.HARMONIC:
            return parabolic_confinement(self.grid, ri, confinement.rc)
        else:
            raise ValueError(f"Unknown confinement type: {confinement.type}")
    #.......................................................
    def get_all_states(self, lmax: int, nmax: int, confinement: ConfinementInput | None = None):
        """
        Returns all bound states including unbound states.
        """
        V_eff = self.get_effective_potential()

        if confinement:
            ri = confinement.ri_factor * confinement.rc
            # Vconf = SoftConfinement(self.grid, ri, rc)

            Vconf = self.get_confinement(confinement)

            if confinement.polarization_mode is None:
                eps, psi = self.solve_schrodinger(V_eff, lmax, nmax, Vconf=Vconf)
            elif confinement.polarization_mode.lower() == 'softcoul':

                # solve first for the bound states
                eps_bound, psi_bound = self.solve_schrodinger(V_eff, self.lmax_pseudo, nmax,
                                                             Vconf=Vconf)

                # now solve remaining l-channels with soft Coulomb potential
                Vsoftcoul = soft_coulomb_potential(self.grid, confinement.softcoul_charge,
                                                  confinement.softcoul_delta)

                print("Solving for unbound states with soft Coulomb potential...")
                eps_unbound, psi_unbound = self.solve_schrodinger(Vsoftcoul, lmax, nmax,
                                                                 Vconf=Vconf, lmin=self.lmax_pseudo + 1)

                # combine bound and unbound states
                eps = np.zeros([lmax + 1, nmax + 1], dtype=np.float64)
                psi = np.zeros([lmax + 1, nmax + 1, self.num_grid], dtype=np.float64)
                eps[:self.lmax_pseudo+1, :] = eps_bound
                psi[:self.lmax_pseudo+1, :, :] = psi_bound
                eps[self.lmax_pseudo+1:lmax+1, :] = eps_unbound
                psi[self.lmax_pseudo+1:lmax+1, :, :] = psi_unbound
            else:
                raise ValueError(f"Unknown polarization mode: {confinement.polarization_mode}")

        else:
            eps, psi = self.SolveSchrodinger(V_eff, lmax, nmax)

        eigenvalues = {}
        for l in range(lmax + 1):
            tag = f'{l}'
            eigenvalues[tag] = eps[l, :].tolist()

        return eigenvalues, psi
    #.......................................................
    def optimize_soft_coul(self, confinement: ConfinementInput):
        """
        Optimize the soft Coulomb potential parameters for a given lmax and nmax.
        """
        if confinement.polarization_mode != 'softcoul':
            raise ValueError("Polarization mode must be 'softcoul' for this method.")

        _, psi_bound = self.get_bound_states()
        psi_ref = psi_bound[-1, 0, :]  # Use the state with highest l as reference

        Vconf = self.get_confinement(confinement)

        # wrapper for the optimization function
        def objective_func(Q: float) -> float:

            # Set up the soft Coulomb potential
            Vsoftcoul = soft_coulomb_potential(self.grid, Q, confinement.softcoul_delta)

            _, psi = self.solve_schrodinger(Vsoftcoul, self.lmax_pseudo, 1,
                                             Vconf=Vconf, lmin=self.lmax_pseudo)

            ovlp = np.abs(self.basis.get_overlap(psi_ref, psi[0, 0, :]))**2

            # print(f"Q: {Q:.4f}, Overlap: {ovlp:.4f}")

            return 1.0 - ovlp  # We want to maximize the overlap

        # Optimize the charge Q
        Qmin = 0.2 * self.Zval
        Qmax = 10.0 * self.Zval
        result = minimize_scalar(objective_func, bounds=(Qmin, Qmax), method='bounded')  # type: ignore
        assert isinstance(result, OptimizeResult)
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        Q_opt = result.x

        return Q_opt
    #.......................................................
    def get_states_energy_shift(self, lmax:int, nmax:int, confinement: ConfinementInput):
        eigenvalues_bounds, psi_bound = self.get_bound_states()
        eigenvalues_all, psi_all = self.get_all_states(lmax, nmax, confinement=confinement)

        energy_shifts = np.zeros(self.lmax_pseudo + 1, dtype=np.float64)

        for l in range(self.lmax_pseudo + 1):
            tag = f'{l}'
            epsl_bound = np.array(eigenvalues_bounds[tag])
            epsl_all = np.array(eigenvalues_all[tag])
            n = np.argmax(epsl_bound)
            energy_shifts[l] = epsl_all[n] - epsl_bound[n]

        return energy_shifts, eigenvalues_all, psi_all
    #.......................................................
    def export_projectors(self, lmax:int, nmax:int, psi:np.ndarray, confinement: ConfinementInput,
                          output: OutputInput, out_dir:str):

        # here we build the tag for the output files
        # following quantum chemistry conventions: SZ, DZP etc.

        elem = self.element
        prefs = ['S', 'D', 'T', 'Q', 'H']
        zeta_tag = f"{prefs[nmax]}Z" if nmax < len(prefs) else f"{nmax}Z"
        
        assert self.upf is not None
        lmax_upf = np.amax(self.upf.lchi)
        extra_l = lmax - lmax_upf
        if extra_l > 0:
            p_tag = 'P' * extra_l
        else:
            p_tag = ''

        tag = zeta_tag + p_tag

        # add confinement info to tag
        if confinement.type != ConfinementType.NONE:
            tag += f'_rc{confinement.rc}'

        if output.output_wfc_qe:
            nr = output.qe_num_points
            rmin = output.qe_rmin
            write_projector_file(self.basis, psi, elem, tag, nr=nr, rmin=rmin,
                                 out_dir=out_dir, output_format='qe')

        if output.output_wfc_hdf5:
            write_projector_file(self.basis, psi, elem, tag, 
                                 out_dir=out_dir, output_format='hdf5')
            
        if output.output_wfc_bessel:
            npoints = output.bessel_quad_npoints
            method = output.bessel_quad_method
            qmax = output.bessel_qmax
            nq = output.bessel_nq
            rpow = output.bessel_rpow

            qgrid = np.linspace(0.0, qmax, nq)

            write_projector_file(self.basis, psi, elem, tag, 
                                 bessel_method=method, bessel_npoints=npoints,
                                 qgrid=qgrid, rpow=rpow,
                                 out_dir=out_dir, output_format='bessel')
    #.......................................................
    def ks_self_consistency(self, max_iter:int=100, tol:float=1.0e-6, alpha_mix:float=0.6):
        """
        Performs Kohn-Sham self-consistency to find the ground state density.
        """
        V_eff = self.get_effective_potential()
        lmax = np.amax(self.upf.lchi)
        nmax = np.amax(self.upf.nnodes_chi)

        # Initial guess for the wavefunctions
        eps, psi = self.solve_schrodinger(V_eff, lmax, nmax)

        rho_old = self.rho_grid.copy()

        err = 1.0e8
        iter_count = 0
        while err > tol and iter_count < max_iter:
            iter_count += 1

            # Compute charge density
            rho_new = density_potential.charge_density(self.basis, self.upf.nnodes_chi,
                                                      self.upf.lchi, self.upf.oc, psi)

            # linear mixing of the density
            rho_new = alpha_mix * rho_new + (1 - alpha_mix) * rho_old

            # Update effective potential
            V_eff = self.get_effective_potential(rho_grid=rho_new)

            # Compute error
            err = np.linalg.norm(rho_new - rho_old)

            # Solve Schrödinger equation with new potential
            eps, psi = self.solve_schrodinger(V_eff, lmax, nmax)

            rho_old = rho_new.copy()


        self.rho_grid = rho_new.copy()

        return iter_count, err

    #.......................................................
    def save_density_potential(self):
        """
        Saves the charge density and potential to a file.
        """
        V_eff = self.get_effective_potential()

        if not os.path.exists(self.control.storage_dir):
            os.makedirs(self.control.storage_dir)

        filename = f'{self.element}_density_potential.h5'
        with h5py.File(self.control.storage_dir / filename, 'w') as f:
            f.create_dataset('grid', data=self.grid)
            f.create_dataset('rho', data=self.rho_grid)
            f.create_dataset('Veff', data=V_eff)
    #.......................................................
    def read_density_potential(self):
        """
        Reads the charge density and potential from a file.
        """
        storage_dir = self.control.storage_dir
        filename = f'{self.element}_density_potential.h5'
        filepath = os.path.join(storage_dir, filename)

        if not os.path.isfile(filepath):
            return False

        with h5py.File(filepath, 'r') as f:
            grid = f['grid'][:]
            rho_grid = f['rho'][:]
            Veff_grid = f['Veff'][:]

        if len(grid) != self.num_grid:
            restart_success = False
        else:
            restart_success = np.allclose(grid, self.grid)

        if restart_success:
            self.rho_grid = rho_grid.copy()

        return restart_success