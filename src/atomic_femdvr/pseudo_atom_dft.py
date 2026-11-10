import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult, minimize_scalar

import atomic_femdvr.density_potential as density_potential
import atomic_femdvr.kohn_sham as kohn_sham
from atomic_femdvr.adaptive_elements import optimize_elements
from atomic_femdvr.atom_dft import AtomDFT
from atomic_femdvr.confinement import parabolic_confinement, soft_coulomb_potential, soft_step
from atomic_femdvr.dipoles import dipole_moments, save_dipole_moments
from atomic_femdvr.input import (
    ConfinementInput,
    ConfinementType,
    ControlInput,
    DFTInput,
    OutputInput,
    SolverInput,
    SysParamsInput,
)
from atomic_femdvr.interp_tools import interpolate_density, interpolate_potential
from atomic_femdvr.projector_output import write_projector_file
from atomic_femdvr.upf import UPFInterface


#==========================================================================
class PseudoAtomDFT(AtomDFT):
    #.......................................................
    def __init__(self, control: ControlInput, sysparams: SysParamsInput,
                 solver: SolverInput, dft: DFTInput):
        self.control = control
        self.sysparams = sysparams
        self.solver = solver
        self.dft = dft

        self._upf: UPFInterface | None = None
        self.rho_grid = None
        self.rho_nlcc = None
        self._Vloc_grid: np.ndarray | None = None

        self.Zval = 1.0  # Default value, can be set later
        self.element = self.sysparams.element

        # placeholder grid; rebuilt in read_upf() once the valence charge is known
        solver.Rmax = solver.Rmax or 30.0
        self.set_basis(optimize_elements(self.Zval, solver.h_min, solver.h_max, solver.Rmax,
                                         solver.elem_tol), solver.ng)
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
        self.upf = UPFInterface.from_upf(self.sysparams.file_upf,
                                         oc_override=self.sysparams.oc_override)

        self.Zval = self.upf.zp

        # Rebuild grid now that Zval is known. __init__ used Zval=1 as a placeholder;
        # the correct Zval is needed so that optimize_elements places a fine first
        # element near the nucleus where compact orbitals (e.g. 3d) have large amplitude.
        self.set_basis(optimize_elements(self.Zval, self.solver.h_min, self.solver.h_max,
                                         self.solver.Rmax, self.solver.elem_tol), self.solver.ng)

        # Valence channels: from the pseudo-atomic orbitals if present, otherwise the
        # angular momenta of the projectors (lowest state per channel).
        if self.upf.has_paos:
            self.lmax_pseudo = int(np.amax(self.upf.lchi))
            self.nmax_pseudo = int(np.amax(self.upf.nnodes_chi))
        else:
            self.lmax_pseudo = int(np.amax(self.upf.lll)) if self.upf.nbeta > 0 else 0
            self.nmax_pseudo = 0

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
    def occupied_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.upf.has_paos:
            raise ValueError("The UPF file contains no pseudo-atomic orbitals (PP_PSWFC), so the "
                             "occupations needed for self-consistency are unknown. Use the 'nscf' "
                             "task, which takes the density from PP_RHOATOM.")
        return self.upf.nnodes_chi, self.upf.lchi, self.upf.oc
    #.......................................................
    def principal_quantum_numbers(self) -> dict[int, int]:
        """Principal quantum number of the lowest pseudo-orbital per l (e.g. {0: 4} for Mo 4s)."""
        n_lowest: dict[int, int] = {}
        for n, l in zip(self.upf.nchi, self.upf.lchi, strict=True):
            if n > l:  # skip missing/invalid labels in the UPF file
                n_lowest[int(l)] = min(n_lowest.get(int(l), int(n)), int(n))
        return n_lowest
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
    def get_confinement(self, confinement: ConfinementInput) -> np.ndarray | None:
        """
        Returns the confinement potential based on the specified type.
        """
        ri = confinement.ri_factor * confinement.rc

        if confinement.type == ConfinementType.SOFTSTEP:
            return soft_step(self.grid, ri, confinement.rc, Vbarrier=confinement.Vbarrier)
        elif confinement.type == ConfinementType.HARMONIC:
            return parabolic_confinement(self.grid, ri, confinement.rc)
        return None
    #.......................................................
    def get_all_states(self, lmax: int, nmax: int, confinement: ConfinementInput | None = None):
        """
        Returns all bound states including unbound states.
        """
        V_eff = self.get_effective_potential()

        if confinement:
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

                eps_unbound, _ = self.solve_schrodinger(V_eff, lmax, nmax,
                                                                 Vconf=Vconf, lmin=self.lmax_pseudo + 1)

                print("Solving for unbound states with soft Coulomb potential...")
                _, psi_softcoul = self.solve_schrodinger(Vsoftcoul, lmax, nmax,
                                                                 Vconf=Vconf, lmin=self.lmax_pseudo + 1)

                # Combine bound and unbound states. For the polarization channels, the
                # orbitals come from the soft-Coulomb potential (localized shape), but the
                # eigenvalues from the confined KS potential: they are the selection criterion
                # for PAOs of a low-energy subspace in downstream applications (the
                # soft-Coulomb eigenvalues would be artificially negative).
                eps = np.zeros([lmax + 1, nmax + 1], dtype=np.float64)
                psi = np.zeros([lmax + 1, nmax + 1, self.num_grid], dtype=np.float64)
                eps[:self.lmax_pseudo+1, :] = eps_bound
                psi[:self.lmax_pseudo+1, :, :] = psi_bound
                eps[self.lmax_pseudo+1:lmax+1, :] = eps_unbound
                psi[self.lmax_pseudo+1:lmax+1, :, :] = psi_softcoul
            else:
                raise ValueError(f"Unknown polarization mode: {confinement.polarization_mode}")

        else:
            eps, psi = self.solve_schrodinger(V_eff, lmax, nmax)

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
        V_eff = self.get_effective_potential()
        eps_unconf, _ = self.solve_schrodinger(V_eff, self.lmax_pseudo, self.nmax_pseudo)

        eigenvalues_all, psi_all = self.get_all_states(lmax, nmax, confinement=confinement)

        energy_shifts = np.zeros(self.lmax_pseudo + 1, dtype=np.float64)

        for l in range(self.lmax_pseudo + 1):
            tag = f'{l}'
            l_mask = self.upf.lchi == l
            occ_mask = l_mask & (self.upf.oc > 0)
            # fall back to the highest pseudo-orbital for this l if none is occupied
            # (e.g. an unoccupied polarization/scattering channel in the UPF)
            # and to the lowest state if the UPF has no orbital for this l (or none at all)
            mask = occ_mask if np.any(occ_mask) else l_mask
            n = int(np.amax(self.upf.nnodes_chi[mask])) if np.any(mask) else 0

            epsl_all = np.array(eigenvalues_all[tag])
            energy_shifts[l] = epsl_all[n] - eps_unconf[l, n]

        return energy_shifts, eigenvalues_all, psi_all
    #.......................................................
    def _build_basis_tag(self, lmax: int, nmax: int, confinement: ConfinementInput | None = None,
                         include_confinement: bool = True) -> str:
        # following quantum chemistry conventions: SZ, DZP, TZPP, etc.
        prefs = ['S', 'D', 'T', 'Q', 'H']
        zeta_tag = f"{prefs[nmax]}Z" if nmax < len(prefs) else f"{nmax}Z"

        assert self.upf is not None
        extra_l = lmax - self.lmax_pseudo
        p_tag = 'P' * extra_l if extra_l > 0 else ''

        tag = zeta_tag + p_tag

        if include_confinement and confinement is not None and confinement.type != ConfinementType.NONE:
            tag += f'_rc{confinement.rc}'

        return tag
    #.......................................................
    def export_eigenvalues(self, eigenvalues: dict[str, list[float]], out_dir:str,
                           lmax: int, nmax: int):
        elem = self.element
        basis_tag = self._build_basis_tag(lmax, nmax, include_confinement=False)
        file_eigenvalues = os.path.join(out_dir, f"{elem}_{basis_tag}_eigenvalues.dat")

        with open(file_eigenvalues, 'w') as f:
            f.write("# l  n  eigenvalue (Hartree)\n")
            for l_str, eps_list in eigenvalues.items():
                l = int(l_str)
                for n, eps in enumerate(eps_list):
                    f.write(f"{l:2d} {n:2d} {eps:16.8f}\n")
    #.......................................................
    def export_projectors(self, lmax:int, nmax:int, psi:np.ndarray, confinement: ConfinementInput,
                          output: OutputInput, out_dir:str):

        elem = self.element
        tag = self._build_basis_tag(lmax, nmax, confinement=confinement, include_confinement=True)

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
    def export_dipole_moments(self, lmax:int, nmax:int, psi:np.ndarray, output: OutputInput, out_dir:str):
        orb_indx, D_matrix = dipole_moments(self.basis, psi)

        elem = self.element
        file_dipoles = os.path.join(out_dir, f"{elem}_dipole_moments.h5")

        save_dipole_moments(file_dipoles, orb_indx, D_matrix)
