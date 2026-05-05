import logging
import os

import h5py
import numpy as np

import atomic_femdvr.density_potential as density_potential
import atomic_femdvr.kohn_sham as kohn_sham
from atomic_femdvr.adaptive_elements import optimize_elements
from atomic_femdvr.anderson import AndersonMixing
from atomic_femdvr.diis import DIIS
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.initial_density import get_slater_density
from atomic_femdvr.input import ControlInput, DFTInput, ElectronsInput, SolverInput, SysParamsInput
from atomic_femdvr.periodic_table import PeriodicTable

logger = logging.getLogger(__name__)


# ==========================================================================
class FullAtomDFT:
    # .......................................................
    def __init__(
        self,
        control: ControlInput,
        sysparams: SysParamsInput,
        electrons: ElectronsInput,
        solver: SolverInput,
        dft: DFTInput,
    ):
        self.control = control
        self.electrons = electrons
        self.sysparams = sysparams
        self.solver = solver
        self.dft = dft

        self.validate_configuration()

        self.rho_grid = None

        self.element = self.sysparams.element

        periodic_table = PeriodicTable()
        Z_ = float(periodic_table.get_atomic_number(self.element))
        if self.electrons.Z == 0.0:
            self.Z = Z_
        else:
            self.Z = self.electrons.Z

        if np.abs(self.Z - Z_) / Z_ > 1.0e-5:
            logger.warning(
                "Specified Z = %s differs from atomic number of element %s (Z = %s). Using Z = %s.",
                self.electrons.Z,
                self.element,
                Z_,
                self.Z,
            )

        # optimize elements
        solver.Rmax = solver.Rmax or 50 * (self.lmax + 1)
        # rescale grid limits based on Z
        h_min = solver.h_min / (self.Z ** (1 / 3))
        h_max = solver.h_max / (self.Z ** (1 / 3))
        Rmax = solver.Rmax / (self.Z ** (1 / 3))

        self.r_elements = optimize_elements(self.Z, h_min, h_max, Rmax, solver.elem_tol)

        # set up the basis
        ne = len(self.r_elements) - 1
        self.basis = FEDVR_Basis(
            ne, solver.ng, self.r_elements, build_derivatives=True, build_integrals=True
        )

        self.grid = self.basis.get_gridpoints()
        self.num_grid = len(self.grid)

        self.V0_grid = np.zeros(self.num_grid, dtype=np.float64)
        self.V0_grid[1:] = -self.Z / self.grid[1:]
        self.V0_grid[0] = self.V0_grid[1]  # avoid singularity at r=0

    # .......................................................
    def validate_configuration(self):
        shell_labels = ["S", "P", "D", "F", "G", "H", "I", "J", "K", "L"]

        self.lmax = 0

        # check shell labels
        self.ll = []
        self.nrad = []
        self.nprin = []
        self.occ = []
        for shell in self.electrons.configuration:
            # split into principal quantum number and angular momentum
            n_char = shell[0]
            l_char = shell[1].upper()
            occ_str = shell[2]

            if l_char not in shell_labels:
                raise ValueError(f"Invalid shell label {l_char} in configuration {shell}.")

            try:
                n = int(n_char)
            except ValueError as err:
                raise ValueError(
                    f"Invalid principal quantum number {n_char} in configuration {shell}."
                ) from err

            try:
                occ = float(occ_str)
            except ValueError as err:
                raise ValueError(
                    f"Invalid occupation number {occ_str} in configuration {shell}."
                ) from err

            l = shell_labels.index(l_char)
            self.ll.append(l)

            nrad = n - l - 1
            if nrad < 0:
                raise ValueError(f"Invalid shell {shell}: n must be greater than l.")
            self.nrad.append(nrad)
            self.nprin.append(n)

            max_occ = 2 * (2 * l + 1)
            if occ < 0.0 or occ > max_occ:
                raise ValueError(
                    f"Invalid occupation {occ} for shell {shell}. Max occupation is {max_occ}."
                )
            self.occ.append(occ)

        self.ll = np.array(self.ll, dtype=int)
        self.nrad = np.array(self.nrad, dtype=int)
        self.nprin = np.array(self.nprin, dtype=int)
        self.occ = np.array(self.occ, dtype=float)
        self.nshells = len(self.occ)

        self.nmax = np.amax(self.nrad)
        self.lmax = np.amax(self.ll)
        self.num_electrons = np.sum(self.occ)

    # .......................................................
    def initialize_density(self) -> None:
        self.rho_grid = get_slater_density(self.grid, self.Z, self.nprin, self.ll, self.occ)

    # .......................................................
    def get_effective_potential(self, rho_grid: np.ndarray | None = None) -> np.ndarray:
        if rho_grid is None:
            rho_grid = self.rho_grid

        V_Ha = density_potential.hartree_potential(self.basis, rho_grid)

        V_xc = density_potential.exchange_correlation_potential(
            self.basis,
            rho_grid,
            xc_functional=self.dft.xc_functional,
            x_functional=self.dft.x_functional,
            c_functional=self.dft.c_functional,
            alpha_x=self.dft.alpha_x,
            driver=self.dft.driver,
        )

        V_eff = self.V0_grid + V_Ha + V_xc
        return V_eff

    # .......................................................
    def solve_schrodinger(
        self,
        Veff: np.ndarray,
        lmax: int,
        nmax: int,
        Vconf: np.ndarray | None = None,
        lmin: int = 0,
        theory_level: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if theory_level is None:
            theory_flag = "non-relativistic"
        else:
            theory_flag = theory_level.lower()

        if theory_flag == "non-relativistic":
            eps, psi = kohn_sham.solve_schrodinger_local(
                self.basis, Veff, lmax, nmax, Vconf=Vconf, lmin=lmin, solver=self.solver.eigensolver
            )

            return eps, psi

        elif theory_flag == "scalar-relativistic":
            eps, psi = kohn_sham.solve_scalar_relativistic(
                self.basis, Veff, lmax, nmax, Vconf=Vconf, lmin=lmin
            )

        return eps, psi

    # .......................................................
    def get_bound_states(
        self,
        theory_level: str | None = None,
    ) -> tuple[dict[str, list[float]], np.ndarray]:
        V_eff = self.get_effective_potential()
        eps, psi = self.solve_schrodinger(V_eff, self.lmax, self.nmax, theory_level=theory_level)

        eigenvalues = {}
        for l in range(self.lmax + 1):
            (Ie,) = np.where(eps[l, : self.nmax + 1] < 0)
            eps_bound = eps[l, Ie]
            tag = f"{l}"
            eigenvalues[tag] = eps_bound.tolist()

        return eigenvalues, psi

    # .......................................................

    # .......................................................
    def ks_self_consistency(self, theory_level: str | None = None) -> tuple[int, float]:
        """
        Performs Kohn-Sham self-consistency to find the ground state density.
        """
        max_iter = self.dft.max_iter
        tol = self.dft.conv_tol
        alpha_mix = self.dft.alpha_mix

        mixing_scheme = self.dft.mixing_scheme.lower()
        if mixing_scheme.lower() == "diis":
            diis_history = self.dft.diis_history
            diis = DIIS(max_history=diis_history)
        elif mixing_scheme.lower() == "anderson":
            diis_history = self.dft.diis_history
            anderson = AndersonMixing(max_history=diis_history)

        V_eff = self.get_effective_potential()

        # Initial guess for the wavefunctions
        _eps, psi = self.solve_schrodinger(V_eff, self.lmax, self.nmax)

        rho = self.rho_grid.copy()

        err = 1.0e8
        iter_count = 0
        while err > tol and iter_count < max_iter:
            iter_count += 1

            # Compute charge density
            rho_out = density_potential.charge_density(
                self.basis,
                self.nrad,
                self.ll,
                self.occ,
                psi,
            )

            if mixing_scheme.lower() == "diis":
                r = rho_out - rho
                diis.update(rho, r)

                if iter_count > 1:
                    rho = diis.extrapolate(dot_product=lambda a, b: np.dot(a, b), beta=alpha_mix)
                else:
                    rho = rho_out

                err = np.linalg.norm(r)

            elif mixing_scheme.lower() == "anderson":
                r = rho_out - rho
                anderson.update(rho, rho_out, r)

                if iter_count > 1:
                    rho = anderson.extrapolate(
                        dot_product=lambda a, b: np.dot(a, b), beta=alpha_mix
                    )
                else:
                    rho = rho_out

                err = np.linalg.norm(r)

            else:
                # linear mixing of the density
                rho = alpha_mix * rho_out + (1 - alpha_mix) * rho
                err = np.linalg.norm(rho - rho_out)

            # regularize density to be non-negative
            rho[rho < 0.0] = 0.0

            # Update effective potential
            V_eff = self.get_effective_potential(rho_grid=rho)

            # Solve Schrödinger equation with new potential
            _eps, psi = self.solve_schrodinger(
                V_eff,
                self.lmax,
                self.nmax,
                theory_level=theory_level,
            )

        self.rho_grid = rho.copy()

        return iter_count, err

    # .......................................................

    # .......................................................
    def save_density_potential(self):
        """
        Saves the charge density and potential to a file.
        """
        V_eff = self.get_effective_potential()

        if not os.path.exists(self.control.storage_dir):
            os.makedirs(self.control.storage_dir)

        filename = f"{self.element}_density_potential.h5"
        with h5py.File(self.control.storage_dir / filename, "w") as f:
            f.create_dataset("grid", data=self.grid)
            f.create_dataset("rho", data=self.rho_grid)
            f.create_dataset("Veff", data=V_eff)

    # .......................................................
    def read_density_potential(self):
        """
        Reads the charge density and potential from a file.
        """
        storage_dir = self.control.storage_dir
        filename = f"{self.element}_density_potential.h5"
        filepath = os.path.join(storage_dir, filename)

        if not os.path.isfile(filepath):
            return False

        with h5py.File(filepath, "r") as f:
            grid = f["grid"][:]
            rho_grid = f["rho"][:]
            f["Veff"][:]

        if len(grid) != self.num_grid:
            restart_success = False
        else:
            restart_success = np.allclose(grid, self.grid)

        if restart_success:
            self.rho_grid = rho_grid.copy()

        return restart_success
