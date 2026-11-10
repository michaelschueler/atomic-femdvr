"""Functionality shared by the all-electron and pseudo-atom Kohn-Sham solvers."""

import os

import h5py
import numpy as np

import atomic_femdvr.density_potential as density_potential
from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.input import ControlInput, DFTInput
from atomic_femdvr.mixing import make_mixer


#==========================================================================
class AtomDFT:
    """Base class: radial FE-DVR basis, SCF loop and density storage.

    Subclasses implement ``get_effective_potential``, ``solve_schrodinger`` and
    ``occupied_states``.
    """

    control: ControlInput
    dft: DFTInput
    element: str | None
    rho_grid: np.ndarray

    #.......................................................
    def set_basis(self, r_elements: np.ndarray, ng: int) -> None:
        """Set up the FE-DVR basis for the given element boundaries."""
        self.r_elements = r_elements
        ne = len(r_elements) - 1
        self.basis = FEDVR_Basis(ne, ng, r_elements, build_derivatives=True, build_integrals=True)
        self.grid = self.basis.get_gridpoints()
        self.num_grid = len(self.grid)

        # Quadrature weights for the grid-invariant L2 density norm.
        # Bridge points accumulate contributions from both adjacent elements.
        self.quad_w = np.zeros(ne * ng + 1)
        for ie in range(ne):
            h_e = 0.5 * (self.basis.xp[ie + 1] - self.basis.xp[ie])
            self.quad_w[ie * ng : ie * ng + ng + 1] += h_e * self.basis.leg.w_i
    #.......................................................
    def get_effective_potential(self, rho_grid: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError
    #.......................................................
    def solve_schrodinger(self, Veff: np.ndarray, lmax: int, nmax: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    #.......................................................
    def occupied_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (radial quantum numbers, angular momenta, occupations) of the occupied states."""
        raise NotImplementedError
    #.......................................................
    def ks_self_consistency(self) -> tuple[int, float]:
        """Iterate the Kohn-Sham equations to self-consistency.

        Returns the number of iterations and the final density residual.
        """
        nrad, ll, occ = self.occupied_states()
        lmax, nmax = int(np.amax(ll)), int(np.amax(nrad))
        mixer = make_mixer(self.dft.mixing_scheme, self.dft.alpha_mix, self.dft.mixing_history)

        rho = self.rho_grid.copy()
        _, psi = self.solve_schrodinger(self.get_effective_potential(rho), lmax, nmax)

        err = np.inf
        iter_count = 0
        while err > self.dft.conv_tol and iter_count < self.dft.max_iter:
            iter_count += 1
            rho_out = density_potential.charge_density(self.basis, nrad, ll, occ, psi)
            err = np.sqrt(np.dot(self.quad_w, (rho_out - rho)**2))

            rho = mixer.mix(rho, rho_out)
            rho[rho < 0.0] = 0.0

            _, psi = self.solve_schrodinger(self.get_effective_potential(rho), lmax, nmax)

        self.rho_grid = rho
        return iter_count, err
    #.......................................................
    def _density_file(self) -> str:
        return os.path.join(self.control.storage_dir, f'{self.element}_density_potential.h5')
    #.......................................................
    def save_density_potential(self) -> None:
        """Save the density and effective potential (if ``control.store_density`` is set)."""
        if not self.control.store_density:
            return

        os.makedirs(self.control.storage_dir, exist_ok=True)
        with h5py.File(self._density_file(), 'w') as f:
            f.create_dataset('grid', data=self.grid)
            f.create_dataset('rho', data=self.rho_grid)
            f.create_dataset('Veff', data=self.get_effective_potential())
    #.......................................................
    def read_density_potential(self) -> bool:
        """Load a saved density if it exists and was computed on the same grid."""
        filepath = self._density_file()
        if not os.path.isfile(filepath):
            return False

        with h5py.File(filepath, 'r') as f:
            grid = f['grid'][:]
            rho_grid = f['rho'][:]

        if len(grid) != self.num_grid or not np.allclose(grid, self.grid):
            return False

        self.rho_grid = rho_grid.copy()
        return True
