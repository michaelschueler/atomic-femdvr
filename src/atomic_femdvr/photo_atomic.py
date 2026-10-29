"""
Orchestration for all-electron photoemission calculations.

Loads the self-consistent KS potential from a previous 'atomic scf' run,
re-solves for bound states, then computes photoemission matrix elements
and scattering phase shifts over an energy grid.
"""

from pathlib import Path
from time import perf_counter

import numpy as np

from atomic_femdvr.full_atom_dft import FullAtomDFT
from atomic_femdvr.full_atomic import FullAtomicInput
from atomic_femdvr.input import PhotoemissionInput
from atomic_femdvr.photoemission import compute_photoemission, save_photoemission
from atomic_femdvr.utils import print_time


def solve_photoemission_atomic(inp: FullAtomicInput, photo_inp: PhotoemissionInput) -> None:
    """
    Compute photoemission matrix elements from a completed all-electron SCF run.

    Reads the KS effective potential saved by 'atomic -t scf', re-solves the
    Schrödinger equation to get bound-state wavefunctions, then for each occupied
    shell and each kinetic energy solves the scattering equation at l_f = l_i ± 1
    and computes the requested radial matrix elements.
    """
    print(60 * '*')
    print("Photoemission Matrix Elements (All-electron)".center(60))
    print(60 * '*')

    tic = perf_counter()
    atom = FullAtomDFT(inp.control, inp.sysparams, inp.electrons, inp.solver, inp.dft)
    toc = perf_counter()
    print_time(tic, toc, "Initializing FullAtomDFT")
    print(f"element: {atom.element},  Z = {atom.Z:.0f}")
    print(f"grid points: {atom.num_grid},  lmax = {atom.lmax},  nmax = {atom.nmax}\n")

    tic = perf_counter()
    ok = atom.read_density_potential()
    toc = perf_counter()
    if not ok:
        raise RuntimeError(
            "No saved density/potential found for this grid. "
            "Run 'atomic -t scf' before photoemission."
        )
    print("Loaded KS density and potential.\n")
    print_time(tic, toc, "Loading density/potential")

    tic = perf_counter()
    V_eff = atom.get_effective_potential()
    eps, psi_bound = atom.solve_schrodinger(V_eff, atom.lmax, atom.nmax)
    toc = perf_counter()
    print_time(tic, toc, "Solving bound states")

    print("\nBound-state eigenvalues (Ha):")
    for ishell in range(atom.nshells):
        l_i = atom.ll[ishell]
        n_i = atom.nrad[ishell]
        e = eps[l_i, n_i]
        if e < 0.0:
            print(f"  l={l_i}, n_rad={n_i}: {e:.6f} Ha  (occ={atom.occ[ishell]:.2f})")

    energies = np.linspace(photo_inp.energy_min, photo_inp.energy_max,
                           photo_inp.n_energies)

    print(f"\nEnergy range: {photo_inp.energy_min:.3f} – {photo_inp.energy_max:.3f} Ha"
          f"  ({photo_inp.n_energies} points)")
    print(f"Gauges: {photo_inp.gauges}")
    print(f"Store wavefunctions: {photo_inp.store_wavefunctions}\n")

    output_dir = inp.control.storage_dir
    output_prefix = str(output_dir / photo_inp.output_prefix)

    tic = perf_counter()
    results = compute_photoemission(
        basis=atom.basis,
        Veff_grid=V_eff,
        eps=eps,
        psi_bound=psi_bound,
        nrad=atom.nrad,
        ll=atom.ll,
        occ=atom.occ,
        energies=energies,
        gauges=photo_inp.gauges,
        store_wavefunctions=photo_inp.store_wavefunctions,
        output_prefix=output_prefix,
        theory_level=inp.dft.theory_level,
        Z=atom.Z,
        nuclear_sigma=inp.solver.nuclear_sigma,
        lmax_scatter=photo_inp.lmax_scatter,
    )
    toc = perf_counter()
    print_time(tic, toc, "Computing photoemission matrix elements")

    out_file = output_dir / (photo_inp.output_prefix + '.h5')
    save_photoemission(results, str(out_file), photo_inp.gauges)

    print(60 * '*')
