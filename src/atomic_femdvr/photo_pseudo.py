"""
Orchestration for pseudo-atomic photoemission calculations.

Loads the self-consistent KS potential from a previous 'pseudoatomic scf' run,
re-solves for bound states, then computes photoemission matrix elements and
scattering phase shifts over an energy grid using the non-local PP scattering solver.
"""

from pathlib import Path
from time import perf_counter

import numpy as np

from atomic_femdvr.pseudo_atom_dft import PseudoAtomDFT
from atomic_femdvr.pseudo_atomic import PseudoAtomicInput
from atomic_femdvr.input import PhotoemissionInput
from atomic_femdvr.photoemission import compute_photoemission, save_photoemission
from atomic_femdvr.utils import print_time


def solve_photoemission_pseudo(inp: PseudoAtomicInput, photo_inp: PhotoemissionInput) -> None:
    """
    Compute photoemission matrix elements from a completed pseudo-atomic SCF run.

    Reads the KS effective potential saved by 'pseudoatomic -t scf', re-solves the
    Schrödinger equation to get bound-state wavefunctions, then for each occupied
    shell and each kinetic energy solves the scattering equation at l_f = l_i ± 1
    using the non-local KB pseudopotential projectors.
    """
    print(60 * '*')
    print("Photoemission Matrix Elements (Pseudo-atomic)".center(60))
    print(60 * '*')

    tic = perf_counter()
    atom = PseudoAtomDFT(inp.control, inp.sysparams, inp.solver, inp.dft)
    toc = perf_counter()
    print_time(tic, toc, "Initializing PseudoAtomDFT")

    tic = perf_counter()
    atom.read_upf(read_density=True, read_potential=True)
    toc = perf_counter()
    print_time(tic, toc, "Reading UPF file")
    print(f"element: {atom.element},  Z_val = {atom.Zval:.0f}")
    print(f"grid points: {atom.num_grid},  lmax = {atom.lmax_pseudo},  nmax = {atom.nmax_pseudo}\n")

    tic = perf_counter()
    ok = atom.read_density_potential()
    toc = perf_counter()
    if not ok:
        raise RuntimeError(
            "No saved density/potential found for this grid. "
            "Run 'pseudoatomic -t scf' before photoemission."
        )
    print("Loaded KS density and potential.\n")
    print_time(tic, toc, "Loading density/potential")

    tic = perf_counter()
    V_eff = atom.get_effective_potential()
    eigenvalues, psi_bound = atom.get_bound_states()
    eps, _ = atom.solve_schrodinger(V_eff, atom.lmax_pseudo, atom.nmax_pseudo)
    toc = perf_counter()
    print_time(tic, toc, "Solving bound states")

    print("\nBound-state eigenvalues (Ha):")
    for l in range(atom.lmax_pseudo + 1):
        for n_i, e in enumerate(eigenvalues.get(str(l), [])):
            print(f"  l={l}, n_rad={n_i}: {e:.6f} Ha")

    # Shell info from UPF
    ll   = atom.upf.lchi
    occ  = atom.upf.oc
    nrad = atom.upf.nnodes_chi

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
        nrad=nrad,
        ll=ll,
        occ=occ,
        energies=energies,
        gauges=photo_inp.gauges,
        store_wavefunctions=photo_inp.store_wavefunctions,
        output_prefix=output_prefix,
        lll=atom.upf.lll,
        Dion=atom.upf.dion,
        beta_pp=atom.beta_grid,
        theory_level=inp.dft.theory_level,
    )
    toc = perf_counter()
    print_time(tic, toc, "Computing photoemission matrix elements")

    out_file = output_dir / (photo_inp.output_prefix + '.h5')
    save_photoemission(results, str(out_file), photo_inp.gauges)

    print(60 * '*')
