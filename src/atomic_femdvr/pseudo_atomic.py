import json
import sys
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import Field

from atomic_femdvr.input import (
    BaseModel,
    ConfinementInput,
    DFTInput,
    ProjectorInput,
    PseudoConfigInput,
    SolverInput,
    SysParamsInput,
    solver_input_factory,
)
from atomic_femdvr.pseudo_atom_dft import PseudoAtomDFT
from atomic_femdvr.utils import plot_wavefunctions, print_eigenvalues, print_time

if TYPE_CHECKING:
    # Stub for type checking
    PseudoAtomicSolverInput = SolverInput
else:
    # Runtime type
    PseudoAtomicSolverInput = solver_input_factory(default_hmin=0.5, default_hmax=4.0)

class PseudoAtomicInput(BaseModel):
    sysparams: SysParamsInput
    solver: PseudoAtomicSolverInput
    pseudo_config: PseudoConfigInput = Field(default_factory=lambda: PseudoConfigInput())
    dft: DFTInput = Field(default_factory=lambda: DFTInput())
    confinement: ConfinementInput = Field(default_factory=lambda: ConfinementInput())
    projector: ProjectorInput = Field(default_factory=lambda: ProjectorInput())

#==================================================================
def read_input(fname: str):
    """
    Read input parameters from a JSON file.
    """
    with open(fname) as f:
        data = json.load(f)

    pseudo_config = data.get('pseudo_config', {})
    if not pseudo_config:
        raise ValueError("No 'pseudo_config' found in the input file.")

    sysparams = data.get('sysparams', {})
    if not sysparams:
        raise ValueError("No 'sysparams' found in the input file.")
    solver = data.get('solver', {})
    if not solver:
        raise ValueError("No 'solver' parameters found in the input file.")

    dft = data.get('dft', {})
    confinement = data.get('confinement', {})
    proj = data.get('projector', {})

    return pseudo_config, sysparams, solver, dft, confinement, proj
#==================================================================


#==================================================================
def solve_pseudo_atomic(inp: PseudoAtomicInput, task_list: tuple[str, ...], 
                        plot: bool = False, export_dir: str | None = None) -> dict[str, dict[str, list[float]]]:
    """Solve the pseudo-atomic problem."""
    print(60 * '*')
    print("Pseudo-atomic Schrödinger Equation Solver".center(60))
    print(60 * '*')
    tic = perf_counter()

    # Initialize the PseudoAtomDFT class
    tic = perf_counter()
    pseudo_atom = PseudoAtomDFT(inp.pseudo_config, inp.sysparams, inp.solver, inp.dft)
    toc = perf_counter()
    print_time(tic, toc, "Initializing PseudoAtomDFT")
    print("")

    print(f"number of elements: {len(pseudo_atom.r_elements) - 1}")
    print(f"number of grid points: {pseudo_atom.num_grid}\n")

    # Read UPF file
    tic = perf_counter()
    pseudo_atom.read_upf(read_density=True, read_potential=True)
    toc = perf_counter()
    print_time(tic, toc, "Reading UPF file")
    print("")

    restart_success = pseudo_atom.read_density_potential()
    if restart_success:
        print("Restarting from saved density and potential.\n")
    else:
        print("No saved density and potential found. Starting from scratch.\n")


    scf_done = False
    nscf_done = False

    all_eigenvalues = {}
    if 'scf' in task_list:

        tic = perf_counter()
        if inp.dft.max_iter > 0:
            num_iter, err = pseudo_atom.ks_self_consistency(max_iter=inp.dft.max_iter, tol=inp.dft.conv_tol,
                                                             alpha_mix=inp.dft.alpha_mix)

            if err < inp.dft.conv_tol:
                print(f"Self-consistency converged in {num_iter} iterations with error: {err:.2e}")
            else:
                print(f"Self-consistency did not converge within {inp.dft.max_iter} iterations. Final error: {err:.2e}")
        else:
            print("Skipping self-consistency loop as max_iter is set to 0.")

        toc = perf_counter()
        print_time(tic, toc, "SCF")

        eigenvalues, psi = pseudo_atom.get_bound_states()
        all_eigenvalues['scf'] = eigenvalues

        assert pseudo_atom.upf is not None

        print_eigenvalues(int(pseudo_atom.upf.lmax), eigenvalues)

        pseudo_atom.save_density_potential()

        if plot:
            plot_wavefunctions(pseudo_atom.grid, psi, int(pseudo_atom.upf.lmax), eigenvalues)

        scf_done = True

    if 'optimize' in task_list:
        if not scf_done and not restart_success:
            print("Error: Non-SCF task requires SCF to be completed first or a valid restart file.")
            sys.exit(2)

        tic = perf_counter()
        Q_opt = pseudo_atom.optimize_soft_coul(inp.confinement)
        toc = perf_counter()
        print_time(tic, toc, "Optimizing Soft Coulomb Confinement")
        print("")
        print(f"Optimized soft Coulomb confinement parameter Q: {Q_opt:.4f}\n")
        inp.confinement.softcoul_charge = Q_opt

    if 'nscf' in task_list:
        if not scf_done and not restart_success:
            print("Error: Non-SCF task requires SCF to be completed first or a valid restart file.")
            sys.exit(2)

        tic = perf_counter()
        energy_shifts, eigenvalues, psi = pseudo_atom.get_states_energy_shift(
            inp.sysparams.lmax, 
            inp.sysparams.nmax, 
            confinement=inp.confinement)
        
        toc = perf_counter()
        print_time(tic, toc, "Non-SCF Calculation")
        print("")

        print_eigenvalues(inp.sysparams.lmax, eigenvalues, energy_shifts=energy_shifts)
        all_eigenvalues['nscf'] = eigenvalues

        if plot:
            plot_wavefunctions(pseudo_atom.grid, psi, inp.sysparams.lmax, eigenvalues)

        nscf_done = True

    if export_dir is not None:
        if not nscf_done:
            print("Error: Exporting projectors requires non-SCF task to be completed first.")
            sys.exit(2)

        tic = perf_counter()
        pseudo_atom.export_projector(inp.sysparams.lmax, inp.sysparams.nmax, psi, inp.confinement, 
                                    export_dir, nr=inp.projector.nr, rmin=inp.projector.rmin)
        toc = perf_counter()
        print_time(tic, toc, "Exporting Projectors")

    toc = perf_counter()
    print_time(tic, toc, "Total")
    print(60 * '*')

    return all_eigenvalues
