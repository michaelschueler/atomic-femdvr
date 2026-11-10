"""Driver for pseudo-atom calculations (``atomic_femdvr pseudoatomic``)."""

import os
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import Field

from atomic_femdvr.input import (
    BaseModel,
    ConfinementInput,
    ControlInput,
    DFTInput,
    OutputInput,
    SolverInput,
    SysParamsInput,
    solver_input_factory,
)
from atomic_femdvr.pseudo_atom_dft import PseudoAtomDFT
from atomic_femdvr.utils import parse_tasks, plot_wavefunctions, print_eigenvalues, print_time

if TYPE_CHECKING:
    # Stub for type checking
    PseudoAtomicSolverInput = SolverInput
else:
    # Runtime type
    PseudoAtomicSolverInput = solver_input_factory(default_hmin=0.5, default_hmax=4.0)

TASKS = ("scf", "optimize", "nscf")


class ConvergenceError(Exception):
    """Custom exception for convergence errors in the SCF process."""

class MissingSCFError(Exception):
    """Custom exception for missing SCF when required."""

class PseudoAtomicInput(BaseModel):
    control: ControlInput = Field(default_factory=lambda: ControlInput())
    sysparams: SysParamsInput = Field(default_factory=lambda: SysParamsInput())
    solver: PseudoAtomicSolverInput = Field(default_factory=lambda: PseudoAtomicSolverInput())
    dft: DFTInput = Field(default_factory=lambda: DFTInput())
    confinement: ConfinementInput = Field(default_factory=lambda: ConfinementInput())
    output: OutputInput = Field(default_factory=lambda: OutputInput())

#==================================================================
def solve_pseudo_atomic(inp: PseudoAtomicInput, task_list: tuple[str, ...],
                        plot: bool = False, export_dir: str | None = None) -> dict[str, dict[str, list[float]]]:
    """Solve the pseudo-atomic problem.

    task_list contains any of 'scf', 'optimize', 'nscf' (comma-separated entries are split).
    Returns the eigenvalues (Hartree) per task, keyed by angular momentum.
    """
    tasks = parse_tasks(task_list, TASKS)

    print(60 * '*')
    print("Pseudo-atomic Schrödinger Equation Solver".center(60))
    print(60 * '*')
    tic_total = perf_counter()

    tic = perf_counter()
    pseudo_atom = PseudoAtomDFT(inp.control, inp.sysparams, inp.solver, inp.dft)
    pseudo_atom.read_upf(read_density=True, read_potential=True)
    toc = perf_counter()
    print_time(tic, toc, "Reading UPF file and setting up the grid")
    print("")

    print(f"number of elements: {len(pseudo_atom.r_elements) - 1}")
    print(f"number of grid points: {pseudo_atom.num_grid}\n")

    if not pseudo_atom.upf.has_paos:
        # Without orbitals the occupations are unknown: no SCF, the density is taken as is.
        if 'scf' in tasks:
            raise ValueError("The UPF file contains no pseudo-atomic orbitals (PP_PSWFC), so it "
                             "cannot be solved self-consistently. Use the 'nscf' task, which "
                             "takes the density from PP_RHOATOM.")
        print("No pseudo-atomic orbitals in the UPF file: using the density from PP_RHOATOM.\n")
        restart_success = True
    else:
        restart_success = pseudo_atom.read_density_potential()
        if restart_success:
            print("Restarting from saved density and potential.\n")
        else:
            print("No saved density and potential found. Starting from scratch.\n")

    n_lowest = pseudo_atom.principal_quantum_numbers()
    all_eigenvalues = {}
    if 'scf' in tasks:

        tic = perf_counter()
        if inp.dft.max_iter > 0:
            num_iter, err = pseudo_atom.ks_self_consistency()

            if err < inp.dft.conv_tol:
                print(f"Self-consistency converged in {num_iter} iterations with error: {err:.2e}")
            else:
                raise ConvergenceError(f"Self-consistency did not converge within {inp.dft.max_iter} iterations. Final error: {err:.2e}")
        else:
            print("Skipping self-consistency loop as max_iter is set to 0.")

        toc = perf_counter()
        print_time(tic, toc, "SCF")

        eigenvalues, psi = pseudo_atom.get_bound_states()
        all_eigenvalues['scf'] = eigenvalues

        lmax = int(pseudo_atom.upf.lmax)
        print_eigenvalues(lmax, eigenvalues, n_lowest=n_lowest)

        pseudo_atom.save_density_potential()

        if plot:
            plot_wavefunctions(pseudo_atom.grid, psi, lmax, eigenvalues, n_lowest=n_lowest)

        restart_success = True

    if 'optimize' in tasks:
        if not restart_success:
            raise MissingSCFError("Optimize task requires SCF to be completed first or a valid restart file.")

        tic = perf_counter()
        Q_opt = pseudo_atom.optimize_soft_coul(inp.confinement)
        toc = perf_counter()
        print_time(tic, toc, "Optimizing Soft Coulomb Confinement")
        print("")
        print(f"Optimized soft Coulomb confinement parameter Q: {Q_opt:.4f}\n")
        inp.confinement.softcoul_charge = Q_opt

    if 'nscf' in tasks:
        if not restart_success:
            raise MissingSCFError("Non-SCF task requires SCF to be completed first or a valid restart file.")

        tic = perf_counter()
        energy_shifts, eigenvalues, psi = pseudo_atom.get_states_energy_shift(
            inp.sysparams.lmax,
            inp.sysparams.nmax,
            confinement=inp.confinement)

        toc = perf_counter()
        print_time(tic, toc, "Non-SCF Calculation")
        print("")

        print_eigenvalues(inp.sysparams.lmax, eigenvalues, energy_shifts=energy_shifts,
                          n_lowest=n_lowest)
        all_eigenvalues['nscf'] = eigenvalues

        if plot:
            plot_wavefunctions(pseudo_atom.grid, psi, inp.sysparams.lmax, eigenvalues,
                               n_lowest=n_lowest)

    if export_dir is not None:
        if 'nscf' not in tasks:
            raise ValueError("Exporting wave-functions requires the 'nscf' task.")

        os.makedirs(export_dir, exist_ok=True)
        tic = perf_counter()
        pseudo_atom.export_projectors(inp.sysparams.lmax, inp.sysparams.nmax, psi, inp.confinement,
                                          inp.output, out_dir=export_dir)
        toc = perf_counter()
        print_time(tic, toc, "Exporting Wave-Functions")

        pseudo_atom.export_eigenvalues(
            eigenvalues,
            out_dir=export_dir,
            lmax=inp.sysparams.lmax,
            nmax=inp.sysparams.nmax,
        )

        if inp.output.output_dipole_moments:
            tic = perf_counter()
            pseudo_atom.export_dipole_moments(inp.sysparams.lmax, inp.sysparams.nmax, psi,
                                             inp.output, out_dir=export_dir)
            toc = perf_counter()
            print_time(tic, toc, "Exporting Dipole Moments")

    print_time(tic_total, perf_counter(), "Total")
    print(60 * '*')

    return all_eigenvalues
