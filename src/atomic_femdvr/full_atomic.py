"""Driver for all-electron calculations (``atomic_femdvr atomic``)."""

import warnings
from time import perf_counter

from pydantic import Field

from atomic_femdvr.full_atom_dft import FullAtomDFT
from atomic_femdvr.input import (
    BaseModel,
    ControlInput,
    DFTInput,
    ElectronsInput,
    OutputInput,
    SolverInput,
    SysParamsInput,
)
from atomic_femdvr.projector_output import write_wfc_pyimpurity
from atomic_femdvr.utils import parse_tasks, plot_wavefunctions, print_eigenvalues, print_time

TASKS = ("scf",)


class FullAtomicInput(BaseModel):
    control: ControlInput
    sysparams: SysParamsInput
    solver: SolverInput
    dft: DFTInput = Field(default_factory=lambda: DFTInput())
    electrons: ElectronsInput = Field(default_factory=lambda: ElectronsInput())
    output: OutputInput = Field(default_factory=lambda: OutputInput())

#==================================================================
def _run_scf(atom: FullAtomDFT, label: str) -> None:
    print(f"Starting Kohn-Sham self-consistency: {label} ...\n")
    tic = perf_counter()
    num_iter, err = atom.ks_self_consistency()
    if err < atom.dft.conv_tol:
        print(f"Self-consistency converged in {num_iter} iterations with error: {err:.2e}")
    else:
        print(f"Self-consistency did not converge within {atom.dft.max_iter} iterations. Final error: {err:.2e}")
    print_time(tic, perf_counter(), f"SCF ({label})")
#==================================================================
def solve_atomic(inp: FullAtomicInput, task_list: tuple[str, ...],
                 plot: bool = False, export_dir: str | None = None) -> dict[str, dict[str, list[float]]]:
    """Solve the all-electron atomic problem.

    Returns the eigenvalues (Hartree) per task, keyed by angular momentum.
    """
    tasks = parse_tasks(task_list, TASKS)

    print(60 * '*')
    print("All-electrons Schrödinger Equation Solver".center(60))
    print(60 * '*')
    tic_total = perf_counter()

    theory_level = inp.dft.theory_level
    if theory_level != 'non-relativistic':
        warnings.warn(f"theory_level = '{theory_level}' is an approximate relativistic treatment "
                      "and has not been benchmarked.", stacklevel=2)

    tic = perf_counter()
    atom = FullAtomDFT(inp.control, inp.sysparams, inp.electrons, inp.solver, inp.dft)
    print_time(tic, perf_counter(), "Initializing FullAtomDFT")
    print("")

    print(f"number of elements: {len(atom.r_elements) - 1}")
    print(f"number of grid points: {atom.num_grid}\n")

    print(40 * '.')
    print("electronic configuration".center(40))
    print(40 * '.')
    for ishell in range(atom.nshells):
        print(f"  l = {atom.ll[ishell]}, nr = {atom.nrad[ishell]} : occ = {atom.occ[ishell]:.2f}")
    print(40 * '.')
    print(f"number of electrons: {atom.num_electrons}\n")

    if atom.read_density_potential():
        print("Restarting from saved density and potential.\n")
    else:
        print("No saved density and potential found. Starting from scratch.\n")
        atom.initialize_density()

    all_eigenvalues = {}
    if 'scf' in tasks:
        if inp.dft.max_iter == 0:
            print("Skipping self-consistency loop as max_iter is set to 0.")
        elif theory_level == 'non-relativistic':
            _run_scf(atom, theory_level)
        else:
            # The relativistic Hamiltonians depend on the energy; a non-relativistic
            # SCF first gives the fixed-point iteration a good starting density.
            atom.dft.theory_level = 'non-relativistic'
            _run_scf(atom, 'non-relativistic warm-up')
            atom.dft.theory_level = theory_level
            if atom.dft.relativistic_scf:
                _run_scf(atom, theory_level)
            else:
                print(f"Skipping {theory_level} SCF; using the non-relativistic density.")

        eigenvalues, psi = atom.get_bound_states()
        all_eigenvalues['scf'] = eigenvalues

        print_eigenvalues(atom.lmax, eigenvalues)

        atom.save_density_potential()

        if inp.output.output_wfc_pyimpurity:
            out_dir = export_dir or str(atom.control.storage_dir)
            write_wfc_pyimpurity(atom.basis, psi, eigenvalues, atom.element, out_dir)
            print(f"Wavefunctions written to {out_dir}/{atom.element}_wfc_ae.h5")

        if plot:
            plot_wavefunctions(atom.grid, psi, atom.lmax, eigenvalues)

    print_time(tic_total, perf_counter(), "Total")
    print(60 * '*')

    return all_eigenvalues
