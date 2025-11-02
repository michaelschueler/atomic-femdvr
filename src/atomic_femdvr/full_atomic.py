import json
import sys
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import Field

from atomic_femdvr.input import (
    BaseModel,
    DFTInput,
    ElectronsInput,
    SolverInput,
    SysParamsInput
)
from atomic_femdvr.full_atom_dft import FullAtomDFT
from atomic_femdvr.utils import plot_wavefunctions, print_eigenvalues, print_time

class FullAtomicInput(BaseModel):
    sysparams: SysParamsInput
    solver: SolverInput
    dft: DFTInput = Field(default_factory=lambda: DFTInput())
    electrons: ElectronsInput = Field(default_factory=lambda: ElectronsInput())

#==================================================================
def read_input(fname: str):
    """
    Read input parameters from a JSON file.
    """
    with open(fname) as f:
        data = json.load(f)

    electrons = data.get('electrons', {})
    if not electrons:
        raise ValueError("No 'electrons' found in the input file.")

    sysparams = data.get('sysparams', {})
    if not sysparams:
        raise ValueError("No 'sysparams' found in the input file.")
    solver = data.get('solver', {})
    if not solver:
        raise ValueError("No 'solver' parameters found in the input file.")

    dft = data.get('dft', {})


    return sysparams, electrons, solver, dft
#==================================================================


#==================================================================
def solve_atomic(inp: FullAtomicInput, task_list: tuple[str, ...], 
                        plot: bool = False, export_dir: str | None = None) -> dict[str, dict[str, list[float]]]:
    """Solve the all-electrons atomic problem."""
    print(60 * '*')
    print("All-electrons Schrödinger Equation Solver".center(60))
    print(60 * '*')
    tic = perf_counter()

    # Initialize the FullAtomDFT class
    tic = perf_counter()
    atom = FullAtomDFT(inp.sysparams, inp.solver, inp.dft, inp.electrons)
    toc = perf_counter()
    print_time(tic, toc, "Initializing FullAtomDFT")
    print("")

    print(f"number of elements: {len(atom.r_elements) - 1}")
    print(f"number of grid points: {atom.num_grid}\n")

    print(40 * '.')
    print("electronic configuration".center(40))
    print(40 * '.')
    for shell, occ in inp.electrons.elect_config.items():
        print(f"  {shell:>5} : {occ:.2f}")
    print(40 * '.')
    print(f"number of grid points: {atom.num_electrons}\n")

    restart_success = atom.read_density_potential()
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
            num_iter, err = atom.ks_self_consistency()

            if err < inp.dft.conv_tol:
                print(f"Self-consistency converged in {num_iter} iterations with error: {err:.2e}")
            else:
                print(f"Self-consistency did not converge within {inp.dft.max_iter} iterations. Final error: {err:.2e}")
        else:
            print("Skipping self-consistency loop as max_iter is set to 0.")

        toc = perf_counter()
        print_time(tic, toc, "SCF")

        eigenvalues, psi = atom.get_bound_states()
        all_eigenvalues['scf'] = eigenvalues

        print_eigenvalues(atom.lmax, eigenvalues)

        atom.save_density_potential()

        if plot:
            plot_wavefunctions(atom.grid, psi, atom.lmax, eigenvalues)

        scf_done = True

    toc = perf_counter()
    print_time(tic, toc, "Total")
    print(60 * '*')

    return all_eigenvalues
#==================================================================
