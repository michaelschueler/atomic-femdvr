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
from atomic_femdvr.PseudoAtomDFT import PseudoAtomDFT
from atomic_femdvr.utils import PlotWavefunctions, PrintEigenvalues, PrintTime

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
def ReadInput(fname: str):
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
def solve_pseudo_atomic(inp: PseudoAtomicInput, task_list: tuple[str, ...], plot: bool, export_dir: str | None) -> None:
    """Solve the pseudo-atomic problem."""
    print(60 * '*')
    print("Pseudo-atomic Schrödinger Equation Solver".center(60))
    print(60 * '*')
    tic = perf_counter()

    # Initialize the PseudoAtomDFT class
    tic = perf_counter()
    pseudo_atom = PseudoAtomDFT(inp.pseudo_config, inp.sysparams, inp.solver, inp.dft)
    toc = perf_counter()
    PrintTime(tic, toc, "Initializing PseudoAtomDFT")
    print("")

    print(f"number of elements: {len(pseudo_atom.r_elements) - 1}")
    print(f"number of grid points: {pseudo_atom.num_grid}\n")

    # Read UPF file
    tic = perf_counter()
    pseudo_atom.ReadUPF(read_density=True, read_potential=True)
    toc = perf_counter()
    PrintTime(tic, toc, "Reading UPF file")
    print("")

    restart_success = pseudo_atom.ReadDensityPotential()
    if restart_success:
        print("Restarting from saved density and potential.\n")
    else:
        print("No saved density and potential found. Starting from scratch.\n")


    scf_done = False
    nscf_done = False

    if 'scf' in task_list:

        tic = perf_counter()
        if inp.dft.max_iter > 0:
            num_iter, err = pseudo_atom.KS_SelfConsistency(max_iter=inp.dft.max_iter, tol=inp.dft.conv_tol, alpha=inp.dft.alpha)

            if err < inp.dft.conv_tol:
                print(f"Self-consistency converged in {num_iter} iterations with error: {err:.2e}")
            else:
                print(f"Self-consistency did not converge within {inp.dft.max_iter} iterations. Final error: {err:.2e}")
        else:
            print("Skipping self-consistency loop as max_iter is set to 0.")

        toc = perf_counter()
        PrintTime(tic, toc, "SCF")

        eigenvalues, psi = pseudo_atom.GetBoundStates()

        assert pseudo_atom.upf is not None

        PrintEigenvalues(int(pseudo_atom.upf.lmax), eigenvalues)

        pseudo_atom.SaveDensityPotential()

        if plot:
            PlotWavefunctions(pseudo_atom.grid, psi, int(pseudo_atom.upf.lmax), eigenvalues)

        scf_done = True

    if 'optimize' in task_list:
        if not scf_done and not restart_success:
            print("Error: Non-SCF task requires SCF to be completed first or a valid restart file.")
            sys.exit(2)

        tic = perf_counter()
        Q_opt = pseudo_atom.OptimizeSoftCoul(inp.confinement)
        toc = perf_counter()
        PrintTime(tic, toc, "Optimizing Soft Coulomb Confinement")
        print("")
        print(f"Optimized soft Coulomb confinement parameter Q: {Q_opt:.4f}\n")
        inp.confinement.softcoul_charge = Q_opt

    if 'nscf' in task_list:
        if not scf_done and not restart_success:
            print("Error: Non-SCF task requires SCF to be completed first or a valid restart file.")
            sys.exit(2)

        tic = perf_counter()
        energy_shifts, eigenvalues, psi = pseudo_atom.GetStatesEnergyShift(inp.sysparams.lmax, inp.sysparams.nmax, confinement=inp.confinement)
        toc = perf_counter()
        PrintTime(tic, toc, "Non-SCF Calculation")
        print("")

        PrintEigenvalues(inp.sysparams.lmax, eigenvalues, energy_shifts=energy_shifts)

        if plot:
            PlotWavefunctions(pseudo_atom.grid, psi, inp.sysparams.lmax, eigenvalues)

        nscf_done = True

    if export_dir is not None:
        if not nscf_done:
            print("Error: Exporting projectors requires non-SCF task to be completed first.")
            sys.exit(2)

        tic = perf_counter()
        pseudo_atom.ExportProjector(inp.sysparams.lmax, inp.sysparams.nmax, psi, inp.confinement, export_dir, nr=inp.projector.nr, rmin=inp.projector.rmin)
        toc = perf_counter()
        PrintTime(tic, toc, "Exporting Projectors")

    toc = perf_counter()
    PrintTime(tic, toc, "Total")
    print(60 * '*')
