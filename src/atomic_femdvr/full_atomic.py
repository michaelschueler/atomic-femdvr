"""High-level all-electron solver entry point.

Defines :class:`FullAtomicInput` and the :func:`solve_atomic` driver
that runs all-electron Kohn-Sham SCF for a single atom.
"""

import logging
from time import perf_counter

from pydantic import Field

from atomic_femdvr.full_atom_dft import FullAtomDFT
from atomic_femdvr.input import (
    BaseModel,
    ControlInput,
    DFTInput,
    ElectronsInput,
    SolverInput,
    SysParamsInput,
)
from atomic_femdvr.utils import plot_wavefunctions, print_eigenvalues, print_time

logger = logging.getLogger(__name__)

__all__ = ["FullAtomicInput", "solve_atomic"]


class FullAtomicInput(BaseModel):
    """Top-level input for :func:`solve_atomic`.

    Bundles all sub-models needed to configure an all-electron run.
    """

    control: ControlInput = Field(
        description="Run-control flags (storage, restart).",
    )
    sysparams: SysParamsInput = Field(
        description="System parameters (element, lmax / nmax).",
    )
    solver: SolverInput = Field(
        description="Discretisation and eigensolver parameters.",
    )
    dft: DFTInput = Field(
        default_factory=DFTInput,
        description="DFT driver, functional, and SCF mixing parameters.",
    )
    electrons: ElectronsInput = Field(
        default_factory=ElectronsInput,
        description="Nuclear charge and electron configuration.",
    )


# ==================================================================
def solve_atomic(
    inp: FullAtomicInput,
    task_list: tuple[str, ...],
    plot: bool = False,
    export_dir: str | None = None,
) -> None:
    """Solve the all-electron Kohn-Sham problem for a single atom.

    Runs the requested tasks in order. The non-relativistic solver runs
    first; if ``inp.solver.theory_level == "scalar-relativistic"`` a
    second SCF pass is run starting from the non-relativistic density.

    Parameters
    ----------
    inp
        Input parameters describing the system, solver, DFT, and
        electron configuration.
    task_list
        Sequence of tasks to run, currently only ``"scf"`` is supported.
        A single comma-separated string is also accepted.
    plot
        If ``True``, plot bound-state wavefunctions after SCF.
    export_dir
        Currently unused; reserved for parity with
        :func:`atomic_femdvr.solve_pseudo_atomic`.
    """
    logger.info(60 * "*")
    logger.info("All-electrons Schrödinger Equation Solver".center(60))
    logger.info(60 * "*")
    tic = perf_counter()

    # Initialize the FullAtomDFT class
    tic = perf_counter()
    atom = FullAtomDFT(inp.control, inp.sysparams, inp.electrons, inp.solver, inp.dft)
    toc = perf_counter()
    print_time(tic, toc, "Initializing FullAtomDFT")
    logger.info("")

    logger.info(f"number of elements: {len(atom.r_elements) - 1}")
    logger.info(f"number of grid points: {atom.num_grid}\n")

    logger.info(40 * ".")
    logger.info("electronic configuration".center(40))
    logger.info(40 * ".")
    for ishell in range(atom.nshells):
        logger.info(
            f"  l = {atom.ll[ishell]}, nr = {atom.nrad[ishell]} : occ = {atom.occ[ishell]:.2f}"
        )
    logger.info(40 * ".")
    logger.info(f"number of electrons: {atom.num_electrons}\n")

    tic = perf_counter()
    restart_success = atom.read_density_potential()
    if restart_success:
        logger.info("Restarting from saved density and potential.\n")
    else:
        logger.info("No saved density and potential found. Starting from scratch.\n")

        atom.initialize_density()

    toc = perf_counter()
    print_time(tic, toc, "Initializing density")

    # split comma-separated tasks
    task_string = task_list[0]
    if "," in task_string:
        task_list = tuple(t.strip() for t in task_string.split(","))

    all_eigenvalues = {}
    if "scf" in task_list:
        logger.info("Starting Kohn-Sham self-consistency: non-relativistic ...\n")

        tic = perf_counter()
        if inp.dft.max_iter > 0:
            num_iter, err = atom.ks_self_consistency(theory_level="non-relativistic")

            if err < inp.dft.conv_tol:
                logger.info(
                    f"Self-consistency converged in {num_iter} iterations with error: {err:.2e}"
                )
            else:
                logger.info(
                    f"Self-consistency did not converge within {inp.dft.max_iter} "
                    f"iterations. Final error: {err:.2e}"
                )
        else:
            logger.info("Skipping self-consistency loop as max_iter is set to 0.")

        toc = perf_counter()
        print_time(tic, toc, "SCF")

        eigenvalues, psi = atom.get_bound_states(theory_level="non-relativistic")
        all_eigenvalues["scf"] = eigenvalues

        print_eigenvalues(atom.lmax, eigenvalues)

        if inp.solver.theory_level.lower() == "scalar-relativistic":
            logger.info("Starting Kohn-Sham self-consistency: scalar-relativistic ...\n")

            tic = perf_counter()
            num_iter, err = atom.ks_self_consistency(theory_level="scalar-relativistic")

            toc = perf_counter()
            print_time(tic, toc, "SCF")

            eigenvalues, psi = atom.get_bound_states(theory_level="scalar-relativistic")
            all_eigenvalues["scf"] = eigenvalues

            print_eigenvalues(atom.lmax, eigenvalues)

        atom.save_density_potential()

        if plot:
            plot_wavefunctions(atom.grid, psi, atom.lmax, eigenvalues)

    toc = perf_counter()
    print_time(tic, toc, "Total")
    logger.info(60 * "*")


# ==================================================================
