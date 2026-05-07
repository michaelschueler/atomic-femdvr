"""High-level pseudo-atomic solver entry point.

Defines :class:`PseudoAtomicInput`, :class:`PseudoAtomicResult`, and the
:func:`solve_pseudo_atomic` driver that runs SCF, soft-Coulomb optimisation,
and non-SCF tasks in sequence.
"""

import logging
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
from atomic_femdvr.utils import plot_wavefunctions, print_eigenvalues, print_time

logger = logging.getLogger(__name__)

__all__ = [
    "ConvergenceError",
    "MissingSCFError",
    "PseudoAtomicInput",
    "PseudoAtomicResult",
    "solve_pseudo_atomic",
]

if TYPE_CHECKING:
    # Stub for type checking
    PseudoAtomicSolverInput = SolverInput
else:
    # Runtime type
    PseudoAtomicSolverInput = solver_input_factory(default_hmin=0.5, default_hmax=4.0)


class ConvergenceError(Exception):
    """Raised when the SCF loop fails to converge within ``max_iter`` iterations."""


class MissingSCFError(Exception):
    """Raised when an ``"optimize"`` or ``"nscf"`` task is requested without prior SCF."""


class PseudoAtomicInput(BaseModel):
    """Top-level input for :func:`solve_pseudo_atomic`.

    Bundles all sub-models needed to configure a pseudo-atomic run.
    """

    control: ControlInput = Field(
        default_factory=ControlInput,
        description="Run-control flags (storage, restart).",
    )
    sysparams: SysParamsInput = Field(
        default_factory=SysParamsInput,
        description="System parameters (element, UPF file, lmax / nmax).",
    )
    solver: PseudoAtomicSolverInput = Field(
        default_factory=PseudoAtomicSolverInput,
        description="Discretisation and eigensolver parameters.",
    )
    dft: DFTInput = Field(
        default_factory=DFTInput,
        description="DFT driver, functional, and SCF mixing parameters.",
    )
    confinement: ConfinementInput = Field(
        default_factory=ConfinementInput,
        description="Confining potential applied to virtual orbitals during the nscf task.",
    )
    output: OutputInput = Field(
        default_factory=OutputInput,
        description="Output toggles (wavefunction format, dipole moments, ...).",
    )


class PseudoAtomicResult(BaseModel):
    """Output of :func:`solve_pseudo_atomic`."""

    eigenvalues: dict[str, dict[str, list[float]]] = Field(
        description=(
            "Nested mapping ``{task_name: {l_str: [eps_0, eps_1, ...]}}``. "
            "Outer keys are tasks that were actually run (``'scf'`` and / or "
            "``'nscf'``); inner keys are angular-momentum quantum numbers as strings."
        ),
    )
    energy_shifts: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "Per-:math:`\\ell` energy shifts (``'nscf'`` minus ``'scf'`` for each "
            "bound state). ``None`` when ``'nscf'`` was not requested."
        ),
    )


# ==================================================================
def solve_pseudo_atomic(
    inp: PseudoAtomicInput,
    task_list: tuple[str, ...],
    plot: bool = False,
    export_dir: str | None = None,
) -> PseudoAtomicResult:
    """Solve the pseudo-atomic Kohn-Sham problem.

    Runs the requested tasks in order and returns the collected eigenvalues
    and energy shifts.

    Parameters
    ----------
    inp
        Input parameters describing the system, solver, DFT, and confinement.
    task_list
        Sequence of tasks to run, drawn from ``"scf"``, ``"optimize"``,
        ``"nscf"``. A single comma-separated string (e.g. ``("scf,nscf",)``)
        is also accepted.
    plot
        If ``True``, plot bound-state wavefunctions after SCF and after
        non-SCF (requires a display).
    export_dir
        Directory to write projector / eigenvalue / dipole output files
        to. Requires ``"nscf"`` in ``task_list``.

    Returns
    -------
    PseudoAtomicResult
        Eigenvalues per task and per-:math:`\\ell` energy shifts (when
        ``"nscf"`` ran).

    Raises
    ------
    ConvergenceError
        If SCF fails to converge within ``inp.dft.max_iter`` iterations.
    MissingSCFError
        If ``"optimize"`` or ``"nscf"`` is requested without prior SCF or
        a valid restart file.
    ValueError
        If ``export_dir`` is set but ``"nscf"`` did not run.
    """
    logger.info(60 * "*")
    logger.info("Pseudo-atomic Schrödinger Equation Solver".center(60))
    logger.info(60 * "*")
    tic = perf_counter()

    # Initialize the PseudoAtomDFT class
    tic = perf_counter()
    pseudo_atom = PseudoAtomDFT(inp.control, inp.sysparams, inp.solver, inp.dft)
    toc = perf_counter()
    print_time(tic, toc, "Initializing PseudoAtomDFT")
    logger.info("")

    logger.info(f"number of elements: {len(pseudo_atom.r_elements) - 1}")
    logger.info(f"number of grid points: {pseudo_atom.num_grid}\n")

    # Read UPF file
    tic = perf_counter()
    pseudo_atom.read_upf(read_density=True, read_potential=True)
    toc = perf_counter()
    print_time(tic, toc, "Reading UPF file")
    logger.info("")

    restart_success = pseudo_atom.read_density_potential()
    if restart_success:
        logger.info("Restarting from saved density and potential.\n")
    else:
        logger.info("No saved density and potential found. Starting from scratch.\n")

    scf_done = False
    nscf_done = False

    # split comma-separated tasks
    task_string = task_list[0]
    if "," in task_string:
        task_list = tuple(t.strip() for t in task_string.split(","))

    all_eigenvalues = {}
    energy_shifts = None
    if "scf" in task_list:
        tic = perf_counter()
        if inp.dft.max_iter > 0:
            num_iter, err = pseudo_atom.ks_self_consistency(
                max_iter=inp.dft.max_iter, tol=inp.dft.conv_tol, alpha_mix=inp.dft.alpha_mix
            )

            if err < inp.dft.conv_tol:
                logger.info(
                    f"Self-consistency converged in {num_iter} iterations with error: {err:.2e}"
                )
            else:
                raise ConvergenceError(
                    f"Self-consistency did not converge within {inp.dft.max_iter} "
                    f"iterations. Final error: {err:.2e}"
                )
        else:
            logger.info("Skipping self-consistency loop as max_iter is set to 0.")

        toc = perf_counter()
        print_time(tic, toc, "SCF")

        eigenvalues, psi = pseudo_atom.get_bound_states()
        all_eigenvalues["scf"] = eigenvalues

        if pseudo_atom.upf is None:
            raise RuntimeError("UPF file was not loaded; SCF cannot proceed.")

        print_eigenvalues(int(pseudo_atom.upf.lmax), eigenvalues)

        pseudo_atom.save_density_potential()

        if plot:
            lmax = int(pseudo_atom.upf.lmax)
            plot_wavefunctions(pseudo_atom.grid, psi, lmax, eigenvalues)

        scf_done = True

    if "optimize" in task_list:
        if not scf_done and not restart_success:
            raise MissingSCFError(
                "Optimize task requires SCF to be completed first or a valid restart file."
            )

        tic = perf_counter()
        Q_opt = pseudo_atom.optimize_soft_coul(inp.confinement)
        toc = perf_counter()
        print_time(tic, toc, "Optimizing Soft Coulomb Confinement")
        logger.info("")
        logger.info(f"Optimized soft Coulomb confinement parameter Q: {Q_opt:.4f}\n")
        inp.confinement.softcoul_charge = Q_opt

    if "nscf" in task_list:
        if not scf_done and not restart_success:
            raise MissingSCFError(
                "Non-SCF task requires SCF to be completed first or a valid restart file."
            )

        tic = perf_counter()
        energy_shifts, eigenvalues, psi = pseudo_atom.get_all_states_energy_shifts(
            inp.sysparams.lmax, inp.sysparams.nmax, confinement=inp.confinement
        )

        toc = perf_counter()
        print_time(tic, toc, "Non-SCF Calculation")
        logger.info("")

        outermost_shifts = [
            energy_shifts[f"{l}"][-1]
            for l in range(inp.sysparams.lmax + 1)
            if f"{l}" in energy_shifts
        ]
        print_eigenvalues(inp.sysparams.lmax, eigenvalues, energy_shifts=outermost_shifts)
        all_eigenvalues["nscf"] = eigenvalues

        if plot:
            plot_wavefunctions(pseudo_atom.grid, psi, inp.sysparams.lmax, eigenvalues)

        nscf_done = True

    if export_dir is not None:
        if not nscf_done:
            raise ValueError(
                "Exporting wave-functions requires non-SCF task to be completed first."
            )

        tic = perf_counter()
        pseudo_atom.export_projectors(
            inp.sysparams.lmax,
            inp.sysparams.nmax,
            psi,
            inp.confinement,
            inp.output,
            out_dir=export_dir,
        )
        toc = perf_counter()
        print_time(tic, toc, "Exporting Wave-Functions")

        pseudo_atom.export_eigenvalues(eigenvalues, out_dir=export_dir)

        if inp.output.output_dipole_moments:
            tic = perf_counter()
            pseudo_atom.export_dipole_moments(
                inp.sysparams.lmax, inp.sysparams.nmax, psi, inp.output, out_dir=export_dir
            )
            toc = perf_counter()
            print_time(tic, toc, "Exporting Dipole Moments")

    toc = perf_counter()
    print_time(tic, toc, "Total")
    logger.info(60 * "*")

    return PseudoAtomicResult(eigenvalues=all_eigenvalues, energy_shifts=energy_shifts)
