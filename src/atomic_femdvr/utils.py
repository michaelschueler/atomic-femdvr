"""Logging-based progress / output helpers used by the solvers."""

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ==================================================================
def print_time(tic: float, toc: float, msg: str) -> None:
    """Log the elapsed time for a given operation at INFO level.

    Parameters
    ----------
    tic
        Start time, typically from :func:`time.perf_counter`.
    toc
        End time.
    msg
        Short label identifying the operation.
    """
    elapsed = toc - tic
    if elapsed < 1:
        logger.info("Time[%s] : %.2f ms", msg, elapsed * 1000)
    elif elapsed > 300:
        logger.info("Time[%s] : %.2f m", msg, elapsed / 60)
    else:
        logger.info("Time[%s] : %.2f s", msg, elapsed)


# ==================================================================
def get_orbital_label(n: int, l: int) -> str:
    """Return the orbital label for principal quantum number n and angular momentum l."""
    l_labels = ["s", "p", "d", "f", "g", "h", "i", "j"]

    nq = n + l + 1  # Principal quantum number

    if l < len(l_labels):
        return f"{nq}{l_labels[l]}"
    else:
        return f"{nq}l{l}"  # Fallback for higher angular momentum states


# ==================================================================
def print_eigenvalues(lmax: int, eigenvalues: dict, energy_shifts: list | None = None) -> None:
    """Log the eigenvalues for each angular momentum quantum number at INFO level.

    Parameters
    ----------
    lmax
        Maximum angular momentum quantum number to print.
    eigenvalues
        Mapping ``{l_str: [eps_0, eps_1, ...]}`` (Hartree).
    energy_shifts
        Optional per-:math:`\\ell` energy shifts (Hartree); printed alongside
        the corresponding ``l`` block when provided.
    """
    Hr_to_eV = 2.0 * 13.605693009  # Hartree to eV conversion factor

    lines = [40 * "-", "eigenvalues (in eV)".center(40), 40 * "-"]
    for l in range(lmax + 1):
        lines.append(f"l = {l}")
        eps_bound = eigenvalues.get(f"{l}", [])
        n_bound = len(eps_bound)
        if n_bound == 0:
            lines.append("  No bound states found.")
        else:
            for n in range(n_bound):
                orb = get_orbital_label(n, l)
                lines.append(
                    f"  E({orb}) = {eps_bound[n]:.6f} Hr = {Hr_to_eV * eps_bound[n]:.6f} eV"
                )

        if energy_shifts is not None and l < len(energy_shifts):
            lines.append(f"  Energy shift = {Hr_to_eV * energy_shifts[l]:.6f} eV")
    lines.append(40 * "-")
    logger.info("\n".join(lines))


# ==================================================================
def plot_wavefunctions(r_grid: np.ndarray, psi: np.ndarray, lmax: int, eigenvalues: dict) -> None:
    """Plot bound-state wavefunctions for each angular momentum quantum number.

    Parameters
    ----------
    r_grid
        Radial grid, shape ``(nr,)``.
    psi
        Wavefunctions, shape ``(lmax + 1, nmax + 1, nr)``.
    lmax
        Maximum angular momentum quantum number to plot.
    eigenvalues
        Mapping ``{l_str: [eps_0, ...]}`` used to label and count bound states.
    """
    _, ax = plt.subplots(1, lmax + 1, figsize=(4 * (lmax + 1), 6))

    for l in range(lmax + 1):
        ax[l].set_title(rf"$\ell$ = {l}")
        ax[l].set_xlabel("r (a.u.)")
        ax[l].set_ylabel("wave-function")

        eps_bound = eigenvalues.get(f"{l}", [])
        n_bound = len(eps_bound)

        for n in range(n_bound):
            orb = get_orbital_label(n, l)
            ax[l].plot(r_grid, psi[l, n, :], label=orb)

        ax[l].legend()
        ax[l].set_xlim([0, r_grid[-1]])

    plt.tight_layout()
    plt.show()
