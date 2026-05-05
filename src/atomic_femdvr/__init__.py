"""Solver for the atomic Kohn-Sham equation.

Supports all-electron and pseudo-potential calculations.
"""

from .full_atomic import FullAtomicInput, solve_atomic
from .pseudo_atomic import PseudoAtomicInput, PseudoAtomicResult, solve_pseudo_atomic
from .version import VERSION as __version__  # noqa: N811

__all__ = [
    "FullAtomicInput",
    "PseudoAtomicInput",
    "PseudoAtomicResult",
    "__version__",
    "solve_atomic",
    "solve_pseudo_atomic",
]
