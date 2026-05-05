"""Solver for the atomic Kohn-Sham equation, including all-electron and pseudo-potential calculation."""

from .atomic import AtomicInput, solve_atomic
from .pseudo_atomic import PseudoAtomicInput, PseudoAtomicResult, solve_pseudo_atomic
from .version import VERSION as __version__

__all__ = [
    "AtomicInput",
    "PseudoAtomicInput",
    "PseudoAtomicResult",
    "__version__",
    "solve_atomic",
    "solve_pseudo_atomic",
]
