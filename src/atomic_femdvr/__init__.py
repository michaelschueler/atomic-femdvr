"""Solver for the atomic Kohn-Sham equation.

This package provides finite-element / discrete-variable-representation
(FEM-DVR) radial Kohn-Sham solvers for isolated atoms, in two flavours:

* **All-electron** -- :func:`solve_atomic` consumes a
  :class:`FullAtomicInput` and produces a self-consistent all-electron
  density and bound-state spectrum.
* **Pseudo-atomic** -- :func:`solve_pseudo_atomic` consumes a
  :class:`PseudoAtomicInput` (which references a UPF norm-conserving
  pseudopotential) and produces a :class:`PseudoAtomicResult` with
  eigenvalues and per-:math:`\\ell` energy shifts.

Both solvers expose a CLI: ``atomic_femdvr atomic <input.json>`` and
``atomic_femdvr pseudoatomic <input.json>``.
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
