"""Solver for the atomic Kohn-Sham equation, including all-electron and pseudo-potential calculation."""

from .api import hello, square

# being explicit about exports is important!
__all__ = [
    "hello",
    "square",
]
