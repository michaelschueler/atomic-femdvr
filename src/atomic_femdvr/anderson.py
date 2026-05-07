"""Anderson mixing for SCF fixed-point iteration."""

from collections.abc import Callable

import numpy as np


class AndersonMixing:
    """Fixed-window Anderson mixing of an iterate ``x`` and its image ``y = F(x)``."""

    def __init__(self, max_history: int = 5) -> None:
        """Create an Anderson mixer that retains up to ``max_history`` past iterates."""
        self.max_history = max_history
        self.x_list: list[np.ndarray] = []  # stored iterates
        self.y_list: list[np.ndarray] = []  # stored function values
        self.e_list: list[np.ndarray] = []  # stored residuals

    def update(self, x: np.ndarray, y: np.ndarray, e: np.ndarray) -> None:
        """Store a new pair (x, e)."""
        self.x_list.append(x.copy())
        self.y_list.append(y.copy())
        self.e_list.append(e.copy())

        # Keep only the most recent entries
        if len(self.x_list) > self.max_history + 1:
            self.x_list.pop(0)
            self.y_list.pop(0)
            self.e_list.pop(0)

    def extrapolate(
        self,
        dot_product: Callable[[np.ndarray, np.ndarray], float],
        beta: float,
    ) -> np.ndarray:
        """Return the Anderson-mixed x."""
        m = len(self.e_list) - 1

        if m < 2:
            # Not enough history — just return the last x
            return (1.0 - beta) * self.x_list[-1] + beta * self.y_list[-1]

        # Build A matrix
        A = np.empty([m, m], dtype=self.x_list[0].dtype)
        b = np.empty(m, dtype=self.x_list[0].dtype)

        delta_r = []
        for i in range(m):
            delta_r.append(self.e_list[-1] - self.e_list[-1 - (i + 1)])

        for i in range(m):
            for j in range(m):
                A[i, j] = dot_product(delta_r[i], delta_r[j])

        for i in range(m):
            b[i] = dot_product(self.e_list[-1], delta_r[i])

        # Solve for coefficients
        coeff = np.linalg.solve(A, b)

        u = self.x_list[-1].copy()
        v = self.y_list[-1].copy()

        for i in range(m):
            u += coeff[i] * (self.x_list[-1 - (i + 1)] - self.x_list[-1])
            v += coeff[i] * (self.y_list[-1 - (i + 1)] - self.y_list[-1])

        x_new = u + beta * (v - u)
        return x_new
