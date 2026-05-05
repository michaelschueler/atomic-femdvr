import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


class DIIS:
    def __init__(self, max_history: int = 6) -> None:
        self.max_history = max_history
        self.x_list = []  # stored iterates
        self.e_list = []  # stored residuals

    def update(self, x: np.ndarray, e: np.ndarray) -> None:
        """Store a new pair (x, e)."""
        self.x_list.append(x.copy())
        self.e_list.append(e.copy())

        # Keep only the most recent entries
        if len(self.x_list) > self.max_history:
            self.x_list.pop(0)
            self.e_list.pop(0)

    def extrapolate(
        self,
        dot_product: Callable[[np.ndarray, np.ndarray], float],
        beta: float = 1.0,
    ) -> np.ndarray:
        """Return the DIIS-extrapolated x."""
        m = len(self.e_list)

        logger.debug("DIIS: m = %d", m)
        if m < 2:
            # Not enough history — just return the last x
            return beta * self.x_list[-1] + (1.0 - beta) * self.x_list[-2]

        # Build B matrix
        B = np.empty((m + 1, m + 1), dtype=self.x_list[0].dtype)
        B[-1, :] = 1.0
        B[:, -1] = 1.0
        B[-1, -1] = 0
        for i in range(m):
            for j in range(m):
                B[i, j] = dot_product(self.e_list[j], self.e_list[i])

        # Right-hand side
        rhs = np.zeros(m + 1)
        rhs[-1] = 1.0

        # Solve for coefficients
        coeff = np.linalg.solve(B, rhs)[:-1]

        logger.debug("DIIS: coeff = %s, sum = %.6f", coeff, np.sum(coeff))

        # Extrapolate new x
        x_new = np.zeros_like(self.x_list[0])
        for c, x in zip(coeff, self.x_list, strict=False):
            x_new += c * x

        x_new = (1.0 - beta) * self.x_list[-1] + beta * x_new

        return x_new
