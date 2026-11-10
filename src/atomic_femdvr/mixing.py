"""Density mixing schemes for the Kohn-Sham self-consistency loop."""

import numpy as np


class LinearMixer:
    """Simple linear mixing: rho_in + alpha * (rho_out - rho_in)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        """Return the next input density."""
        return self.alpha * rho_out + (1.0 - self.alpha) * rho_in


class AndersonMixer:
    """Anderson (equivalently Pulay/DIIS) mixing of densities.

    The first step takes the output density unmixed; while fewer than two previous
    residuals are stored, it falls back to linear mixing with ``alpha``.
    """

    def __init__(self, alpha: float, history: int = 5) -> None:
        self.alpha = alpha
        self.history = history
        self.x_list: list[np.ndarray] = []  # input densities
        self.y_list: list[np.ndarray] = []  # output densities

    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        """Return the next input density."""
        self.x_list.append(rho_in.copy())
        self.y_list.append(rho_out.copy())
        if len(self.x_list) > self.history + 1:
            self.x_list.pop(0)
            self.y_list.pop(0)

        if len(self.x_list) == 1:
            return rho_out.copy()

        m = len(self.x_list) - 1
        if m < 2:
            return (1.0 - self.alpha) * rho_in + self.alpha * rho_out

        residuals = [y - x for x, y in zip(self.x_list, self.y_list, strict=True)]
        delta_r = [residuals[-1] - residuals[-2 - i] for i in range(m)]
        a = np.array([[np.dot(di, dj) for dj in delta_r] for di in delta_r])
        b = np.array([np.dot(residuals[-1], di) for di in delta_r])
        coeff = np.linalg.solve(a, b)

        u = self.x_list[-1].copy()
        v = self.y_list[-1].copy()
        for i in range(m):
            u += coeff[i] * (self.x_list[-2 - i] - self.x_list[-1])
            v += coeff[i] * (self.y_list[-2 - i] - self.y_list[-1])

        return u + self.alpha * (v - u)


def make_mixer(scheme: str, alpha: float, history: int) -> LinearMixer | AndersonMixer:
    """Create a density mixer for the given scheme ('linear' or 'anderson')."""
    if scheme == "linear":
        return LinearMixer(alpha)
    if scheme == "anderson":
        return AndersonMixer(alpha, history)
    raise ValueError(f"Unknown mixing scheme: {scheme}")
