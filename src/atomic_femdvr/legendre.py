"""Author: H. U.R. Strand, 2020."""

import logging

import numpy as np
import numpy.polynomial.legendre as leg

logger = logging.getLogger(__name__)


def legendre_spectral_derivative_matrix(N):
    """Return the ``N x N`` derivative matrix in the Legendre-spectral basis."""
    D_nn = np.zeros((N, N))
    k = np.arange(N)

    for n in range(N):
        s = 1 - n % 2
        D_nn[s:n:2, n] = 2 * k[s:n:2] + 1

    return D_nn


class Legendre:
    """Legendre-Gauss-Lobatto nodes, weights, and basis-conversion matrices on ``[-1, 1]``."""

    def __init__(self, N):
        """Precompute LGL nodes/weights and spectral/collocation transforms of order ``N``."""
        self.N = N
        n = np.arange(N)

        self.W_n = 2.0 / (2.0 * n + 1.0)
        self.W_n[-1] = 2.0 / (N - 1)

        self.x_i = self.__leggausslobatto_quadrature()
        self.w_i = self.__leggausslobatto_quadrature_weights(self.x_i)

        self.L_in = leg.legvander(self.x_i, N - 1)
        self.S_ni = (self.L_in * self.w_i[:, None] / self.W_n[None, :]).T

        self.D_ii = self.__derivative_matrix()
        self.D_nn = legendre_spectral_derivative_matrix(N)

    def to_collocation(self, f_nX):
        """Transform spectral coefficients to nodal values at the LGL points."""
        return np.tensordot(self.L_in, f_nX, axes=(1, 0))

    def to_spectral(self, f_iX):
        """Transform nodal values at the LGL points back to spectral coefficients."""
        return np.tensordot(self.S_ni, f_iX, axes=(1, 0))

    def __derivative_matrix(self):
        """Compute the LGL collocation differentiation matrix on ``[-1, 1]``."""
        x_i = self.x_i
        N = len(x_i)

        c_n = np.zeros(N)
        c_n[-1] = 1.0
        LN_i = leg.legval(x_i, c_n)

        # -- D_ij
        N = len(x_i)

        xx_ii = x_i[:, None] - x_i[None, :]
        xx_ii += np.eye(N)  # just to not get inverse warnings... :P

        D_ii = LN_i[:, None] / LN_i[None, :] / xx_ii

        diag = np.zeros(N)
        diag[0] = -(N - 1) * (N) / 4.0
        diag[-1] = +(N - 1) * (N) / 4.0

        D_ii += np.diag(diag) - np.diag(np.diag(D_ii))

        return D_ii

    def __leggausslobatto_quadrature_weights(self, x_i):
        """Closed-form LGL quadrature weights at the supplied nodes."""
        N = self.N - 1

        c_n = np.zeros(N + 1)
        c_n[-1] = 1.0

        w_i = 2.0 / (N * (N + 1.0)) / leg.legval(x_i, c_n) ** 2
        return w_i

    def __leggausslobatto_quadrature(self):
        """Compute the full set of LGL nodes on ``[-1, 1]`` (including endpoints)."""
        N = self.N - 1

        x_i = self.__x_i_guess()

        x_i = x_i[: N // 2] if N % 2 else x_i[: N // 2 - 1]

        x_i, _rerr = self.__newton_iter(x_i)

        x_i_out = np.zeros(N + 1)
        x_i_out[0] = -1.0
        x_i_out[-1] = 1.0

        if N % 2:
            x_i_out[1 : N // 2 + 1] = -x_i
            x_i_out[N // 2 + 1 : -1] = x_i[::-1]
        else:
            x_i_out[1 : N // 2] = -x_i
            x_i_out[N // 2 + 1 : -1] = x_i[::-1]

        return x_i_out

    def __x_i_guess(self):
        """Asymptotic starting guess for the interior LGL nodes."""
        N = self.N - 1

        k = np.arange(1, N + 1)
        theta_k = (4.0 * k - 1.0) / (4.0 * N + 2.0) * np.pi
        sigma_k = np.cos(theta_k) * (
            1
            - (N - 1.0) / (8.0 * N**3)
            - 1.0 / (384.0 * N**4) * (39.0 - 28.0 / np.sin(theta_k) ** 2)
        )

        x_i = 0.5 * (sigma_k[:-1] + sigma_k[1:])

        return x_i

    def __newton_iter(self, x_i_guess, niter_max=1000, tol=1e-16):
        """Newton-iterate the LGL node guesses to machine precision."""
        N = self.N - 1

        c_n = np.zeros(N + 1)
        c_n[-1] = 1.0

        dc_n = leg.legder(c_n)

        rerr = np.zeros_like(x_i_guess)
        x_i = np.zeros_like(x_i_guess)

        for i, x in enumerate(x_i_guess):
            converged = False
            for _iter in range(niter_max):
                L_n = leg.legval(x, c_n)
                dL_n = leg.legval(x, dc_n)
                ratio = (1 - x**2) * dL_n / (2 * x * dL_n - N * (N + 1) * L_n)
                x = x - ratio

                if np.abs(ratio) < tol:
                    converged = True
                    break

            if not converged:
                logger.warning(
                    "Legendre-Gauss-Lobatto Newton iteration not converged (relative error: %.3e)",
                    np.abs(ratio),
                )

            rerr[i] = np.abs(ratio)
            x_i[i] = x

        return x_i, rerr
