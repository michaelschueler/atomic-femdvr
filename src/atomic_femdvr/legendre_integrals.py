"""Indefinite-integral tables of Legendre-polynomial products on each FEDVR element."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import eval_legendre

from atomic_femdvr.legendre import Legendre


# ===================================================================
def gen_btensor_ode(N: int) -> np.ndarray:
    """Build the rank-3 tensor of pairwise Legendre indefinite integrals at the LGL nodes."""
    np.zeros([N, N, N])

    leg = Legendre(N)
    x_i = leg.x_i

    def integrand(t, x):
        """ODE RHS: outer product ``P_i(t) P_j(t)`` flattened to a vector."""
        pn = np.zeros(N)
        for n in range(N):
            pn[n] = eval_legendre(n, t)
        return np.outer(pn, pn).flatten()

    y0 = np.zeros([N, N]).flatten()

    sol = solve_ivp(integrand, [-1, 1], y0, t_eval=x_i, method="DOP853", rtol=1e-13, atol=1e-13)

    y = sol.y.T
    return y.reshape([N, N, N])


# ===================================================================
def gen_bvector_ode(N: int) -> np.ndarray:
    """Build the indefinite integrals of the first ``N`` Legendre polynomials at the LGL nodes."""
    np.zeros([N, N])

    leg = Legendre(N)
    x_i = leg.x_i

    def integrand(t, x):
        """ODE RHS: vector of Legendre polynomials ``P_n(t)``."""
        pn = np.zeros(N)
        for n in range(N):
            pn[n] = eval_legendre(n, t)
        return pn

    y0 = np.zeros(N)

    sol = solve_ivp(integrand, [-1, 1], y0, t_eval=x_i, method="DOP853", rtol=1e-13, atol=1e-13)

    y = sol.y.T
    return y


# ===================================================================
def get_legendre_integrals(leg: Legendre, xp: np.ndarray) -> np.ndarray:
    """Per-element table of Legendre-basis indefinite integrals scaled to ``xp``."""
    B_in = gen_bvector_ode(leg.N)

    ne = len(xp) - 1
    L_integs = np.zeros([ne, leg.N, leg.N])
    for i in range(ne):
        h = 0.5 * (xp[i + 1] - xp[i])
        L_integs[i, :, :] = h * B_in @ leg.S_ni

    return L_integs


# ===================================================================
