"""Adaptive ODE-driven generation of FEDVR element boundaries."""

import numpy as np


# =================================================================
def adaptive_runge_kutta_23(f, y0, t0, t1, h_min, h_max, tol, arg=None):
    """Adaptive Runge-Kutta 2(3) integrator with bounded step size."""

    def rk23_step(f, y, t, h, arg=None):
        """Take one embedded RK2/RK3 step and return both estimates."""
        if arg is not None:

            def fn(t, y):
                """Bind ``arg`` into ``f`` so the inner stepper sees ``f(t, y)``."""
                return f(t, y, arg)
        else:
            fn = f
        # Runge-Kutta 2-3 step
        k1 = fn(t, y)
        k2 = fn(t + h / 2, y + h / 2 * k1)
        k3 = fn(t + h, y + h * k2)

        y2 = y + h * (k1 + 4 * k2 + k3) / 6
        y3 = y + h * (k1 + 3 * k2) / 4

        return y2, y3

    t = t0
    y = y0
    h = h_max
    t_values = [t]
    y_values = [y]
    while t < t1:
        if t + h > t1:
            h = t1 - t

        y2, y3 = rk23_step(f, y, t, h, arg=arg)

        error = np.linalg.norm(y3 - y2)

        if error < tol:
            t += h
            y = y2
            t_values.append(t)
            y_values.append(y)
        else:
            # Decrease step size
            if h <= h_min:
                # If already at minimal step size, proceed with h_min
                t += h
                y = y2
                t_values.append(t)
                y_values.append(y)
                h = h_min
            else:
                h = max(h * 0.5, h_min)

    return np.array(t_values), np.array(y_values)


# =================================================================
def optimize_elements(
    Zc: float,
    h_min: float,
    h_max: float,
    Rmax: float,
    tol: float = 1.0e-2,
    Za: float = 1.0,
    method: str = "exponential",
) -> np.ndarray:
    """Place FEDVR element boundaries on ``[0, Rmax]`` according to a weight rule."""
    if method.lower() == "exponential":

        def wght_fnc(r):
            """Exponential refinement weight peaking near the nucleus."""
            return np.exp(-Zc * r) + np.exp(-Za * r)

    elif method.lower() == "wkb":

        def Vc_fnc(r):
            """Smoothed nuclear Coulomb potential used to define the WKB weight."""
            return -Zc / np.sqrt(r**2 + 1.0e-2)

        def wght_fnc(r):
            """WKB local-momentum weight ``sqrt(2|V(r)|)``."""
            return np.sqrt(2.0 * np.abs(Vc_fnc(r)))
    else:
        raise ValueError(f"Unknown method '{method}' for optimizing elements.")

    def wght_fnc_r(r, y, L):
        """ODE RHS: weight evaluated on the reflected coordinate ``Rmax - r``."""
        return wght_fnc(Rmax - r)

    xk, _wk = adaptive_runge_kutta_23(wght_fnc_r, 0.0, 0.0, Rmax, h_min, h_max, tol, arg=Rmax)
    grid = np.flip(Rmax - xk)

    return grid


# =================================================================
