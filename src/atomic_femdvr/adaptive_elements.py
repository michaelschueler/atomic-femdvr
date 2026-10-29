import numpy as np


#=================================================================
def adaptive_runge_kutta_23(f, y0, t0, t1, h_min, h_max, tol, arg=None):
    # Adaptive Runge-Kutta 2-3 method for solving ODEs

    def rk23_step(f, y, t, h, arg=None):
        if arg is not None:
            fn = lambda t, y: f(t, y, arg)
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


        # print(f"t: {t:.4f}, h: {h:.4f}, error: {error:.4e}")

    return np.array(t_values), np.array(y_values)
#=================================================================
def adaptive_runge_kutta_23_outward(f, y0, t0, t1, h_min, h_max, tol):
    """
    Adaptive RK23 propagating outward from t0 to t1.

    Unlike the inward variant, the step size is allowed to grow (up to h_max)
    when the local error is small, so coarser elements are placed automatically
    as the weight function decays away from the nucleus.
    """
    def rk23_step(fn, y, t, h):
        k1 = fn(t, y)
        k2 = fn(t + h / 2, y + h / 2 * k1)
        k3 = fn(t + h, y + h * k2)
        y2 = y + h * (k1 + 4 * k2 + k3) / 6
        y3 = y + h * (k1 + 3 * k2) / 4
        return y2, y3

    t = t0
    y = y0
    h = h_min  # start small at the core boundary, grow as weight decays
    t_values = [t]
    y_values = [y]

    while t < t1:
        if t + h > t1:
            h = t1 - t

        y2, y3 = rk23_step(f, y, t, h)
        error = np.linalg.norm(y3 - y2)

        if error < tol:
            t += h
            y = y2
            t_values.append(t)
            y_values.append(y)
            # Error-based growth: jump h aggressively when error << tol,
            # capped at 5x per step to avoid overshooting.
            growth = min(5.0, 0.9 * (tol / max(error, 1e-15)) ** (1.0 / 3.0))
            h = min(h * growth, h_max)
        else:
            if h <= h_min:
                # at minimum step, accept and stay at h_min
                t += h
                y = y2
                t_values.append(t)
                y_values.append(y)
            else:
                h = max(h * 0.5, h_min)

    return np.array(t_values), np.array(y_values)
#=================================================================
def optimize_elements_ae(Zc: float, h_min: float, h_max: float, Rmax: float,
                         tol: float = 1.0e-2, Za: float = 1.0,
                         n_core_factor: float = 5.0,
                         method: str = 'exponential') -> np.ndarray:
    """
    Element boundaries for all-electron calculations.

    Core region [0, r_c]: uniform spacing with n_core elements, spacing
    chosen as close to h_min as possible while exactly filling the interval.

    Outer region [r_c, Rmax]: outward adaptive RK23 starting at h_min,
    step size grows as the weight function decays, reaching h_max in the
    asymptotic region.

    r_c = n_core_factor / Zc defines the core boundary (default: 5/Z).
    """
    r_c = n_core_factor / Zc

    # Core: uniform grid with spacing as close to h_min as possible
    n_core = max(1, round(r_c / h_min))
    core_grid = np.linspace(0.0, r_c, n_core + 1)

    # Outer: outward adaptive RK23 from r_c to Rmax
    if method.lower() == 'exponential':
        wght_fnc = lambda r: np.exp(-Zc * r) + np.exp(-Za * r)
    elif method.lower() == 'wkb':
        Vc_fnc = lambda r: -Zc / np.sqrt(r**2 + 1.0e-2)
        wght_fnc = lambda r: np.sqrt(2.0 * np.abs(Vc_fnc(r)))
    else:
        raise ValueError(f"Unknown method '{method}' for optimizing elements.")

    outer_t, _ = adaptive_runge_kutta_23_outward(
        lambda r, _: wght_fnc(r), 0.0, r_c, Rmax, h_min, h_max, tol
    )

    # Concatenate, dropping the duplicate point at r_c
    return np.concatenate([core_grid, outer_t[1:]])
#=================================================================
def optimize_elements(Zc: float, h_min: float, h_max: float, Rmax: float,
                      tol: float = 1.0e-2, Za: float = 1.0,
                      method: str = 'exponential') -> np.ndarray:

    if method.lower() == 'exponential':
        wght_fnc = lambda r: np.exp(-Zc *r) + np.exp(-Za * r)

    elif method.lower() == 'wkb':
        Vc_fnc  = lambda r: -Zc / np.sqrt(r**2 + 1.0e-2)
        wght_fnc = lambda r: np.sqrt(2.0 * np.abs(Vc_fnc(r)))
    else:
        raise ValueError(f"Unknown method '{method}' for optimizing elements.")

    wght_fnc_r = lambda r, y, L: wght_fnc(Rmax - r)
    xk, wk = adaptive_runge_kutta_23(wght_fnc_r, 0.0, 0.0, Rmax,
                                    h_min, h_max, tol, arg=Rmax)
    grid = np.flip(Rmax - xk)

    return grid
#=================================================================
