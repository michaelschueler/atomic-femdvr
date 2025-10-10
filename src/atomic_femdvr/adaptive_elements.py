import numpy as np
#=================================================================
def AdapativeRungaKutta23(f, y0, t0, t1, h_min, h_max, tol, arg=None):
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
def OptimizeElements(Zc, h_min, h_max, Rmax, tol=1.0e-2, Za=1.0):

    wght_fnc = lambda r: np.exp(-Zc *r) + np.exp(-Za * r)
    wght_fnc_r = lambda r, y, L: wght_fnc(Rmax - r)

    xk, wk = AdapativeRungaKutta23(wght_fnc_r, 0.0, 0.0, Rmax, 
                                   h_min, h_max, tol, arg=Rmax)
    grid = np.flip(Rmax - xk)

    return grid
#=================================================================