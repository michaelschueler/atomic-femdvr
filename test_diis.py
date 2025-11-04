import numpy as np


class DIIS:
    def __init__(self, max_history=6):
        self.max_history = max_history
        self.x_list = []  # stored iterates
        self.e_list = []  # stored residuals

    def update(self, x, e):
        """Store a new pair (x, e)."""
        self.x_list.append(x.copy())
        self.e_list.append(e.copy())

        # Keep only the most recent entries
        if len(self.x_list) > self.max_history:
            self.x_list.pop(0)
            self.e_list.pop(0)

    def extrapolate(self):
        """Return the DIIS-extrapolated x."""
        m = len(self.e_list)
        if m < 2:
            # Not enough history — just return the last x
            return self.x_list[-1]

        # Build B matrix
        B = np.empty((m + 1, m + 1))
        B[-1, :] = B[:, -1] = -1
        B[-1, -1] = 0
        for i in range(m):
            for j in range(m):
                B[i, j] = np.dot(self.e_list[i].ravel(), self.e_list[j].ravel())

        # Right-hand side
        rhs = np.zeros(m + 1)
        rhs[-1] = -1

        # Solve for coefficients
        coeff = np.linalg.solve(B, rhs)[:-1]

        # Extrapolate new x
        x_new = np.zeros_like(self.x_list[0])
        for c, x in zip(coeff, self.x_list):
            x_new += c * x
        return x_new

def main():

    def F(x):
        return np.cos(x)

    x = np.array([0.5])  # initial guess
    diis = DIIS(max_history=4)

    for iteration in range(30):
        x_out = F(x)
        e = x_out - x
        diis.update(x, e)

        if iteration > 1:
            x = diis.extrapolate()
        else:
            x = x_out

        print(f"Iter {iteration:02d}: x = {x[0]:.8f}, residual = {abs(e[0]):.3e}")
        if abs(e[0]) < 1e-10:
            break

if __name__ == "__main__":
    main()
