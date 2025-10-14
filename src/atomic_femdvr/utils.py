import matplotlib.pyplot as plt


#==================================================================
def PrintTime(tic: float, toc: float, msg: str):
    """
    Print the elapsed time for a given operation.
    """
    elapsed = toc - tic
    if elapsed < 1:
        print(f"Time[{msg}] : {elapsed * 1000:.2f} ms")
    elif elapsed > 300:
        print(f"Time[{msg}] : {elapsed / 60:.2f} m")
    else:
        print(f"Time[{msg}] : {elapsed:.2f} s")
#==================================================================
#==================================================================
def GetOrbitalLabel(n: int, l: int) -> str:
    """
    Get the orbital label for a given principal quantum number n and angular momentum quantum number l.
    """
    l_labels = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j']

    nq = n + l + 1  # Principal quantum number

    if l < len(l_labels):
        return f"{nq}{l_labels[l]}"
    else:
        return f"{nq}l{l}"  # Fallback for higher angular momentum states
#==================================================================
def PrintEigenvalues(lmax: int, eigenvalues, energy_shifts=None):
    """
    Print the eigenvalues for each angular momentum quantum number.
    """
    Hr_to_eV = 2. * 13.605693009  # Hartree to eV conversion factor


    print(40 * '-')
    print("eigenvalues (in eV)".center(40))
    print(40 * '-')
    for l in range(lmax + 1):
        print(f"l = {l}")
        eps_bound = eigenvalues.get(f'{l}', [])
        n_bound = len(eps_bound)
        if n_bound == 0:
            print("  No bound states found.")
        else:
            for n in range(n_bound):
                orb = GetOrbitalLabel(n, l)
                print(f"  E({orb}) = {Hr_to_eV * eps_bound[n]:.6f} eV")

        if energy_shifts is not None:
            if l < len(energy_shifts):
                print(f"  Energy shift = {Hr_to_eV * energy_shifts[l]:.6f} eV")
    print(40 * '-')
#==================================================================
def PlotWavefunctions(r_grid, psi, lmax: int, eigenvalues):
    """
    Plot the wavefunctions for each angular momentum quantum number.
    """
    _, ax = plt.subplots(1, lmax + 1, figsize=(4*(lmax + 1), 6))

    for l in range(lmax + 1):
        ax[l].set_title(rf"$\ell$ = {l}")
        ax[l].set_xlabel("r (a.u.)")
        ax[l].set_ylabel("wave-function")

        eps_bound = eigenvalues.get(f'{l}', [])
        n_bound = len(eps_bound)

        for n in range(n_bound):
            orb = GetOrbitalLabel(n, l)
            ax[l].plot(r_grid, psi[l, n, :], label=orb)

        ax[l].legend()
        ax[l].set_xlim([0, r_grid[-1]])
        # ax[l].set_ylim([0, np.max(psi**2) * 1.1])

    plt.tight_layout()
    plt.show()
