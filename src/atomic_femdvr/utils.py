import numpy as np


#==================================================================
def print_time(tic: float, toc: float, msg: str) -> None:
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
def parse_tasks(task_list: tuple[str, ...], allowed: tuple[str, ...]) -> list[str]:
    """Split comma-separated task options (``-t scf,nscf`` or ``-t scf -t nscf``) and validate them."""
    tasks = [t.strip().lower() for entry in task_list for t in entry.split(',') if t.strip()]
    unknown = [t for t in tasks if t not in allowed]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}. Allowed tasks: {list(allowed)}")
    return tasks
#==================================================================
def get_orbital_label(n: int, l: int, n_lowest: dict[int, int] | None = None) -> str:
    """
    Get the orbital label for the n-th (0-based) radial state with angular momentum l.

    n_lowest maps l to the principal quantum number of the lowest state in that channel
    (e.g. 4 for the 4s valence state of a pseudo-atom); the default is l + 1.
    """
    l_labels = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j']

    nq = n + (n_lowest or {}).get(l, l + 1)  # Principal quantum number

    if l < len(l_labels):
        return f"{nq}{l_labels[l]}"
    else:
        return f"{nq}l{l}"  # Fallback for higher angular momentum states
#==================================================================
def print_eigenvalues(lmax: int, eigenvalues: dict, energy_shifts: list | None = None,
                      n_lowest: dict[int, int] | None = None) -> None:
    """
    Print the eigenvalues for each angular momentum quantum number.
    """
    Hr_to_eV = 2. * 13.605693009  # Hartree to eV conversion factor


    print(40 * '-')
    print("eigenvalues".center(40))
    print(40 * '-')
    for l in range(lmax + 1):
        print(f"l = {l}")
        eps_bound = eigenvalues.get(f'{l}', [])
        n_bound = len(eps_bound)
        if n_bound == 0:
            print("  No bound states found.")
        else:
            for n in range(n_bound):
                orb = get_orbital_label(n, l, n_lowest)
                print(f"  E({orb}) = { eps_bound[n]:.6f} Hr = {Hr_to_eV * eps_bound[n]:.6f} eV")

        if energy_shifts is not None:
            if l < len(energy_shifts):
                print(f"  Energy shift = {Hr_to_eV * energy_shifts[l]:.6f} eV")
    print(40 * '-')
#==================================================================
def plot_wavefunctions(r_grid: np.ndarray, psi: np.ndarray, lmax: int, eigenvalues: dict,
                       n_lowest: dict[int, int] | None = None) -> None:
    """
    Plot the wavefunctions for each angular momentum quantum number.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("Plotting requires matplotlib: pip install 'atomic_femdvr[plot]'") from err

    _, ax = plt.subplots(1, lmax + 1, figsize=(4*(lmax + 1), 6), squeeze=False)
    ax = ax[0]

    for l in range(lmax + 1):
        ax[l].set_title(rf"$\ell$ = {l}")
        ax[l].set_xlabel("r (a.u.)")
        ax[l].set_ylabel("wave-function")

        eps_bound = eigenvalues.get(f'{l}', [])
        n_bound = len(eps_bound)

        for n in range(n_bound):
            orb = get_orbital_label(n, l, n_lowest)
            ax[l].plot(r_grid, psi[l, n, :], label=orb)

        ax[l].legend()
        ax[l].set_xlim([0, r_grid[-1]])
        # ax[l].set_ylim([0, np.max(psi**2) * 1.1])

    plt.tight_layout()
    plt.show()
