import getopt
import json
import os
import sys
from time import perf_counter

from PseudoAtomDFT import PseudoAtomDFT
from utils import PlotWavefunctions, PrintEigenvalues, PrintTime


#==================================================================
def ReadInput(fname):
    """
    Read input parameters from a JSON file.
    """
    with open(fname) as f:
        data = json.load(f)

    pseudo_config = data.get('pseudo_config', {})
    if not pseudo_config:
        raise ValueError("No 'pseudo_config' found in the input file.")

    sysparams = data.get('sysparams', {})
    if not sysparams:
        raise ValueError("No 'sysparams' found in the input file.")
    solver = data.get('solver', {})
    if not solver:
        raise ValueError("No 'solver' parameters found in the input file.")

    dft = data.get('dft', {})
    confinement = data.get('confinement', {})
    proj = data.get('projector', {})

    return pseudo_config, sysparams, solver, dft, confinement, proj
#==================================================================


#==================================================================
def main(argv):

    short_options = "hpi:t:e:"
    long_options = ["help", "plot", "input=", "task=", "export="]

    print(60 * '*')
    print("Pseudo-atomic Schrödinger Equation Solver".center(60))
    print(60 * '*')
    tic = perf_counter()

    try:
        opts, args = getopt.getopt(argv, short_options, long_options)
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    # get input and output file names
    input_file = ''
    task_list = []
    plot_results = False
    export_dir = None

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: python pseudo_atomic.py -i <input_file> -t <task> [--plot]")
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-t", "--task"):
            task = arg
            # split task into components if needed
            if task:
                task_list = task.split(',')
                # convert to lowercase for consistency
                task_list = [t.strip().lower() for t in task_list]
        elif opt == "--plot":
            plot_results = True
        elif opt == "--export":
            export_dir = arg

    if not input_file:
        print("Error: Input file is required. Use -i <input_file> to specify it.")
        sys.exit(2)

    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(2)

    if len(task_list) == 0:
        print("No tasks specified. Use -t <task> to specify tasks.")
        sys.exit(2)


    # Read input parameters
    pseudo_config, sysparams, solver, dft, confinement, proj = ReadInput(input_file)

    # Initialize the PseudoAtomDFT class
    tic = perf_counter()
    pseudo_atom = PseudoAtomDFT(pseudo_config, sysparams, solver, dft)
    toc = perf_counter()
    PrintTime(tic, toc, "Initializing PseudoAtomDFT")
    print("")

    print(f"number of elements: {len(pseudo_atom.r_elements) - 1}")
    print(f"number of grid points: {pseudo_atom.num_grid}\n")

    # Read UPF file
    tic = perf_counter()
    pseudo_atom.ReadUPF(read_density=True, read_potential=True)
    toc = perf_counter()
    PrintTime(tic, toc, "Reading UPF file")
    print("")

    restart_success = pseudo_atom.ReadDensityPotential()
    if restart_success:
        print("Restarting from saved density and potential.\n")
    else:
        print("No saved density and potential found. Starting from scratch.\n")


    scf_done = False
    nscf_done = False

    if 'scf' in task_list:

        tic = perf_counter()
        conv_tol = dft.get('conv_tol', 1.0e-6)
        max_iter = dft.get('max_iter', 100)
        alpha = dft.get('alpha', 0.6)

        if max_iter > 0:
            num_iter, err = pseudo_atom.KS_SelfConsistency(max_iter=max_iter, tol=conv_tol, alpha=alpha)

            if err < conv_tol:
                print(f"Self-consistency converged in {num_iter} iterations with error: {err:.2e}")
            else:
                print(f"Self-consistency did not converge within {max_iter} iterations. Final error: {err:.2e}")
        else:
            print("Skipping self-consistency loop as max_iter is set to 0.")

        toc = perf_counter()
        PrintTime(tic, toc, "SCF")

        eigenvalues, psi = pseudo_atom.GetBoundStates()

        PrintEigenvalues(pseudo_atom.upf.lmax, eigenvalues)

        pseudo_atom.SaveDensityPotential()

        if plot_results:
            PlotWavefunctions(pseudo_atom.grid, psi, pseudo_atom.upf.lmax, eigenvalues)

        scf_done = True

    if 'optimize' in task_list:
        if not scf_done and not restart_success:
            print("Error: Non-SCF task requires SCF to be completed first or a valid restart file.")
            sys.exit(2)

        tic = perf_counter()
        Q_opt = pseudo_atom.OptimizeSoftCoul(confinement)
        toc = perf_counter()
        PrintTime(tic, toc, "Optimizing Soft Coulomb Confinement")
        print("")
        print(f"Optimized soft Coulomb confinement parameter Q: {Q_opt:.4f}\n")
        confinement['softcoul_charge'] = Q_opt

    if 'nscf' in task_list:
        if not scf_done and not restart_success:
            print("Error: Non-SCF task requires SCF to be completed first or a valid restart file.")
            sys.exit(2)

        tic = perf_counter()
        lmax = sysparams.get('lmax', 2)
        nmax = sysparams.get('nmax', 2)
        energy_shifts, eigenvalues, psi = pseudo_atom.GetStatesEnergyShift(lmax, nmax, confinement=confinement)
        toc = perf_counter()
        PrintTime(tic, toc, "Non-SCF Calculation")
        print("")

        PrintEigenvalues(lmax, eigenvalues, energy_shifts=energy_shifts)

        if plot_results:
            PlotWavefunctions(pseudo_atom.grid, psi, lmax, eigenvalues)

        nscf_done = True

    if export_dir is not None:
        if not nscf_done:
            print("Error: Exporting projectors requires non-SCF task to be completed first.")
            sys.exit(2)

        tic = perf_counter()
        nr = proj.get('nr', 1001)
        rmin = proj.get('rmin', 1.0e-8)
        pseudo_atom.ExportProjector(lmax, nmax, psi, confinement, export_dir, nr=nr, rmin=rmin)
        toc = perf_counter()
        PrintTime(tic, toc, "Exporting Projectors")

    toc = perf_counter()
    PrintTime(tic, toc, "Total")
    print(60 * '*')

#==================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
