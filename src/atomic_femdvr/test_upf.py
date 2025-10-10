import json
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from upf_interface import upf_class
from utils import PrintTime


#==================================================================
def main(argv):

    # read json input file
    if len(argv) < 2:
        print("Usage: python test_upf.py <input_file> <upf file>")
        return

    input_file = argv[0]
    try:
        with open(input_file) as f:
            params = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    upf_file = argv[1]



    upflib_dir = params.get('upflib_dir', '')
    lib_ext = params.get('lib_ext', 'so')

    tic = perf_counter()
    upf = upf_class(upflib_dir, lib_ext)
    toc = perf_counter()
    PrintTime(tic, toc, "Loading UPF library")

    tic = perf_counter()
    upf.Read_UPF(upf_file)
    toc = perf_counter()
    PrintTime(tic, toc, "Reading UPF file")

    tic = perf_counter()
    upf.ReadWavefunctions()
    toc = perf_counter()
    PrintTime(tic, toc, "Reading UPF wavefunctions")


    tic = perf_counter()
    upf.Read_PP()
    toc = perf_counter()
    PrintTime(tic, toc, "Reading UPF pseudo")


    # print occupations
    for iwfc in range(upf.nwfc):
        print(f"Wavefunction {iwfc+1}: n={upf.nchi[iwfc]}, l={upf.lchi[iwfc]}, occupation={upf.oc[iwfc]}")

    fig, ax = plt.subplots(figsize=(12, 8))

    for iwfc in range(upf.nwfc):
        r = upf.r

        psi = np.copy(upf.chi[:, iwfc])
        ax.plot(r, psi, label=f'wfc {iwfc+1} (l={upf.lchi[iwfc]}, n={upf.nchi[iwfc]})')

    ax.set_xlabel('r (a.u.)')
    ax.set_ylabel('Wavefunction')
    ax.set_title('Pseudo Wavefunctions')
    ax.legend()
    plt.tight_layout()
    plt.show()

    rho = np.zeros(upf.mesh, dtype=np.float64)
    for iwfc in range(upf.nwfc):
        rho[1:] += upf.oc[iwfc] * np.abs(upf.chi[1:, iwfc])**2 / upf.r[1:]**2
    rho[0] = rho[1]  # Set the first element to zero to avoid division by zero

    # Integrate to get total charge
    total_charge = simpson(rho * upf.r**2, x=upf.r)
    print(f"Total charge from wavefunctions: {total_charge:.6f}")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(upf.r, rho, label='Charge Density')
    ax.set_xlabel('r (a.u.)')
    ax.set_ylabel('Charge Density')
    ax.set_title('Charge Density from Wavefunctions')
    ax.legend()
    plt.show()


    fig, ax = plt.subplots(figsize=(12, 8))

    for ibeta in range(upf.nbeta):
        r = upf.r[0:upf.kbeta_max]

        beta = np.copy(upf.beta[0:upf.kbeta_max, ibeta])
        ax.plot(r, beta, label=f'ibeta {ibeta+1} (l={upf.lll[ibeta]})')

    ax.set_xlabel('r (a.u.)')
    ax.set_ylabel('Beta function')
    ax.legend()
    plt.tight_layout()
    plt.show()

#==================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
