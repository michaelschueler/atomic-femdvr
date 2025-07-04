import sys
import numpy as np
import json
from time import perf_counter
import matplotlib.pyplot as plt

from utils import PrintTime
from upf_interface import upf_class
#==================================================================
def main(argv):

    # read json input file
    if len(argv) < 3:
        print("Usage: python test_upf.py <input_file> <upf file>")
        return
    
    input_file = argv[1]
    try:
        with open(input_file, 'r') as f:
            params = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    upf_file = argv[2]

    

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
    main(sys.argv)