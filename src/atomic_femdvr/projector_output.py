import os
import numpy as np
import h5py

from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.bessel_transform import bessel_integral
#----------------------------------------------------------
def write_projector_file(basis: FEDVR_Basis, phi: np.ndarray, 
                         elem:str, tag:str, nr:int = 1001, rmin:float=1.0e-8, 
                         bessel_method:str = 'simpson', bessel_npoints:int = 41,
                         qgrid:np.ndarray | None = None, rpow:int = 1,
                         out_dir:str = './', output_format:str = 'qe') -> None:

    if output_format.lower() == 'qe':
        # Quantum ESPRESSO format
        lmax = phi.shape[0] - 1
        nmax = phi.shape[1] - 1

        # check if nr is odd 
        if nr % 2 == 0:
            nr += 1

        Rmax = basis.xp[-1]
        rs = np.logspace(np.log10(rmin), np.log10(Rmax), nr)

        # interpolate wavefunctions onto rs grid
        larr = []
        psi_interp = np.zeros([lmax + 1, nmax + 1, nr])
        for l in range(lmax + 1):
            for n in range(nmax + 1):
                psi_interp[l, n, :] = basis.interpolate(phi[l, n, :], rs)
                larr.append(l)

        nproj = len(larr)
        psi_interp = psi_interp.reshape(nproj, nr)

        write_projector_qe(out_dir, elem, tag, larr, psi_interp, rs)

    elif output_format.lower() == 'hdf5':

        write_projector_hdf5(out_dir, elem, tag, phi, basis)

    elif output_format.lower() == 'bessel' and qgrid is not None:


        lmax = phi.shape[0] - 1
        nmax = phi.shape[1] - 1
        phi_bessel = np.zeros([ lmax+1, nmax+1, len(qgrid)])

        for l in range(lmax + 1):
            for n in range(nmax + 1):
                phi_bessel[l, n, :] = bessel_integral(
                    basis, l, rpow, qgrid, phi[l, n, :],
                    npoints=bessel_npoints, method=bessel_method)


        write_bessel_hdf5(out_dir, elem, tag, phi_bessel, qgrid)
        
    else:
        raise ValueError(f"Unknown output format: {output_format}. Supported formats are 'qe', 'hdf5', 'bessel'.")

#----------------------------------------------------------
def write_projector_qe(out_dir:str, elem:str, tag:str, proj_l:list,
                       phi:np.ndarray, rs:np.ndarray) -> None:
    fname = os.path.join(out_dir, f"{elem}_{tag}_qe.dat")

    nproj = phi.shape[0]
    nr = len(rs)

    with open(fname, 'w') as f:
        f.write(f"{nr} {nproj} \n")
        for j in range(nproj):
            f.write(f"{proj_l[j]} ")
        f.write("\n")
        for i, r in enumerate(rs):
            x = np.log(r)
            s = f"{x}  {r} "
            for j in range(nproj):
                s += f"{phi[j, i]} "
            f.write(s + "\n")
#----------------------------------------------------------
def write_projector_hdf5(out_dir:str, elem:str, tag:str, 
                         phi:np.ndarray, basis:FEDVR_Basis) -> None:
    
    fname = os.path.join(out_dir, f"{elem}_{tag}_wfc.h5")

    with h5py.File(fname, 'w') as f:
        f.attrs['ne'] = basis.ne
        f.attrs['ng'] = basis.ng
        f.attrs['lmax'] = phi.shape[0] - 1
        f.attrs['nmax'] = phi.shape[1] - 1
        f.create_dataset('xp', data=basis.xp)
        f.create_dataset('wf', data=phi)
#----------------------------------------------------------
def write_bessel_hdf5(out_dir:str, elem:str, tag:str,
                       phi_bessel:np.ndarray, qgrid:np.ndarray) -> None:
    
    fname = os.path.join(out_dir, f"{elem}_{tag}_bessel.h5")

    with h5py.File(fname, 'w') as f:
        f.attrs['lmax'] = phi_bessel.shape[0] - 1
        f.attrs['nmax'] = phi_bessel.shape[1] - 1
        f.create_dataset('qgrid', data=qgrid)
        f.create_dataset('wf_bessel', data=phi_bessel)
