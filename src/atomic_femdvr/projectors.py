import os

import numpy as np


#----------------------------------------------------------
def write_projector_file(out_dir:str, elem:str, tag:str, proj_l:list,
                       basis:np.ndarray, rs:np.ndarray) -> None:
    fname = os.path.join(out_dir, f"{elem}_{tag}.dat")

    nproj = basis.shape[0]
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
                s += f"{basis[j, i]} "
            f.write(s + "\n")
