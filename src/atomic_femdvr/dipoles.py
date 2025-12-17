import numpy as np
import h5py
from sympy.physics.wigner import gaunt


from atomic_femdvr.femdvr import FEDVR_Basis
#=================================================================
def radial_integrals(basis: FEDVR_Basis, psi: np.ndarray,
                     r_pow: int) -> np.ndarray:
    """
    Compute radial integrals of the form:
        I(ln, l'n') = ∫ r^r_pow * psi_{n, l}(r) psi_{n', l'}(r) dr
    """
    lmax = psi.shape[0] - 1
    nmax = psi.shape[1] - 1
    
    # integrals = np.zeros([lmax+1, nmax+1, lmax+1, nmax+1], dtype=np.float64)

    nx = (lmax + 1) * (nmax + 1)
    integrals = np.zeros([nx, nx], dtype=np.float64)


    ne = basis.ne
    ng = basis.ng
    xp = basis.xp
    grid = basis.get_gridpoints()

    for i in range(ne):
        psi_elem = np.ascontiguousarray(psi[:, :, i*ng : i*ng + ng + 1])
        psi_elem = np.reshape(psi_elem, [nx, ng + 1])
        r_elem = grid[i*ng : i*ng + ng + 1]
        h_elem = 0.5 * (xp[i+1] - xp[i])
        w_elem = h_elem * basis.leg.w_i

        psi_x_psi = np.einsum('ik,jk->ijk', psi_elem, psi_elem)
        integrand = (r_elem[None, None, :] ** r_pow) * psi_x_psi
        integrals += np.sum(integrand * w_elem[None, None, :], axis=2)

    integrals = integrals.reshape((lmax + 1, nmax + 1, lmax + 1, nmax + 1))

    return integrals
#=================================================================
def minus_one_pow(n: int) -> int:
    """Returns (-1)^n"""
    if n % 2 == 0:
        return 1
    else:
        return -1
#=================================================================
def dipole_moments(basis: FEDVR_Basis, psi: np.ndarray) -> np.ndarray:
    """
    Compute dipole moments between all states:
        D(l n, l' n') = ∫ r * psi_{n, l}(r) * psi_{n', l'}(r) dr
    """
    lmax = psi.shape[0] - 1
    nmax = psi.shape[1] - 1

    r_integs = radial_integrals(basis, psi, r_pow=1)


    Indices = []
    for l in range(lmax + 1):
        for n in range(nmax + 1):
            for m in range(-l, l + 1):
                Indices.append( (l, n, m) )
    
    Indices = np.array(Indices)
    norbs_tot = Indices.shape[0]

    D_matrix = np.zeros((3, norbs_tot, norbs_tot), dtype=np.complex128)

    for i in range(norbs_tot):
        for j in range(norbs_tot):
            l1, n1, m1 = Indices[i]
            l2, n2, m2 = Indices[j]

            if abs(l1 - l2) != 1:
                continue

            s = minus_one_pow(m1)

            gaunt_m1 = float(gaunt(l1, 1, l2, -m1, -1, m2)) * s
            gaunt_0 = float(gaunt(l1, 1, l2, -m1, 0, m2)) * s
            gaunt_p1 = float(gaunt(l1, 1, l2, -m1, 1, m2)) * s
            D_matrix[0, i, j] = (gaunt_m1 - gaunt_p1) * r_integs[l1, n1, l2, n2] / np.sqrt(2)
            D_matrix[1, i, j] = 1j * (gaunt_m1 + gaunt_p1) * r_integs[l1, n1, l2, n2] / np.sqrt(2)
            D_matrix[2, i, j] = gaunt_0 * r_integs[l1, n1, l2, n2]

    return Indices, D_matrix
#=================================================================
def save_dipole_moments(filename: str, Indices: np.ndarray, D_matrix: np.ndarray) -> None:
    """
    Save dipole moment matrix to an HDF5 file.
    """
    with h5py.File(filename, 'w') as f:
        f.create_dataset('Indices', data=Indices)
        f.create_dataset('real_part', data=D_matrix.real)
        f.create_dataset('imag_part', data=D_matrix.imag)
#=================================================================