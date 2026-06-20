"""
Photoemission matrix elements over an energy grid.

The scattering phase shift delta_l is a property of the final state only — it depends
on the angular momentum l_f and kinetic energy E_k, not on which occupied shell the
electron came from. Accordingly, the computation is structured as:

  for each unique l_f:
      for each E_k:
          solve scattering WF  (one solve, shared by all coupled initial shells)
          extract delta_{l_f}(E_k)
          for each initial shell (l_i, n_i) with l_i = l_f ± 1:
              compute radial matrix element

The results dict keeps phase shifts keyed by l_f and matrix elements keyed by
(l_i, n_i, l_f).

Radial matrix elements (psi_f = scattering, psi_i = bound; both are reduced WFs u = r*psi):

  Length gauge:       M_L  = ∫ u_f*(r) * r * u_i(r) dr
  Velocity gauge 1/r: M_Vr = ∫ u_f*(r) * (1/r) * u_i(r) dr
  Velocity gauge d/dr:M_Vd = ∫ u_f*(r) * (d/dr) u_i(r) dr   [in DVR coefficient space]

Combined velocity-gauge radial ME: M_Vd - M_Vr
(accounts for d/dr acting on the full WF phi = u/r: d/dr(u/r) = u'/r - u/r²)

Integration uses DVR quadrature; bridge points accumulate weights from both elements.
"""

import numpy as np
import h5py

from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.scattering import solve_scattering_local, solve_scattering_nonlocal, extract_phase


def _build_weight_grid(basis: FEDVR_Basis) -> np.ndarray:
    """
    Build the DVR quadrature weight grid (length ne*ng+1).

    Bridge points (shared element boundaries) accumulate contributions from
    both adjacent elements via in-place addition.
    """
    ne, ng, xp = basis.ne, basis.ng, basis.xp
    w_i = basis.leg.w_i

    w_grid = np.zeros(ne * ng + 1)
    for ie in range(ne):
        h_e = 0.5 * (xp[ie + 1] - xp[ie])
        for m in range(ng + 1):
            w_grid[ie * ng + m] += h_e * w_i[m]

    return w_grid


def _radial_matrix_elements(basis: FEDVR_Basis, psi_f: np.ndarray, psi_i: np.ndarray,
                             r_grid: np.ndarray, w_grid: np.ndarray,
                             gauges: list[str]) -> dict[str, complex]:
    """
    Compute radial photoemission matrix elements for one (psi_f, psi_i) pair.

    Returns a dict with keys from {'length', 'velocity_r', 'velocity_d'} depending
    on the requested gauges.
    """
    result = {}

    if 'length' in gauges:
        result['length'] = np.dot(w_grid, psi_f.conj() * r_grid * psi_i)

    if 'velocity' in gauges:
        inv_r = np.zeros_like(r_grid)
        inv_r[1:] = 1.0 / r_grid[1:]  # psi_i[0] = 0 so inv_r[0] value is irrelevant
        result['velocity_r'] = np.dot(w_grid, psi_f.conj() * inv_r * psi_i)

        D = basis.get_deriv_matrix()
        cff_i = basis.get_coeffs(psi_i.real if np.isrealobj(psi_i) else psi_i.real, cplx=False)
        cff_f = basis.get_coeffs(psi_f, cplx=True)
        result['velocity_d'] = np.dot(cff_f.conj(), D @ cff_i)

    return result


def compute_photoemission(
    basis: FEDVR_Basis,
    Veff_grid: np.ndarray,
    eps: np.ndarray,
    psi_bound: np.ndarray,
    nrad: np.ndarray,
    ll: np.ndarray,
    occ: np.ndarray,
    energies: np.ndarray,
    gauges: list[str],
    store_wavefunctions: bool = False,
    output_prefix: str = 'photoemission',
    lll: np.ndarray | None = None,
    Dion: np.ndarray | None = None,
    beta_pp: np.ndarray | None = None,
) -> dict:
    """
    Compute photoemission matrix elements and scattering phase shifts over an energy grid.

    Phase shifts are computed once per (l_f, E_k) and stored keyed by l_f.
    Matrix elements are computed per transition (l_i, n_i) -> l_f.

    Parameters
    ----------
    basis : FEDVR_Basis
    Veff_grid : np.ndarray
        Self-consistent KS effective potential on the FEM-DVR grid (Hartree).
    eps : np.ndarray, shape (lmax+1, nmax+1)
        Bound-state eigenvalues in Hartree. Bound states have eps < 0.
    psi_bound : np.ndarray, shape (lmax+1, nmax+1, ngrid)
        Reduced radial wavefunctions of bound states.
    nrad : np.ndarray
        Radial quantum numbers of occupied shells (n - l - 1).
    ll : np.ndarray
        Angular momenta of occupied shells.
    occ : np.ndarray
        Occupations of occupied shells.
    energies : np.ndarray
        Kinetic energies of the photoelectron in Hartree.
    gauges : list[str]
        Which gauges to compute. Subset of ['length', 'velocity'].
    store_wavefunctions : bool
        If True, save scattering wavefunctions to HDF5, one group per l_f.
    output_prefix : str
        Path prefix for output files.
    lll, Dion, beta_pp : optional
        Non-local PP projectors. If provided, uses the non-local scattering solver.

    Returns
    -------
    results : dict
        'energies'      : np.ndarray (nE,)
        'shells'        : list of (l_i, n_i) for occupied shells
        'eps_bound'     : eigenvalue for each shell (Ha)
        'phase_shifts'  : dict  l_f -> np.ndarray complex (nE,)
        'delta'         : dict  l_f -> np.ndarray real (nE,)
        'me_length'     : dict  (l_i, n_i, l_f) -> np.ndarray complex (nE,)
        'me_velocity_r' : dict  (l_i, n_i, l_f) -> np.ndarray complex (nE,)
        'me_velocity_d' : dict  (l_i, n_i, l_f) -> np.ndarray complex (nE,)
    """
    r_grid = basis.get_gridpoints()
    w_grid = _build_weight_grid(basis)
    use_nonlocal = (lll is not None) and (Dion is not None) and (beta_pp is not None)
    nE = len(energies)

    occupied_shells = []
    for ishell in range(len(ll)):
        l_i = ll[ishell]
        n_i = nrad[ishell]
        if occ[ishell] > 0.0 and eps[l_i, n_i] < 0.0:
            occupied_shells.append((l_i, n_i))

    # All unique final-state angular momenta required by the dipole selection rule
    unique_lf = sorted({l_i + dl
                        for (l_i, _) in occupied_shells
                        for dl in (-1, +1)
                        if l_i + dl >= 0})

    results = {
        'energies': energies,
        'shells': occupied_shells,
        'eps_bound': [eps[l_i, n_i] for (l_i, n_i) in occupied_shells],
        'phase_shifts': {},
        'delta': {},
        'me_length': {},
        'me_velocity_r': {},
        'me_velocity_d': {},
    }

    wfc_file = None
    if store_wavefunctions:
        wfc_path = output_prefix + '_wavefunctions.h5'
        wfc_file = h5py.File(wfc_path, 'w')
        wfc_file.create_dataset('r_grid', data=r_grid)
        wfc_file.create_dataset('energies', data=energies)

    for l_f in unique_lf:
        # Initial shells coupled to this l_f via dipole selection rule
        coupled = [(l_i, n_i) for (l_i, n_i) in occupied_shells
                   if abs(l_i - l_f) == 1]

        S_arr = np.zeros(nE, dtype=np.complex128)
        delta_arr = np.zeros(nE, dtype=np.float64)

        # Accumulate matrix elements per coupled shell
        me_L  = {s: np.zeros(nE, dtype=np.complex128) for s in coupled}
        me_Vr = {s: np.zeros(nE, dtype=np.complex128) for s in coupled}
        me_Vd = {s: np.zeros(nE, dtype=np.complex128) for s in coupled}

        wfc_grp = wfc_file.require_group(f'lf{l_f}') if wfc_file is not None else None

        for iE, Ek in enumerate(energies):
            k = np.sqrt(2.0 * Ek)

            # One scattering solve for this (l_f, E_k) — shared by all coupled shells
            if use_nonlocal:
                psi_f = solve_scattering_nonlocal(basis, Veff_grid, k, l_f,
                                                  lll, Dion, beta_pp)
            else:
                psi_f = solve_scattering_local(basis, Veff_grid, k, l_f)

            S_l, delta_l = extract_phase(psi_f, r_grid, k, l_f)
            S_arr[iE] = S_l
            delta_arr[iE] = delta_l

            for (l_i, n_i) in coupled:
                psi_i = psi_bound[l_i, n_i, :]
                me = _radial_matrix_elements(basis, psi_f, psi_i, r_grid, w_grid, gauges)
                if 'length' in gauges:
                    me_L[(l_i, n_i)][iE] = me['length']
                if 'velocity' in gauges:
                    me_Vr[(l_i, n_i)][iE] = me['velocity_r']
                    me_Vd[(l_i, n_i)][iE] = me['velocity_d']

            if wfc_grp is not None:
                wfc_grp.create_dataset(f'iE{iE:04d}_real', data=psi_f.real)
                wfc_grp.create_dataset(f'iE{iE:04d}_imag', data=psi_f.imag)

        results['phase_shifts'][l_f] = S_arr
        results['delta'][l_f] = delta_arr
        for (l_i, n_i) in coupled:
            key = (l_i, n_i, l_f)
            if 'length' in gauges:
                results['me_length'][key] = me_L[(l_i, n_i)]
            if 'velocity' in gauges:
                results['me_velocity_r'][key] = me_Vr[(l_i, n_i)]
                results['me_velocity_d'][key] = me_Vd[(l_i, n_i)]

    if wfc_file is not None:
        wfc_file.close()

    return results


def save_photoemission(results: dict, filename: str, gauges: list[str]) -> None:
    """
    Save photoemission results to an HDF5 file.

    Structure:
        energies, eps_bound, shells
        phase_shifts/lf{l_f}/  S_l_real, S_l_imag, delta
        transitions/l{l_i}_n{n_i}_lf{l_f}/  me_length_*, me_velocity_*
    """
    with h5py.File(filename, 'w') as f:
        f.create_dataset('energies', data=results['energies'])
        f.create_dataset('eps_bound', data=np.array(results['eps_bound']))
        f.create_dataset('shells', data=np.array(results['shells'], dtype=int))

        ps_grp = f.create_group('phase_shifts')
        for l_f, S_arr in results['phase_shifts'].items():
            g = ps_grp.create_group(f'lf{l_f}')
            g.create_dataset('S_l_real', data=S_arr.real)
            g.create_dataset('S_l_imag', data=S_arr.imag)
            g.create_dataset('delta', data=results['delta'][l_f])

        tr_grp = f.create_group('transitions')
        all_keys = (set(results.get('me_length', {}))
                    | set(results.get('me_velocity_r', {})))
        for key in sorted(all_keys):
            l_i, n_i, l_f = key
            g = tr_grp.create_group(f'l{l_i}_n{n_i}_lf{l_f}')
            if 'length' in gauges and key in results.get('me_length', {}):
                me = results['me_length'][key]
                g.create_dataset('me_length_real', data=me.real)
                g.create_dataset('me_length_imag', data=me.imag)
            if 'velocity' in gauges and key in results.get('me_velocity_r', {}):
                me_r = results['me_velocity_r'][key]
                me_d = results['me_velocity_d'][key]
                g.create_dataset('me_velocity_r_real', data=me_r.real)
                g.create_dataset('me_velocity_r_imag', data=me_r.imag)
                g.create_dataset('me_velocity_d_real', data=me_d.real)
                g.create_dataset('me_velocity_d_imag', data=me_d.imag)

    print(f"Photoemission results saved to {filename}")
