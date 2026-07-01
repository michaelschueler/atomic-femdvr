"""Tests for :class:`atomic_femdvr.upf.UPFInterface`."""

from atomic_femdvr.upf import UPFInterface


def test_load_mo_upf(data_directory, num_regression):
    """Loading the bundled Mo UPF: regression-test header values + sampled arrays."""
    upf = UPFInterface.from_upf(data_directory / "Mo/Mo.upf")

    # Sanity checks on shape / type
    assert upf.r.shape == (upf.mesh,)
    assert upf.beta.shape == (upf.mesh, upf.nbeta)
    assert upf.dion.shape == (upf.nbeta, upf.nbeta)
    assert upf.chi.shape == (upf.mesh, upf.nwfc)

    # Sampled arrays for regression
    flat = {
        "r_sample": upf.r[::100].tolist(),
        "vloc_sample": upf.vloc[::100].tolist(),
        "rho_atom_sample": (upf.rho_atom[::100].tolist() if upf.rho_atom is not None else [0.0]),
    }
    num_regression.check(flat, default_tolerance={"rtol": 1e-12})


def test_get_charge_density(data_directory):
    """``get_charge_density`` returns the UPF rho_atom verbatim when present."""
    upf = UPFInterface.from_upf(data_directory / "Mo/Mo.upf")
    rho = upf.get_charge_density()
    assert rho.shape == (upf.mesh,)
    if upf.rho_atom is not None:
        # rho_atom should be returned verbatim
        assert rho[0] == upf.rho_atom[0]
