"""Tests for :class:`atomic_femdvr.femdvr.FEDVR_Basis`."""

import numpy as np
import pytest

from atomic_femdvr.femdvr import FEDVR_Basis


@pytest.fixture
def small_basis() -> FEDVR_Basis:
    """Five-element basis on [0, 5] with 8 Lobatto nodes per element."""
    xp = [0.0, 0.5, 1.5, 2.5, 4.0, 5.0]
    return FEDVR_Basis(ne=len(xp) - 1, ng=8, xp=xp, build_derivatives=False)


def test_get_gridpoints_shape(small_basis):
    """Grid has the expected length and endpoints."""
    g = small_basis.get_gridpoints()
    assert g.shape == (small_basis.ne * small_basis.ng + 1,)
    assert g[0] == pytest.approx(small_basis.xp[0])
    assert g[-1] == pytest.approx(small_basis.xp[-1])


def _vanishing_psi(grid: np.ndarray) -> np.ndarray:
    """Smooth radial function that vanishes at both endpoints (Dirichlet BC)."""
    r0, r1 = grid[0], grid[-1]
    return (grid - r0) * (r1 - grid) * np.exp(-0.3 * grid)


def test_psi_coeffs_round_trip(small_basis):
    """get_psi(get_coeffs(psi)) reproduces psi for endpoint-vanishing inputs."""
    grid = small_basis.get_gridpoints()
    psi = _vanishing_psi(grid)
    cff = small_basis.get_coeffs(psi)
    psi_back = small_basis.get_psi(cff)
    np.testing.assert_allclose(psi_back, psi, atol=1e-12)


def test_overlap_self_norm(small_basis):
    """get_overlap matches a high-resolution Simpson integral."""
    grid = small_basis.get_gridpoints()
    psi = _vanishing_psi(grid)
    ovlp = small_basis.get_overlap(psi, psi)

    # Reference: dense linspace + trapezoid of the analytic function on the
    # same interval (interpolation is exact within each element, so this is
    # consistent with the basis quadrature).
    fine = np.linspace(grid[0], grid[-1], 5001)
    ref = np.trapezoid(_vanishing_psi(fine) ** 2, fine)
    assert ovlp == pytest.approx(ref, rel=1e-4)


def test_interpolate_at_gridpoints_is_identity(small_basis):
    """Interpolating at the basis gridpoints reproduces the input values."""
    grid = small_basis.get_gridpoints()
    # interior points only — interpolate() uses [xp[i], xp[i+1]) half-open
    # ranges, so the very last gridpoint isn't covered (returns 0).
    interior = grid[:-1]
    psi = _vanishing_psi(grid)
    interp = small_basis.interpolate(psi, interior)
    np.testing.assert_allclose(interp, psi[:-1], atol=1e-12)


def test_interpolate_outside_returns_zero(small_basis):
    """Points outside [xp[0], xp[-1]] interpolate to zero."""
    psi = np.ones(small_basis.ne * small_basis.ng + 1)
    out = small_basis.interpolate(psi, np.array([small_basis.xp[-1] + 1.0]))
    assert out[0] == 0.0
