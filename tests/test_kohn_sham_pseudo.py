"""
Unit tests for solve_schrodinger_pseudo.

Three self-contained tests using a hydrogen-like potential (V = -Z/r) so that
exact reference eigenvalues are known analytically: ε(n) = -Z²/(2n²) Ha.

Test 1 — zero non-local:
    Dion = 0  →  solver reduces to local; eigenvalues must match hydrogen.

Test 2 — rank-1 projector aligned with an eigenstate:
    β = ψ₀ (the DVR 1s state).  Because β is an exact eigenstate of H₀,
    adding D|β⟩⟨β| shifts ε₀ by exactly D·‖β‖²_DVR and leaves all other
    eigenvalues untouched.  This is exact, not perturbative.

Test 3 — l-channel isolation:
    Non-local projector added only in l=1 must not alter l=0 eigenvalues.
"""

import numpy as np
import pytest

from atomic_femdvr.femdvr import FEDVR_Basis
from atomic_femdvr.kohn_sham import solve_schrodinger_pseudo, solve_schrodinger_local


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_basis(Rmax: float = 40.0, ne: int = 40, ng: int = 8) -> FEDVR_Basis:
    xp = list(np.linspace(0.0, Rmax, ne + 1))
    return FEDVR_Basis(ne, ng, xp)


def _make_log_basis(r0: float = 0.01, Rmax: float = 40.0,
                    ne: int = 30, ng: int = 8) -> FEDVR_Basis:
    """Logarithmically spaced element boundaries, dense near the nucleus."""
    xp = [0.0] + list(np.geomspace(r0, Rmax, ne))
    return FEDVR_Basis(ne, ng, xp)


def _hydrogen_veff(r_grid: np.ndarray, Z: float = 1.0) -> np.ndarray:
    """V_eff = -Z/r; regularise the r=0 point by continuity."""
    V = np.empty_like(r_grid)
    V[1:] = -Z / r_grid[1:]
    V[0] = V[1]          # r[0]=0 is the Dirichlet node; its value is unused
    return V


def _dvr_norm_sq(basis: FEDVR_Basis, psi_grid: np.ndarray) -> float:
    """‖ψ‖² in the DVR inner product, i.e. b·b where b = get_coeffs(psi)."""
    b = basis.get_coeffs(psi_grid, cplx=False)
    return float(b @ b)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hydrogen_setup():
    """Shared basis + hydrogen potential used by all tests."""
    basis = _make_basis()
    r_grid = basis.get_gridpoints()
    V = _hydrogen_veff(r_grid)

    # Dummy projector arrays (l=0 channel only, one projector)
    lll = np.array([0], dtype=int)
    Dion_zero = np.array([[0.0]])
    beta_dummy = np.zeros((1, len(r_grid)))

    return basis, r_grid, V, lll, Dion_zero, beta_dummy


# ── tests ─────────────────────────────────────────────────────────────────────

class TestZeroNonLocal:
    """Dion=0: pseudo solver must reproduce the local hydrogen eigenvalues."""

    def test_l0_eigenvalues(self, hydrogen_setup):
        basis, r_grid, V, lll, Dion_zero, beta_dummy = hydrogen_setup

        eps_pseudo, _ = solve_schrodinger_pseudo(
            basis, V, lll, Dion_zero, beta_dummy, lmax=0, nmax=3
        )
        eps_local, _ = solve_schrodinger_local(basis, V, lmax=0, nmax=3)

        np.testing.assert_allclose(eps_pseudo[0], eps_local[0], rtol=1e-10,
                                   err_msg="Dion=0 must reproduce the local eigenvalues")

    def test_hydrogen_1s_energy(self, hydrogen_setup):
        """1s eigenvalue should be -0.5 Ha to within grid accuracy."""
        basis, r_grid, V, lll, Dion_zero, beta_dummy = hydrogen_setup

        eps, _ = solve_schrodinger_pseudo(
            basis, V, lll, Dion_zero, beta_dummy, lmax=0, nmax=2
        )
        assert pytest.approx(eps[0, 0], rel=1e-4) == -0.5


class TestRank1Projector:
    """
    β = ψ₀ (DVR 1s state of H).  Adding D|β⟩⟨β| shifts ε₀ by exactly
    D·‖β‖²_DVR and leaves all other states untouched.
    """

    @pytest.fixture(scope="class")
    def local_gs(self, hydrogen_setup):
        """Solve the local problem once; return ε and ψ for l=0."""
        basis, r_grid, V, *_ = hydrogen_setup
        eps, psi = solve_schrodinger_local(basis, V, lmax=0, nmax=3)
        return basis, r_grid, V, eps[0], psi[0]   # l=0 channel

    def test_ground_state_shift(self, hydrogen_setup, local_gs):
        basis, r_grid, V, lll, *_ = hydrogen_setup
        _, _, _, eps0, psi0 = local_gs

        D_val_Ha = 0.25   # desired shift in Hartree

        # β = ψ₀ (DVR normalised ground state); c^T c = ∫ψ₀²dr = 1
        beta_grid = psi0[0:1, :]           # shape (1, ngrid)
        norm_sq = _dvr_norm_sq(basis, psi0[0, :])  # should be 1.0

        # Dion is stored in Rydberg units in UPF files; the solver multiplies by 0.5
        # to convert to Hartree.  Pass 2*D_val_Ha so the effective D is D_val_Ha.
        Dion_Ry = np.array([[2.0 * D_val_Ha]])

        eps_nl, _ = solve_schrodinger_pseudo(
            basis, V, lll, Dion_Ry, beta_grid, lmax=0, nmax=3
        )

        # Exact (non-perturbative): β ∝ ψ₀ is an eigenstate of H₀, so the shift
        # is D_val_Ha * ‖β‖²_DVR = D_val_Ha (since norm_sq = 1).
        expected_gs = eps0[0] + D_val_Ha * norm_sq
        assert pytest.approx(eps_nl[0, 0], rel=1e-8) == expected_gs

    def test_excited_states_unchanged(self, hydrogen_setup, local_gs):
        """Excited states are orthogonal to β=ψ₀, so must not shift."""
        basis, r_grid, V, lll, *_ = hydrogen_setup
        _, _, _, eps0, psi0 = local_gs

        D_val_Ha = 0.25
        beta_grid = psi0[0:1, :]
        Dion_Ry = np.array([[2.0 * D_val_Ha]])

        eps_nl, _ = solve_schrodinger_pseudo(
            basis, V, lll, Dion_Ry, beta_grid, lmax=0, nmax=3
        )

        for n in range(1, 4):
            assert pytest.approx(eps_nl[0, n], rel=1e-8) == eps0[n], (
                f"Excited state n={n} must not shift"
            )


class TestChannelIsolation:
    """Non-local projector in l=1 must not affect l=0 eigenvalues."""

    def test_l0_unaffected_by_l1_projector(self, hydrogen_setup):
        basis, r_grid, V, *_ = hydrogen_setup

        eps_local, _ = solve_schrodinger_local(basis, V, lmax=1, nmax=2)

        # Build a large l=1 projector using the 2p local eigenstate
        psi_2p = eps_local   # we only need the shape; let's re-solve cleanly
        eps_l, psi_l = solve_schrodinger_local(basis, V, lmax=1, nmax=2)

        lll = np.array([1], dtype=int)
        D_val_Ha = 1.0
        Dion_Ry = np.array([[2.0 * D_val_Ha]])
        beta_grid = psi_l[1, 0:1, :]   # 2p ground state as projector, shape (1, ngrid)

        eps_pseudo, _ = solve_schrodinger_pseudo(
            basis, V, lll, Dion_Ry, beta_grid, lmax=1, nmax=2
        )

        # l=0 channel (index 0) must be identical to local
        np.testing.assert_allclose(eps_pseudo[0], eps_l[0], rtol=1e-10,
                                   err_msg="l=1 projector must not affect l=0 channel")


class TestLogGridBasis:
    """
    Verify get_coeffs / get_psi on a non-uniform logarithmic grid.

    A log grid has elements that vary in width by orders of magnitude —
    the smallest element is near r=0 (width ~ r0), the largest is near Rmax.
    The weight in each element scales with its half-width, so this stresses
    the per-element weight computation in get_coeffs / get_psi.
    """

    @pytest.fixture(scope="class")
    def log_basis(self):
        return _make_log_basis()

    def test_get_psi_get_coeffs_inverse(self, log_basis):
        """get_coeffs(get_psi(v)) == v for a random coefficient vector."""
        basis = log_basis
        nb = basis.ne * basis.ng - 1
        rng = np.random.default_rng(0)
        v = rng.standard_normal(nb)
        v /= np.linalg.norm(v)

        v_back = basis.get_coeffs(basis.get_psi(v, cplx=False), cplx=False)
        np.testing.assert_allclose(v_back, v, atol=1e-14,
                                   err_msg="get_coeffs must invert get_psi on a log grid")

    def test_normalisation_invariant(self, log_basis):
        """c^T c = ∫ψ² dr holds on a non-uniform grid."""
        basis = log_basis
        nb = basis.ne * basis.ng - 1
        rng = np.random.default_rng(1)
        v = rng.standard_normal(nb)
        v /= np.linalg.norm(v)   # c^T c = 1 by construction

        psi = basis.get_psi(v, cplx=False)

        # DVR quadrature integral
        w_i = basis.leg.w_i
        quad_w = np.zeros(basis.ne * basis.ng + 1)
        for ie in range(basis.ne):
            h_e = 0.5 * (basis.xp[ie + 1] - basis.xp[ie])
            quad_w[ie * basis.ng : ie * basis.ng + basis.ng + 1] += h_e * w_i

        integral = np.dot(quad_w, psi**2)
        np.testing.assert_allclose(integral, 1.0, atol=1e-14,
                                   err_msg="c^T c = ∫ψ² dr must hold on a log grid")

    def test_hydrogen_1s_log_grid(self, log_basis):
        """Log grid should give better 1s accuracy than uniform (denser near nucleus)."""
        basis = log_basis
        r_grid = basis.get_gridpoints()
        V = _hydrogen_veff(r_grid)

        eps, _ = solve_schrodinger_local(basis, V, lmax=0, nmax=3)

        assert pytest.approx(eps[0, 0], rel=1e-6) == -0.5
