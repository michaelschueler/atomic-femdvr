"""Tests for :func:`atomic_femdvr.bessel_transform.bessel_integral`."""

import numpy as np
import pytest

from atomic_femdvr.bessel_transform import bessel_integral
from atomic_femdvr.femdvr import FEDVR_Basis


@pytest.fixture
def basis() -> FEDVR_Basis:
    """Eight-element basis on [0, 8] with 10 Lobatto nodes per element."""
    xp = np.linspace(0.0, 8.0, 9).tolist()
    return FEDVR_Basis(ne=len(xp) - 1, ng=10, xp=xp, build_derivatives=False)


def test_bessel_simpson_matches_lobatto(basis):
    """Simpson and Lobatto quadratures agree to high precision on a smooth input."""
    grid = basis.get_gridpoints()
    phi = grid * np.exp(-(grid**2))  # u(r) shape, vanishes at endpoints
    qgrid = np.linspace(0.05, 5.0, 32)

    s = bessel_integral(basis, l=0, rpow=1, qgrid=qgrid, phi=phi, method="simpson")
    lo = bessel_integral(basis, l=0, rpow=1, qgrid=qgrid, phi=phi, method="lobatto")
    np.testing.assert_allclose(s, lo, rtol=1e-3)


def test_bessel_unknown_method_raises(basis):
    """An unsupported quadrature method raises ValueError."""
    grid = basis.get_gridpoints()
    phi = grid * np.exp(-(grid**2))
    with pytest.raises(ValueError, match="Unknown integration method"):
        bessel_integral(basis, l=0, rpow=1, qgrid=np.array([1.0]), phi=phi, method="bogus")


def test_bessel_regression(basis, num_regression):
    """Spherical Bessel transform of a Gaussian is regression-tested."""
    grid = basis.get_gridpoints()
    phi = grid * np.exp(-(grid**2))
    qgrid = np.linspace(0.05, 5.0, 16)

    transform = bessel_integral(basis, l=0, rpow=1, qgrid=qgrid, phi=phi)
    num_regression.check(
        {"q": qgrid, "transform": transform},
        default_tolerance={"rtol": 1e-10},
    )
