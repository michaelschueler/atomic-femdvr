"""Tests for the `atomic_femdvr.pseudo_atomic` module."""

import pytest

from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, solve_pseudo_atomic


@pytest.mark.skip(reason="Waiting for upf-independent implementation")
def test_scf(molybdenum_input_dict):
    """Test the pseudo-atomic solver for molybdenum using only SCF."""
    inp = PseudoAtomicInput(**molybdenum_input_dict)

    eigenvalues = solve_pseudo_atomic(inp, task_list=('scf',))

    assert 'scf' in eigenvalues

    benchmark_eigenvalues = {"0": [-2.3767770200534524, -0.1603849407697104],
                             "1": [-1.434869395676964, -0.04179242855203131],
                             "2": [-0.1653870399482107]}

    for angular_momentum in benchmark_eigenvalues:
        assert angular_momentum in eigenvalues['scf']
        for ev, bev in zip(eigenvalues['scf'][angular_momentum],
                           benchmark_eigenvalues[angular_momentum],
                           strict=True):
            assert pytest.approx(ev, rel=1.0e-7) == bev




