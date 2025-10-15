"""Tests for the `atomic_femdvr.atomic` module."""

import pytest
from pathlib import Path
from atomic_femdvr.atomic import AtomicInput, solve_atomic

@pytest.mark.skip(reason="Waiting for upf-independent implementation")
def test_atomic(silicon_input_dict: AtomicInput):
    """Test the atomic solver for molybdenum."""

    inp = AtomicInput(**silicon_input_dict)

    eigenvalues, _, _ = solve_atomic(inp)

    benchmark_eigenvalues = {'0': [-0.9150214513795011, -0.06570263840413143],
                             '1': [-0.22230200068177003],
                             '2': []}
    
    for l in benchmark_eigenvalues:
        for ev, bev in zip(eigenvalues[l], benchmark_eigenvalues[l], strict=True):
            assert pytest.approx(ev, rel=1.0e-7) == bev




