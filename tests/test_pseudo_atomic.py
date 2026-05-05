"""Tests for the `atomic_femdvr.pseudo_atomic` module."""

from atomic_femdvr.pseudo_atomic import (
    PseudoAtomicInput,
    PseudoAtomicResult,
    solve_pseudo_atomic,
)


def _result_as_dict(result: PseudoAtomicResult) -> dict:
    """Cast a PseudoAtomicResult to a regression-friendly dict."""
    return result.model_dump()


def test_scf(molybdenum_input_dict, num_regression):
    """SCF on molybdenum: regression-test the bound-state eigenvalues."""
    inp = PseudoAtomicInput(**molybdenum_input_dict)
    result = solve_pseudo_atomic(inp, task_list=("scf",))

    assert "scf" in result.eigenvalues
    assert result.energy_shifts is None  # no nscf yet

    flat = {f"scf_l{l}": vals for l, vals in result.eigenvalues["scf"].items()}
    num_regression.check(flat, default_tolerance={"rtol": 1e-7})


def test_scf_nscf(molybdenum_input_dict, num_regression):
    """SCF + non-SCF on molybdenum: eigenvalues per task and per-l energy shifts."""
    inp = PseudoAtomicInput(**molybdenum_input_dict)
    result = solve_pseudo_atomic(inp, task_list=("scf", "nscf"))

    assert "scf" in result.eigenvalues
    assert "nscf" in result.eigenvalues
    assert result.energy_shifts is not None

    flat: dict[str, list[float]] = {}
    for task, by_l in result.eigenvalues.items():
        for l, vals in by_l.items():
            flat[f"{task}_l{l}"] = vals
    for l, vals in result.energy_shifts.items():
        flat[f"shift_l{l}"] = vals
    num_regression.check(flat, default_tolerance={"rtol": 1e-7})
