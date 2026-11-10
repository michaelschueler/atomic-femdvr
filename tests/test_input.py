"""Tests for the input models."""

import pytest
from pydantic import ValidationError

from atomic_femdvr.input import ConfinementInput, ConfinementType, DFTInput


def test_diis_is_anderson() -> None:
    """'diis' and the old 'diis_history' key map to Anderson mixing."""
    dft = DFTInput(mixing_scheme="DIIS", diis_history=3)
    assert dft.mixing_scheme == "anderson"
    assert dft.mixing_history == 3


def test_confinement_null() -> None:
    """Explicit nulls in JSON are accepted and mean 'no confinement'."""
    conf = ConfinementInput(type=None, polarization_mode=None)
    assert conf.type == ConfinementType.NONE
    assert conf.polarization_mode is None


def test_unknown_keys_rejected() -> None:
    """Misspelled keys raise instead of being silently ignored."""
    with pytest.raises(ValidationError):
        DFTInput(alpha_mixing=0.3)
