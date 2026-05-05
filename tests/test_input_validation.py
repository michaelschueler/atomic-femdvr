"""Validation tests for the pydantic input models in :mod:`atomic_femdvr.input`."""

import pytest
from pydantic import ValidationError

from atomic_femdvr.input import (
    ConfinementInput,
    ConfinementType,
    DFTInput,
    EnergyUnit,
    SysParamsInput,
)


def test_sysparams_requires_element():
    """SysParamsInput rejects None / missing element."""
    with pytest.raises(ValidationError, match="Element must be specified"):
        SysParamsInput(element=None)


def test_sysparams_rejects_overlong_element():
    """Element symbols longer than 2 characters are rejected."""
    with pytest.raises(ValidationError, match="one or two characters"):
        SysParamsInput(element="Mooo")


def test_sysparams_rejects_empty_element():
    """Empty / whitespace-only element strings are rejected."""
    with pytest.raises(ValidationError, match="non-empty string"):
        SysParamsInput(element="   ")


def test_sysparams_normalises_element_capitalisation():
    """Element symbols are stripped and capitalised."""
    sp = SysParamsInput(element="  mo  ")
    assert sp.element == "Mo"


def test_sysparams_extra_fields_forbidden():
    """Unknown keys raise (typo protection)."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SysParamsInput(element="Mo", flmax=3)  # typo: flmax instead of lmax


def test_pot_energy_unit_normalises():
    """The pot_energy_unit alias map collapses synonyms."""
    sp = SysParamsInput(element="Mo", pot_energy_unit="rydberg")
    assert sp.pot_energy_unit == EnergyUnit.RYDBERG
    sp = SysParamsInput(element="Mo", pot_energy_unit="hartree")
    assert sp.pot_energy_unit == EnergyUnit.HARTREE


def test_pot_energy_unit_rejects_unknown():
    """Unknown energy-unit strings raise a ValidationError."""
    with pytest.raises(ValidationError, match="Invalid value for pot_energy_unit"):
        SysParamsInput(element="Mo", pot_energy_unit="bohr")


def test_confinement_type_lowercased():
    """ConfinementType strings are lower-cased before parsing."""
    c = ConfinementInput(type="SoftStep")
    assert c.type == ConfinementType.SOFTSTEP


def test_confinement_ri_factor_range():
    """ri_factor is constrained to [0, 1]."""
    with pytest.raises(ValidationError):
        ConfinementInput(ri_factor=1.5)
    with pytest.raises(ValidationError):
        ConfinementInput(ri_factor=-0.1)


def test_dft_mixing_scheme_lowercased():
    """mixing_scheme is normalised to lower case."""
    dft = DFTInput(mixing_scheme="Anderson")
    assert dft.mixing_scheme == "anderson"


def test_dft_mixing_scheme_rejects_unknown():
    """An unknown mixing scheme is rejected."""
    with pytest.raises(ValidationError):
        DFTInput(mixing_scheme="newton")
