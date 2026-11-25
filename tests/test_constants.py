import pytest

from quantumforge.physics.constants import SI_CONSTANTS, PhysicalConstants


def test_default_constants_are_positive():
    assert SI_CONSTANTS.hbar > 0
    assert SI_CONSTANTS.electron_mass > 0
    assert SI_CONSTANTS.proton_mass > 0


def test_proton_is_heavier_than_electron():
    assert SI_CONSTANTS.proton_mass > SI_CONSTANTS.electron_mass


def test_custom_physical_constants():
    constants = PhysicalConstants(
        hbar=1.0,
        electron_mass=2.0,
        proton_mass=3.0,
    )

    assert constants.hbar == 1.0
    assert constants.electron_mass == 2.0
    assert constants.proton_mass == 3.0


def test_physical_constants_are_immutable():
    with pytest.raises(AttributeError):
        SI_CONSTANTS.hbar = 1.0