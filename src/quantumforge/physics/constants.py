"""Fundamental physical constants used in QuantumForge."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants expressed in SI units."""

    hbar: float = 1.054571817e-34
    electron_mass: float = 9.1093837139e-31
    proton_mass: float = 1.67262192595e-27


SI_CONSTANTS = PhysicalConstants()
