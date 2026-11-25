"""Analytical reference solutions for quantum-mechanical tests."""

import numpy as np

from quantumforge.physics.constants import SI_CONSTANTS
from quantumforge.physics.wavefunction import WaveFunction


def free_gaussian_wavefunction(
    x: np.ndarray,
    time: float,
    mass: float,
    sigma: float,
    hbar: float = SI_CONSTANTS.hbar,
    initial_center: float = 0.0,
) -> WaveFunction:
    """Return the analytical wavefunction of a freely spreading Gaussian."""
    if mass <= 0.0 or sigma <= 0.0 or hbar <= 0.0:
        raise ValueError("Mass, sigma, and hbar must be positive.")

    tau = 2.0 * mass * sigma**2 / hbar
    spreading = 1.0 + 1j * time / tau

    psi = (
        (2.0 * np.pi * sigma**2) ** (-0.25)
        / np.sqrt(spreading)
        * np.exp(
            -(x - initial_center) ** 2
            / (4.0 * sigma**2 * spreading)
        )
    )

    return WaveFunction(x=x, psi=psi)