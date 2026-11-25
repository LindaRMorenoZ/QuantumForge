
"""Quantum potential calculations in arbitrary spatial dimensions."""

import numpy as np

from quantumforge.physics.constants import SI_CONSTANTS
from quantumforge.physics.wavefunction import WaveFunction


def quantum_potential(
    wavefunction: WaveFunction,
    mass: float,
    hbar: float = SI_CONSTANTS.hbar,
) -> np.ndarray:
    """Calculate the Bohmian quantum potential in arbitrary dimensions.

    The quantum potential is defined as:

        Q = -(hbar**2 / (2 * mass)) * laplacian(R) / R

    where R = |psi|.

    At exact nodes (R = 0), the quantum potential is undefined
    and represented by NaN.
    """
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass must be a positive finite value.")

    if not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be a positive finite value.")

    amplitude = np.abs(wavefunction.psi)
    laplacian = np.zeros_like(amplitude)

    for axis, coordinate in enumerate(
        reversed(wavefunction.coordinates)
    ):
        first_derivative = np.gradient(
            amplitude,
            coordinate,
            axis=axis,
            edge_order=2,
        )

        second_derivative = np.gradient(
            first_derivative,
            coordinate,
            axis=axis,
            edge_order=2,
        )

        laplacian += second_derivative

    potential = np.full(amplitude.shape, np.nan, dtype=float)

    np.divide(
        laplacian,
        amplitude,
        out=potential,
        where=amplitude > 0.0,
    )

    potential *= -(hbar**2) / (2.0 * mass)

    potential.setflags(write=False)

    return potential

