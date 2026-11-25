
"""Bohmian velocity calculations."""

import numpy as np

from quantumforge.physics.constants import SI_CONSTANTS
from quantumforge.physics.current import probability_current
from quantumforge.physics.wavefunction import WaveFunction


def bohmian_velocity(
    wavefunction: WaveFunction,
    mass: float,
    hbar: float = SI_CONSTANTS.hbar,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Calculate the Bohmian velocity field in arbitrary dimensions.

    The velocity is defined as:

        v = j / rho

    where j is the probability current and rho is the probability density.

    At exact nodes (rho = 0), the velocity is undefined and represented
    by NaN.

    In one dimension, returns vx.
    In N dimensions, returns (vx, vy, ..., vN).
    """
    current = probability_current(
        wavefunction,
        mass=mass,
        hbar=hbar,
    )

    density = wavefunction.density

    if wavefunction.ndim == 1:
        current = (current,)

    velocities = []

    for component in current:
        velocity = np.full(density.shape, np.nan, dtype=float)

        np.divide(
            component,
            density,
            out=velocity,
            where=density > 0.0,
        )

        velocity.setflags(write=False)
        velocities.append(velocity)

    if wavefunction.ndim == 1:
        return velocities[0]

    return tuple(velocities)

