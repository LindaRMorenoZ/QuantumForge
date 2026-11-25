
"""Probability current calculations."""

import numpy as np

from quantumforge.physics.constants import SI_CONSTANTS
from quantumforge.physics.wavefunction import WaveFunction


def probability_current(
    wavefunction: WaveFunction,
    mass: float,
    hbar: float = SI_CONSTANTS.hbar,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Calculate the probability current density in arbitrary dimensions.

    The probability current is defined as:

        j = (hbar / mass) * Im(conj(psi) * grad(psi))

    Returns:
        1D: jx
        N-dimensional: (jx, jy, ..., jN)
    """
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass must be a positive finite value.")

    if not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be a positive finite value.")

    psi = wavefunction.psi
    factor = hbar / mass

    derivatives = np.gradient(
        psi,
        *reversed(wavefunction.coordinates),
    )

    if wavefunction.ndim == 1:
        derivatives = (derivatives,)

    currents = []

    for derivative in reversed(derivatives):
        current = factor * np.imag(np.conjugate(psi) * derivative)
        current.setflags(write=False)
        currents.append(current)

    if wavefunction.ndim == 1:
        return currents[0]

    return tuple(currents)
