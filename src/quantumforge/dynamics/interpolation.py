
"""Spatial interpolation of Bohmian velocity fields."""

from itertools import product

import numpy as np

from quantumforge.physics.wavefunction import WaveFunction


def interpolate_velocity(
    wavefunction: WaveFunction,
    velocity: np.ndarray | tuple[np.ndarray, ...],
    position: float | tuple[float, ...] | np.ndarray,
) -> float | np.ndarray:
    """Interpolate a Bohmian velocity field in arbitrary dimensions.

    Linear interpolation in one dimension and multilinear interpolation
    in higher dimensions are used.

    Positions outside the spatial domain and interpolation cells
    containing undefined velocities return NaN.

    In one dimension, returns a scalar.
    In N dimensions, returns a velocity vector.
    """
    ndim = wavefunction.ndim

    if ndim == 1:
        if not isinstance(velocity, np.ndarray):
            raise ValueError("1D velocity must be a NumPy array.")

        if velocity.shape != wavefunction.psi.shape:
            raise ValueError("Velocity shape must match the wavefunction grid.")

        coordinates = (position,) if np.isscalar(position) else tuple(position)

    else:
        if not isinstance(velocity, tuple) or len(velocity) != ndim:
            raise ValueError(
                "Velocity must be a tuple containing one component per dimension."
            )

        if any(component.shape != wavefunction.psi.shape for component in velocity):
            raise ValueError(
                "Each velocity component must match the wavefunction grid."
            )

        coordinates = tuple(position)

    if len(coordinates) != ndim:
        raise ValueError(
            f"Position must contain exactly {ndim} coordinate values."
        )

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Position must be finite.")

    coordinates = tuple(float(value) for value in coordinates)

    axes = wavefunction.coordinates
    brackets = []

    for axis, coordinate in zip(axes, coordinates, strict=True):
        if coordinate < axis[0] or coordinate > axis[-1]:
            return float("nan") if ndim == 1 else np.full(ndim, np.nan)

        index = np.searchsorted(axis, coordinate)

        if index < len(axis) and axis[index] == coordinate:
            brackets.append(((index, 1.0),))
            continue

        left = index - 1
        right = index

        weight = (coordinate - axis[left]) / (axis[right] - axis[left])

        brackets.append(
            (
                (left, 1.0 - weight),
                (right, weight),
            )
        )

    result = np.zeros(ndim, dtype=float) if ndim > 1 else 0.0

    for vertex in product(*brackets):
        indices = tuple(item[0] for item in reversed(vertex))
        weight = np.prod([item[1] for item in vertex])

        if ndim == 1:
            value = velocity[indices[0]]

            if not np.isfinite(value):
                return float("nan")

            result += weight * value

        else:
            values = np.array(
                [component[indices] for component in velocity],
                dtype=float,
            )

            if not np.all(np.isfinite(values)):
                return np.full(ndim, np.nan)

            result += weight * values

    if ndim == 1:
        return float(result)

    result.setflags(write=False)
    return result

# Backward compatibility.
interpolate_velocity_1d = interpolate_velocity