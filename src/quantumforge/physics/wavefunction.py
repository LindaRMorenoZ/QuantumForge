
"""Representation of complex quantum wavefunctions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, init=False)
class WaveFunction:
    """Represent a complex wavefunction in an arbitrary number of dimensions.

    Spatial coordinates are stored in the order (x, y, z, ...).
    The wavefunction array follows NumPy axis order, so its shape is
    (len(last_coordinate), ..., len(y), len(x)).
    """

    coordinates: tuple[np.ndarray, ...]
    psi: np.ndarray

    def __init__(
        self,
        x: np.ndarray | None = None,
        psi: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
        *,
        coordinates: tuple[np.ndarray, ...] | None = None,
    ) -> None:
        """Initialize a wavefunction using legacy axes or arbitrary coordinates."""
        if psi is None:
            raise ValueError("psi must be provided.")

        if coordinates is not None:
            if x is not None or y is not None or z is not None:
                raise ValueError(
                    "Use either coordinates or x, y, z, not both."
                )
            axes = tuple(coordinates)
        else:
            if x is None:
                raise ValueError("At least one spatial coordinate is required.")

            if z is not None and y is None:
                raise ValueError("z cannot be provided without y.")

            axes = (x,) if y is None else (
                (x, y) if z is None else (x, y, z)
            )

        validated_axes = []

        for index, axis in enumerate(axes):
            name = f"coordinate axis {index}"

            grid = np.asarray(axis, dtype=float).copy()
            self._validate_grid(grid, name)
            grid.setflags(write=False)
            validated_axes.append(grid)

        if not validated_axes:
            raise ValueError("At least one spatial coordinate is required.")

        wavefunction = np.asarray(psi, dtype=complex).copy()

        if wavefunction.ndim != len(validated_axes):
            dimension = len(validated_axes)

            if dimension == 1:
                expected = "one-dimensional"
            elif dimension == 2:
                expected = "two-dimensional"
            elif dimension == 3:
                expected = "three-dimensional"
            else:
                expected = f"{dimension}-dimensional"

            raise ValueError(
                f"psi must be {expected} when {dimension} coordinate "
                f"axis{'es' if dimension != 1 else ''} are provided."
            )

        expected_shape = tuple(
            axis.size for axis in reversed(validated_axes)
        )

        if wavefunction.shape != expected_shape:
            raise ValueError(
                f"psi must have shape {expected_shape}, "
                f"not {wavefunction.shape}."
            )

        if not np.all(np.isfinite(wavefunction)):
            raise ValueError("psi must contain only finite values.")

        wavefunction.setflags(write=False)

        object.__setattr__(self, "coordinates", tuple(validated_axes))
        object.__setattr__(self, "psi", wavefunction)

    @staticmethod
    def _validate_grid(grid: np.ndarray, name: str) -> None:
        """Validate a one-dimensional spatial coordinate grid."""
        if grid.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")

        if grid.size == 0:
            raise ValueError(f"{name} cannot be empty.")

        if not np.all(np.isfinite(grid)):
            raise ValueError(f"{name} must contain only finite values.")

        if grid.size > 1 and not np.all(np.diff(grid) > 0):
            raise ValueError(f"{name} must be strictly increasing.")

    @property
    def x(self) -> np.ndarray:
        """Return the first spatial coordinate."""
        return self.coordinates[0]

    @property
    def y(self) -> np.ndarray | None:
        """Return the second spatial coordinate, if present."""
        return self.coordinates[1] if self.ndim >= 2 else None

    @property
    def z(self) -> np.ndarray | None:
        """Return the third spatial coordinate, if present."""
        return self.coordinates[2] if self.ndim >= 3 else None

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return len(self.coordinates)

    @property
    def density(self) -> np.ndarray:
        """Return the probability density |psi|^2."""
        return np.abs(self.psi) ** 2

