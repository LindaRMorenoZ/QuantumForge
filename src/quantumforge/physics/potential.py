
"""External potential energy representations."""

from dataclasses import dataclass

import numpy as np

from quantumforge.physics.wavefunction import WaveFunction


@dataclass(frozen=True)
class Potential:
    """Represent a real external potential energy in arbitrary dimensions.

    Potential values are expressed in joules.

    A scalar represents a spatially constant potential. Array potentials
    are defined on a spatial grid.

    Coordinates are stored in the order (x, y, z, ...), while potential
    arrays follow NumPy axis order:
    (len(last_coordinate), ..., len(y), len(x)).
    """

    values: float | np.ndarray
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    coordinates: tuple[np.ndarray, ...] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).copy()

        if not np.all(np.isfinite(values)):
            raise ValueError("Potential values must be finite.")

        if self.coordinates is not None:
            if self.x is not None or self.y is not None:
                raise ValueError(
                    "Use either coordinates or x, y, not both."
                )
            axes = tuple(self.coordinates)
        else:
            if self.x is None:
                if self.y is not None:
                    raise ValueError("y cannot be provided without x.")

                if values.ndim > 2:
                    raise ValueError(
                        "Potential values must be scalar, 1D, or 2D when coordinates "
                        "are not provided."
                    )

                if values.ndim != 0:
                    raise ValueError(
                        "Spatial coordinates are required for array potentials."
                    )

                axes = ()
            else:
                axes = (
                    (self.x,)
                    if self.y is None
                    else (self.x, self.y)
                )

        if not axes:
            if values.ndim != 0:
                raise ValueError(
                    "Spatial coordinates are required for array potentials."
                )

            object.__setattr__(self, "values", float(values))
            object.__setattr__(self, "x", None)
            object.__setattr__(self, "y", None)
            object.__setattr__(self, "coordinates", ())
            return

        validated_axes = []

        for index, axis in enumerate(axes):
            name = f"coordinate axis {index}"

            grid = np.asarray(axis, dtype=float).copy()

            if grid.ndim != 1 or grid.size == 0:
                raise ValueError(
                    f"{name} must be a non-empty one-dimensional array."
                )

            if not np.all(np.isfinite(grid)):
                raise ValueError(f"{name} must contain only finite values.")

            if grid.size > 1 and not np.all(np.diff(grid) > 0):
                raise ValueError(f"{name} must be strictly increasing.")

            grid.setflags(write=False)
            validated_axes.append(grid)

        expected_shape = tuple(
            axis.size for axis in reversed(validated_axes)
        )

        if values.ndim != len(validated_axes) or values.shape != expected_shape:
            if len(validated_axes) == 1:
                raise ValueError(
                    "One-dimensional potential values must match x."
                )

            if len(validated_axes) == 2:
                raise ValueError(
                    "Two-dimensional potential values must have shape "
                    "(len(y), len(x))."
                )

            raise ValueError(
                f"Potential values must have shape {expected_shape}, "
                f"not {values.shape}."
            )

        values.setflags(write=False)

        coordinates = tuple(validated_axes)

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "x", coordinates[0])
        object.__setattr__(
            self,
            "y",
            coordinates[1] if len(coordinates) >= 2 else None,
        )

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return len(self.coordinates)

    @property
    def is_spatially_constant(self) -> bool:
        """Return whether the potential is spatially constant."""
        return self.ndim == 0

    def validate_compatibility(self, wavefunction: WaveFunction) -> None:
        """Validate compatibility with a wavefunction's spatial grid.

        A spatially constant potential is compatible with any
        wavefunction. Grid-based potentials must have matching
        dimensions and coordinates.
        """
        if self.is_spatially_constant:
            return

        if self.ndim != wavefunction.ndim:
            raise ValueError(
                "Potential and wavefunction dimensions must match."
            )

        for index, (potential_axis, wavefunction_axis) in enumerate(
            zip(self.coordinates, wavefunction.coordinates, strict=True)
        ):
            if not np.array_equal(potential_axis, wavefunction_axis):
                if index == 0:
                    axis_name = "x"
                elif index == 1:
                    axis_name = "y"
                elif index == 2:
                    axis_name = "z"
                else:
                    axis_name = f"axis {index}"

                raise ValueError(
                    f"Potential and wavefunction {axis_name} coordinates must match."
                )

    