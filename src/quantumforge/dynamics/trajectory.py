
"""Bohmian trajectory integration."""

import warnings
from dataclasses import dataclass

import numpy as np

from quantumforge.dynamics.interval import EvolutionInterval
from quantumforge.physics.constants import SI_CONSTANTS
from quantumforge.physics.wavefunction import WaveFunction


@dataclass(frozen=True)
class BohmianTrajectoryIntegrator:
    """Integrate Bohmian trajectories in arbitrary spatial dimensions."""

    mass: float
    dt: float
    hbar: float = SI_CONSTANTS.hbar

    def __post_init__(self) -> None:
        """Validate physical and numerical parameters."""
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("Mass must be finite and positive.")

        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Time step must be finite and positive.")

        if not np.isfinite(self.hbar) or self.hbar <= 0.0:
            raise ValueError("Reduced Planck constant must be finite and positive.")

    @staticmethod
    def _validate_interval(
        interval: EvolutionInterval | tuple[WaveFunction, WaveFunction, WaveFunction],
        reference_coordinates: tuple[np.ndarray, ...] | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Validate an evolution interval and return its spatial coordinates."""
        if isinstance(interval, EvolutionInterval):
            states = (
                interval.initial_state,
                interval.midpoint_state,
                interval.final_state,
            )
        elif isinstance(interval, tuple):
            if len(interval) != 3:
                raise ValueError("Each interval must contain three wavefunctions.")
            states = interval
        else:
            raise TypeError(
                "Each interval must be an EvolutionInterval or a tuple of "
                "three WaveFunction instances."
            )

        if not all(isinstance(state, WaveFunction) for state in states):
            raise TypeError("Each interval element must be a WaveFunction.")

        current, midpoint, next_state = states

        for wavefunction in (midpoint, next_state):
            if wavefunction.ndim != current.ndim:
                raise ValueError(
                    "Wavefunctions in an interval must have the same dimension."
                )

            for axis, reference_axis in zip(
                wavefunction.coordinates,
                current.coordinates,
                strict=True,
            ):
                if not np.array_equal(axis, reference_axis):
                    raise ValueError(
                        "Wavefunctions in an interval must share the same grid."
                    )

        coordinates = current.coordinates

        if reference_coordinates is not None:
            for axis, reference_axis in zip(
                coordinates,
                reference_coordinates,
                strict=True,
            ):
                if not np.array_equal(axis, reference_axis):
                    raise ValueError(
                        "All evolution intervals must share the same spatial grid."
                    )

        return coordinates

    @staticmethod
    def _normalize_position(
        initial_position: float | tuple[float, ...] | np.ndarray,
        ndim: int,
    ) -> np.ndarray:
        """Convert an initial position to a validated coordinate vector."""
        position = np.asarray(initial_position, dtype=float)

        if ndim == 1 and position.ndim == 0:
            position = position.reshape(1)

        if position.shape != (ndim,):
            raise ValueError(
                f"Position must contain exactly {ndim} coordinate values."
            )

        if not np.all(np.isfinite(position)):
            raise ValueError("Initial position must be finite.")

        return position

    @staticmethod
    def _validate_domain(
        position: np.ndarray,
        coordinates: tuple[np.ndarray, ...],
    ) -> bool:
        """Check whether a position lies inside the spatial domain."""
        return all(
            axis[0] <= coordinate <= axis[-1]
            for axis, coordinate in zip(
                coordinates,
                position,
                strict=True,
            )
        )

    def _validate_inputs(
        self,
        initial_position: float | tuple[float, ...] | np.ndarray,
        intervals: list[
            EvolutionInterval | tuple[WaveFunction, WaveFunction, WaveFunction]
        ],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Validate integration inputs and return position and spatial grid."""
        if not isinstance(intervals, list):
            raise TypeError("Intervals must be provided as a list.")

        if not intervals:
            raise ValueError("At least one evolution interval is required.")

        reference_coordinates = self._validate_interval(intervals[0])
        ndim = len(reference_coordinates)

        position = self._normalize_position(initial_position, ndim)

        if not self._validate_domain(position, reference_coordinates):
            raise ValueError("Initial position must lie inside the spatial domain.")

        previous_end_time = (
            intervals[0].t_end
            if isinstance(intervals[0], EvolutionInterval)
            else None
        )

        for interval in intervals[1:]:
            self._validate_interval(interval, reference_coordinates)

            if isinstance(interval, EvolutionInterval):
                if previous_end_time is not None:
                    if not np.isclose(
                        interval.t_start,
                        previous_end_time,
                        rtol=1e-10,
                        atol=1e-12,
                    ):
                        warnings.warn(
                            "Evolution intervals are not temporally continuous. "
                            "The trajectory will be integrated across independent "
                            "time intervals.",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                previous_end_time = interval.t_end
            else:
                previous_end_time = None

        return position, reference_coordinates
 
    def _velocity_at(
        self,
        wavefunction: WaveFunction,
        position: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the Bohmian velocity at an arbitrary position."""
        from quantumforge.dynamics.interpolation import interpolate_velocity
        from quantumforge.physics.velocity import bohmian_velocity

        velocity = bohmian_velocity(
            wavefunction,
            mass=self.mass,
            hbar=self.hbar,
        )

        interpolation_position = (
            float(position[0]) if wavefunction.ndim == 1 else position
        )

        result = interpolate_velocity(
            wavefunction,
            velocity,
            interpolation_position,
        )

        result = np.atleast_1d(np.asarray(result, dtype=float))

        if result.shape != position.shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                f"Undefined Bohmian velocity at position {position}."
            )

        return result

    def _advance(
        self,
        position: np.ndarray,
        current: WaveFunction,
        midpoint: WaveFunction,
        step_index: int,
        dt: float,
    ) -> np.ndarray:
        """Advance one trajectory position using the midpoint method."""
        velocity_start = self._velocity_at(current, position)

        midpoint_position = position + 0.5 * dt * velocity_start

        if not self._validate_domain(midpoint_position, current.coordinates):
            raise RuntimeError(
                f"Trajectory left the spatial domain at midpoint "
                f"of step {step_index}: {midpoint_position}."
            )

        velocity_midpoint = self._velocity_at(
            midpoint,
            midpoint_position,
        )

        next_position = position + dt * velocity_midpoint

        if not self._validate_domain(next_position, current.coordinates):
            raise RuntimeError(
                f"Trajectory left the spatial domain at step "
                f"{step_index}: {next_position}."
            )

        return next_position
    
    def integrate(
        self,
        initial_position: float | tuple[float, ...] | np.ndarray,
        intervals: list[
            EvolutionInterval | tuple[WaveFunction, WaveFunction, WaveFunction]
        ],
    ) -> np.ndarray:
        """Integrate a Bohmian trajectory using the midpoint method.

        Returns
        -------
        numpy.ndarray
            In 1D, particle positions at the initial time and after
            each evolution interval, with shape (n_intervals + 1,).

            In N dimensions, particle positions with shape
            (n_intervals + 1, N).
        """
        position, coordinates = self._validate_inputs(
            initial_position,
            intervals,
        )

        ndim = len(coordinates)

        positions = np.empty(
            (len(intervals) + 1, ndim),
            dtype=float,
        )
        positions[0] = position

        for index, interval in enumerate(intervals):
            if isinstance(interval, EvolutionInterval):
                current = interval.initial_state
                midpoint = interval.midpoint_state
                time_step = interval.duration
            else:
                current, midpoint, _ = interval
                time_step = self.dt

            positions[index + 1] = self._advance(
                position=positions[index],
                current=current,
                midpoint=midpoint,
                step_index=index,
                dt=time_step,
            )

        if ndim == 1:
            positions = positions[:, 0]

        positions.setflags(write=False)

        return positions

BohmianTrajectoryIntegrator1D = BohmianTrajectoryIntegrator


