"""Time-aware representations of quantum evolution intervals."""

from dataclasses import dataclass

import numpy as np

from quantumforge.physics.wavefunction import WaveFunction


@dataclass(frozen=True)
class EvolutionInterval:
    """Represent one evolution interval with its three temporal states."""

    initial_state: WaveFunction
    midpoint_state: WaveFunction
    final_state: WaveFunction
    t_start: float
    t_midpoint: float
    t_end: float

    def __post_init__(self) -> None:
        states = (
            self.initial_state,
            self.midpoint_state,
            self.final_state,
        )

        if not all(isinstance(state, WaveFunction) for state in states):
            raise TypeError("All interval states must be WaveFunction instances.")

        times = (self.t_start, self.t_midpoint, self.t_end)

        try:
            valid_times = all(
                np.isscalar(time)
                and not isinstance(time, (bool, np.bool_))
                and np.isfinite(time)
                for time in times
            )
        except TypeError:
            valid_times = False

        if not valid_times:
            raise ValueError("Interval times must be finite scalar values.")

        if not self.t_start < self.t_midpoint < self.t_end:
            raise ValueError("Interval times must be strictly increasing.")
        expected_midpoint = 0.5 * (self.t_start + self.t_end)
        time_scale = max(
            abs(self.t_start),
            abs(self.t_end),
            self.duration,
        )

        if abs(self.t_midpoint - expected_midpoint) > 1e-10 * time_scale:
            raise ValueError(
                "The midpoint time must be halfway between the start and end times."
            )
        
        if len({state.ndim for state in states}) != 1:
            raise ValueError("All interval states must have the same dimension.")

        reference_coordinates = self.initial_state.coordinates

        if any(
            not all(
                np.array_equal(reference, coordinate)
                for reference, coordinate in zip(
                    reference_coordinates,
                    state.coordinates,
                    strict=True,
                )
            )
            for state in states[1:]
        ):
            raise ValueError(
                "All interval states must use the same spatial grid."
            )                  

    @property
    def duration(self) -> float:
        """Return the total interval duration."""
        return self.t_end - self.t_start

    @property
    def dt(self) -> float:
        """Return the time step between interval endpoints."""
        return self.t_end - self.t_start

    def __iter__(self):
        """Allow unpacking the three temporal states."""
        yield self.initial_state
        yield self.midpoint_state
        yield self.final_state

    def __getitem__(self, index: int) -> WaveFunction:
        """Allow indexed access to the three temporal states."""
        states = (
            self.initial_state,
            self.midpoint_state,
            self.final_state,
        )
        return states[index]
    
            