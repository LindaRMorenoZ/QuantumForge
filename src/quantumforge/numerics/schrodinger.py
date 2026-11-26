
"""Time evolution of multidimensional wavefunctions."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import factorized

from quantumforge.dynamics.interval import EvolutionInterval
from quantumforge.numerics.operators import hamiltonian_nd
from quantumforge.physics.constants import SI_CONSTANTS
from quantumforge.physics.potential import Potential
from quantumforge.physics.wavefunction import WaveFunction


@dataclass(frozen=True)
class SchrodingerSolver:
    """Evolve multidimensional wavefunctions using the Crank-Nicolson method.

    The solver supports arbitrary spatial dimensions and assumes a
    time-independent, real potential with homogeneous Dirichlet boundary
    conditions on every spatial boundary.

    The evolution is performed on interior grid points using a
    second-order finite-difference Hamiltonian.
    """
    mass: float
    potential: Potential
    dt: float
    hbar: float = SI_CONSTANTS.hbar

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be a positive finite value.")

        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be a positive finite value.")

        if not np.isfinite(self.hbar) or self.hbar <= 0:
            raise ValueError("hbar must be a positive finite value.")

    def _prepare_evolution(
        self,
        wavefunction: WaveFunction,
        dt: float | None = None,
    ):
        """Prepare the Crank-Nicolson operators for a given grid and time step."""
        time_step = self.dt if dt is None else dt

        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("dt must be a positive finite value.")

        self.potential.validate_compatibility(wavefunction)

        hamiltonian = hamiltonian_nd(
            coordinates=wavefunction.coordinates,
            mass=self.mass,
            potential=self.potential,
            hbar=self.hbar,
        )

        identity = eye(hamiltonian.shape[0], format="csr")

        coefficient = 1j * time_step / (2.0 * self.hbar)

        lhs = identity + coefficient * hamiltonian
        rhs = identity - coefficient * hamiltonian

        solve = factorized(lhs.tocsc())

        return rhs, solve

    @staticmethod
    def _advance(
        wavefunction: WaveFunction,
        rhs,
        solve,
    ) -> WaveFunction:
        """Advance one time step using prepared operators."""
        interior_slices = tuple(
            slice(1, -1) for _ in wavefunction.coordinates
        )

        psi_interior = wavefunction.psi[interior_slices].ravel()

        next_interior = solve(rhs @ psi_interior)

        next_psi = np.zeros_like(wavefunction.psi)
        next_psi[interior_slices] = next_interior.reshape(
            next_psi[interior_slices].shape
        )

        return WaveFunction(
            coordinates=wavefunction.coordinates,
            psi=next_psi,
        )

    @staticmethod
    def _midpoint(
        wavefunction: WaveFunction,
        rhs_half,
        solve_half,
    ) -> WaveFunction:
        """Calculate the wavefunction at the midpoint of a time interval."""
        return SchrodingerSolver._advance(
            wavefunction,
            rhs_half,
            solve_half,
        )

    def step(self, wavefunction: WaveFunction) -> WaveFunction:
        """Advance a wavefunction by one time step."""
        rhs, solve = self._prepare_evolution(wavefunction)

        return self._advance(wavefunction, rhs, solve)

    def _evolve_with_operators(
        self,
        wavefunction: WaveFunction,
        steps: int,
        rhs,
        solve,
    ) -> list[WaveFunction]:
        """Evolve a wavefunction using previously prepared operators."""
        states = [wavefunction]
        current = wavefunction

        for _ in range(steps):
            current = self._advance(current, rhs, solve)
            states.append(current)

        return states

    def evolve(
        self,
        wavefunction: WaveFunction,
        steps: int,
    ) -> WaveFunction:
        """Advance a wavefunction through multiple time steps."""
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a non-negative integer.")

        if steps == 0:
            return wavefunction

        rhs, solve = self._prepare_evolution(wavefunction)

        return self._evolve_with_operators(
            wavefunction,
            steps,
            rhs,
            solve,
        )[-1]

    def evolve_states(
        self,
        wavefunction: WaveFunction,
        steps: int,
    ) -> list[WaveFunction]:
        """Return the initial wavefunction and all subsequent time steps."""
        if steps == 0:
            return [wavefunction]

        rhs, solve = self._prepare_evolution(wavefunction)

        return self._evolve_with_operators(
            wavefunction,
            steps,
            rhs,
            solve,
        )

    def evolve_with_midpoints(
        self,
        wavefunction: WaveFunction,
        steps: int,
        t0: float = 0.0,
    ) -> list[EvolutionInterval]:
        """Generate evolution intervals with initial, midpoint, and final states."""
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a non-negative integer.")

        if not np.isscalar(t0) or isinstance(t0, (bool, np.bool_)):
            raise ValueError("t0 must be a finite scalar.")

        if not np.isfinite(t0):
            raise ValueError("t0 must be a finite scalar.")

        if steps == 0:
            return []

        rhs_full, solve_full = self._prepare_evolution(wavefunction)
        rhs_half, solve_half = self._prepare_evolution(
            wavefunction,
            dt=self.dt / 2.0,
        )

        intervals = []
        current = wavefunction

        for index in range(steps):
            t_start = t0 + index * self.dt
            t_midpoint = t_start + self.dt / 2.0
            t_end = t_start + self.dt

            midpoint = self._midpoint(current, rhs_half, solve_half)
            next_state = self._advance(current, rhs_full, solve_full)

            intervals.append(
                EvolutionInterval(
                    initial_state=current,
                    midpoint_state=midpoint,
                    final_state=next_state,
                    t_start=t_start,
                    t_midpoint=t_midpoint,
                    t_end=t_end,
                )
            )

            current = next_state

        return intervals
