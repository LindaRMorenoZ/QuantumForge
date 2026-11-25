"""Reference-case validation for Bohmian trajectories."""

import numpy as np

from quantumforge.dynamics.interval import EvolutionInterval
from quantumforge.dynamics.trajectory import BohmianTrajectoryIntegrator
from quantumforge.physics.wavefunction import WaveFunction


def free_gaussian_trajectory(
    initial_position: float,
    initial_center: float,
    initial_momentum: float,
    mass: float,
    sigma: float,
    hbar: float,
    time: float,
) -> float:
    """Return the analytical Bohmian trajectory of a free Gaussian packet."""
    center = initial_center + (initial_momentum / mass) * time

    spreading = np.sqrt(
        1.0 + (hbar * time / (2.0 * mass * sigma**2)) ** 2
    )

    return center + (initial_position - initial_center) * spreading


def free_gaussian_wavefunction(
    x: np.ndarray,
    time: float,
    mass: float = 1.0,
    hbar: float = 1.0,
    sigma: float = 1.0,
    initial_center: float = 0.0,
    initial_momentum: float = 0.0,
) -> WaveFunction:
    """Return the analytical wavefunction of a freely evolving Gaussian."""
    tau = 2.0 * mass * sigma**2 / hbar
    center = initial_center + initial_momentum * time / mass

    spreading = 1.0 + 1j * time / tau

    psi = (
        (2.0 * np.pi * sigma**2) ** (-0.25)
        / np.sqrt(spreading)
        * np.exp(
            -(x - center) ** 2 / (4.0 * sigma**2 * spreading)
            + 1j
            * initial_momentum
            * (x - initial_momentum * time / (2.0 * mass))
            / hbar
        )
    )

    return WaveFunction(x=x, psi=psi)


def test_free_gaussian_trajectory_matches_analytical_center():
    """The central Bohmian trajectory must follow the packet center."""
    mass = 1.0
    hbar = 1.0
    sigma = 1.0
    initial_center = 0.0
    initial_momentum = 1.5
    initial_position = initial_center

    times = np.linspace(0.0, 2.0, 20)

    positions = [
        free_gaussian_trajectory(
            initial_position=initial_position,
            initial_center=initial_center,
            initial_momentum=initial_momentum,
            mass=mass,
            sigma=sigma,
            hbar=hbar,
            time=time,
        )
        for time in times
    ]

    expected_positions = initial_center + (initial_momentum / mass) * times

    np.testing.assert_allclose(positions, expected_positions)


def test_free_gaussian_trajectory_spreads_away_from_center():
    """Trajectories away from the center must spread with time."""
    initial_center = 0.0
    initial_position = 1.0

    initial_distance = abs(initial_position - initial_center)

    final_position = free_gaussian_trajectory(
        initial_position=initial_position,
        initial_center=initial_center,
        initial_momentum=0.0,
        mass=1.0,
        sigma=1.0,
        hbar=1.0,
        time=2.0,
    )

    final_distance = abs(final_position - initial_center)

    assert final_distance > initial_distance 


def test_bohmian_integrator_matches_free_gaussian_reference():
    """The numerical trajectory must approach the analytical solution."""
    mass = 1.0
    hbar = 1.0
    sigma = 1.0
    dt = 0.01
    steps = 20

    x = np.linspace(-15.0, 15.0, 3001)
    initial_position = 1.0

    integrator = BohmianTrajectoryIntegrator(
        mass=mass,
        dt=dt,
        hbar=hbar,
    )

    intervals = []
    for step in range(steps):
        t_start = step * dt
        t_midpoint = t_start + dt / 2.0
        t_end = t_start + dt

        intervals.append(
            EvolutionInterval(
                initial_state=free_gaussian_wavefunction(
                    x, t_start, mass, hbar, sigma
                ),
                midpoint_state=free_gaussian_wavefunction(
                    x, t_midpoint, mass, hbar, sigma
                ),
                final_state=free_gaussian_wavefunction(
                    x, t_end, mass, hbar, sigma
                ),
                t_start=t_start,
                t_midpoint=t_midpoint,
                t_end=t_end,
            )
        )

    trajectory = integrator.integrate(
        initial_position=initial_position,
        intervals=intervals,
    )

    expected_final_position = free_gaussian_trajectory(
        initial_position=initial_position,
        initial_center=0.0,
        initial_momentum=0.0,
        mass=mass,
        sigma=sigma,
        hbar=hbar,
        time=steps * dt,
    )

    np.testing.assert_allclose(
        trajectory[-1],
        expected_final_position,
        rtol=2e-3,
        atol=2e-3,
    )


def test_bohmian_integrator_matches_moving_gaussian_reference():
    """The central trajectory must follow a Gaussian packet with momentum."""
    mass = 1.0
    hbar = 1.0
    sigma = 1.0
    initial_momentum = 1.5
    dt = 0.01
    steps = 20

    x = np.linspace(-15.0, 15.0, 3001)
    initial_position = 0.0

    integrator = BohmianTrajectoryIntegrator(
        mass=mass,
        dt=dt,
        hbar=hbar,
    )

    intervals = []
    for step in range(steps):
        t_start = step * dt
        t_midpoint = t_start + dt / 2.0
        t_end = t_start + dt

        intervals.append(
            EvolutionInterval(
                initial_state=free_gaussian_wavefunction(
                    x, t_start, mass, hbar, sigma,
                    initial_momentum=initial_momentum,
                ),
                midpoint_state=free_gaussian_wavefunction(
                    x, t_midpoint, mass, hbar, sigma,
                    initial_momentum=initial_momentum,
                ),
                final_state=free_gaussian_wavefunction(
                    x, t_end, mass, hbar, sigma,
                    initial_momentum=initial_momentum,
                ),
                t_start=t_start,
                t_midpoint=t_midpoint,
                t_end=t_end,
            )
        )

    trajectory = integrator.integrate(
        initial_position=initial_position,
        intervals=intervals,
    )

    expected_final_position = (
        initial_position + initial_momentum * steps * dt / mass
    )

    np.testing.assert_allclose(
        trajectory[-1],
        expected_final_position,
        rtol=2e-3,
        atol=2e-3,
    )


def test_bohmian_integrator_temporal_convergence():
    """Reducing the time step must reduce the trajectory error."""
    mass = 1.0
    hbar = 1.0
    sigma = 1.0
    initial_position = 1.0
    final_time = 0.2
    x = np.linspace(-15.0, 15.0, 3001)

    errors = []

    for dt in (0.04, 0.02):
        steps = round(final_time / dt)

        integrator = BohmianTrajectoryIntegrator(
            mass=mass,
            dt=dt,
            hbar=hbar,
        )

        intervals = []
        for step in range(steps):
            t_start = step * dt
            t_midpoint = t_start + dt / 2.0
            t_end = t_start + dt

            intervals.append(
                EvolutionInterval(
                    initial_state=free_gaussian_wavefunction(
                        x, t_start, mass, hbar, sigma
                    ),
                    midpoint_state=free_gaussian_wavefunction(
                        x, t_midpoint, mass, hbar, sigma
                    ),
                    final_state=free_gaussian_wavefunction(
                        x, t_end, mass, hbar, sigma
                    ),
                    t_start=t_start,
                    t_midpoint=t_midpoint,
                    t_end=t_end,
                )
            )

        trajectory = integrator.integrate(
            initial_position=initial_position,
            intervals=intervals,
        )

        expected = free_gaussian_trajectory(
            initial_position=initial_position,
            initial_center=0.0,
            initial_momentum=0.0,
            mass=mass,
            sigma=sigma,
            hbar=hbar,
            time=final_time,
        )

        errors.append(abs(trajectory[-1] - expected))

    assert errors[1] < errors[0]

    
              