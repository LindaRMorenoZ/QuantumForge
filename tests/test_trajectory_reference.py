"""Reference-case validation for Bohmian trajectories."""

import numpy as np
import pytest

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

    spreading = np.sqrt(1.0 + (hbar * time / (2.0 * mass * sigma**2)) ** 2)

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
            -((x - center) ** 2) / (4.0 * sigma**2 * spreading)
            + 1j * initial_momentum * (x - initial_momentum * time / (2.0 * mass)) / hbar
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
                initial_state=free_gaussian_wavefunction(x, t_start, mass, hbar, sigma),
                midpoint_state=free_gaussian_wavefunction(x, t_midpoint, mass, hbar, sigma),
                final_state=free_gaussian_wavefunction(x, t_end, mass, hbar, sigma),
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
                    x,
                    t_start,
                    mass,
                    hbar,
                    sigma,
                    initial_momentum=initial_momentum,
                ),
                midpoint_state=free_gaussian_wavefunction(
                    x,
                    t_midpoint,
                    mass,
                    hbar,
                    sigma,
                    initial_momentum=initial_momentum,
                ),
                final_state=free_gaussian_wavefunction(
                    x,
                    t_end,
                    mass,
                    hbar,
                    sigma,
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

    expected_final_position = initial_position + initial_momentum * steps * dt / mass

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
                    initial_state=free_gaussian_wavefunction(x, t_start, mass, hbar, sigma),
                    midpoint_state=free_gaussian_wavefunction(x, t_midpoint, mass, hbar, sigma),
                    final_state=free_gaussian_wavefunction(x, t_end, mass, hbar, sigma),
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


def measure_free_gaussian_trajectory_error(
    dt: float,
    final_time: float = 0.2,
    spatial_points: int = 12001,
) -> float:
    """Measure the absolute final trajectory error."""
    return abs(
        measure_free_gaussian_trajectory_signed_error(
            dt=dt,
            final_time=final_time,
            spatial_points=spatial_points,
        )
    )


def measure_free_gaussian_trajectory_signed_error(
    dt: float,
    final_time: float = 0.2,
    spatial_points: int = 12001,
) -> float:
    """Measure the signed final trajectory error."""
    mass = 1.0
    hbar = 1.0
    sigma = 1.0
    initial_position = 1.0
    x = np.linspace(-15.0, 15.0, spatial_points)

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
                initial_state=free_gaussian_wavefunction(x, t_start, mass, hbar, sigma),
                midpoint_state=free_gaussian_wavefunction(x, t_midpoint, mass, hbar, sigma),
                final_state=free_gaussian_wavefunction(x, t_end, mass, hbar, sigma),
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

    return float(trajectory[-1] - expected)


def free_gaussian_bohmian_velocity(
    position: float,
    time: float,
    mass: float = 1.0,
    hbar: float = 1.0,
    sigma: float = 1.0,
    initial_center: float = 0.0,
    initial_momentum: float = 0.0,
) -> float:
    """Return the analytical Bohmian velocity of a free Gaussian packet."""
    tau = 2.0 * mass * sigma**2 / hbar
    center = initial_center + initial_momentum * time / mass

    return initial_momentum / mass + (position - center) * time / (tau**2 + time**2)


def integrate_analytical_bohmian_velocity(
    initial_position: float,
    final_time: float,
    dt: float,
    mass: float = 1.0,
    hbar: float = 1.0,
    sigma: float = 1.0,
    initial_center: float = 0.0,
    initial_momentum: float = 0.0,
) -> float:
    """Integrate the analytical Bohmian velocity using the midpoint method."""
    position = initial_position
    time = 0.0

    while time < final_time:
        step = min(dt, final_time - time)

        velocity_start = free_gaussian_bohmian_velocity(
            position,
            time,
            mass,
            hbar,
            sigma,
            initial_center,
            initial_momentum,
        )

        midpoint_position = position + 0.5 * step * velocity_start
        midpoint_time = time + 0.5 * step

        velocity_midpoint = free_gaussian_bohmian_velocity(
            midpoint_position,
            midpoint_time,
            mass,
            hbar,
            sigma,
            initial_center,
            initial_momentum,
        )

        position += step * velocity_midpoint
        time += step

    return position


def test_bohmian_integrator_error_decreases_with_temporal_refinement():
    """Verify that trajectory errors decrease under temporal refinement."""

    time_steps = (0.04, 0.02, 0.01, 0.005)

    errors = [measure_free_gaussian_trajectory_error(dt) for dt in time_steps]

    convergence_orders = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0) for index in range(len(errors) - 1)
    ]

    assert all(error > 0.0 for error in errors)
    assert all(convergence_orders[index + 1] > 0.0 for index in range(len(convergence_orders) - 1))


def test_analytical_bohmian_velocity_shows_third_order_convergence():
    """Verify the observed third-order convergence for this analytical case."""
    initial_position = 1.0
    final_time = 0.2
    time_steps = (0.04, 0.02, 0.01, 0.005)

    exact_position = initial_position * np.sqrt(1.0 + (final_time / 2.0) ** 2)

    errors = []

    for dt in time_steps:
        numerical_position = integrate_analytical_bohmian_velocity(
            initial_position=initial_position,
            final_time=final_time,
            dt=dt,
        )
        errors.append(abs(numerical_position - exact_position))

    convergence_orders = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0) for index in range(len(errors) - 1)
    ]

    assert all(errors[index + 1] < errors[index] for index in range(len(errors) - 1))
    assert all(2.8 < order < 3.2 for order in convergence_orders)


def test_bohmian_integrator_temporal_convergence_with_analytical_velocity(
    monkeypatch,
):
    """Test temporal convergence of the actual integrator with analytical velocity."""
    initial_position = 1.0
    final_time = 0.2
    time_steps = (0.04, 0.02, 0.01, 0.005)

    coordinates = np.linspace(-15.0, 15.0, 12001)
    errors = []

    for dt in time_steps:
        intervals = []
        wavefunction_times = {}

        times = np.arange(0.0, final_time, dt)

        for t_start in times:
            t_midpoint = t_start + 0.5 * dt
            t_end = t_start + dt

            initial_state = free_gaussian_wavefunction(coordinates, t_start)
            midpoint_state = free_gaussian_wavefunction(coordinates, t_midpoint)
            final_state = free_gaussian_wavefunction(coordinates, t_end)

            intervals.append(
                EvolutionInterval(
                    initial_state=initial_state,
                    midpoint_state=midpoint_state,
                    final_state=final_state,
                    t_start=t_start,
                    t_midpoint=t_midpoint,
                    t_end=t_end,
                )
            )

            wavefunction_times[id(initial_state)] = t_start
            wavefunction_times[id(midpoint_state)] = t_midpoint

        def analytical_velocity(
            self,
            wavefunction,
            position,
            wavefunction_times=wavefunction_times,
        ):
            time = wavefunction_times[id(wavefunction)]
            return np.array(
                [
                    free_gaussian_bohmian_velocity(
                        position[0],
                        time,
                    )
                ]
            )

        monkeypatch.setattr(
            BohmianTrajectoryIntegrator,
            "_velocity_at",
            analytical_velocity,
        )

        integrator = BohmianTrajectoryIntegrator(mass=1.0, dt=dt)
        trajectory = integrator.integrate(initial_position, intervals)

        exact_position = initial_position * np.sqrt(1.0 + (final_time / 2.0) ** 2)
        errors.append(abs(trajectory[-1] - exact_position))

    convergence_orders = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0) for index in range(len(errors) - 1)
    ]

    assert all(errors[index + 1] < errors[index] for index in range(len(errors) - 1))
    assert all(2.8 < order < 3.2 for order in convergence_orders)


def test_full_integrator_temporal_error_differences_are_third_order():
    """Verify third-order temporal error differences for the free Gaussian case."""
    time_steps = (0.04, 0.02, 0.01, 0.005)
    spatial_resolutions = (24001, 48001)

    for spatial_points in spatial_resolutions:
        errors = [
            measure_free_gaussian_trajectory_signed_error(
                dt,
                spatial_points=spatial_points,
            )
            for dt in time_steps
        ]

        error_differences = [
            abs(errors[index] - errors[index + 1]) for index in range(len(errors) - 1)
        ]

        convergence_orders = [
            np.log(error_differences[index] / error_differences[index + 1]) / np.log(2.0)
            for index in range(len(error_differences) - 1)
        ]

        assert all(difference > 0.0 for difference in error_differences)
        assert all(2.8 < order < 3.2 for order in convergence_orders)


def test_bohmian_integrator_accepts_consecutive_si_time_intervals():
    """Consecutive intervals at SI time scales must be accepted."""
    mass = 1.0
    hbar = 1.0
    sigma = 1.0
    dt = 1.73e-14

    x = np.linspace(-15.0, 15.0, 3001)

    intervals = []
    for step in range(2):
        t_start = step * dt
        t_midpoint = t_start + dt / 2.0
        t_end = t_start + dt

        intervals.append(
            EvolutionInterval(
                initial_state=free_gaussian_wavefunction(x, t_start, mass, hbar, sigma),
                midpoint_state=free_gaussian_wavefunction(x, t_midpoint, mass, hbar, sigma),
                final_state=free_gaussian_wavefunction(x, t_end, mass, hbar, sigma),
                t_start=t_start,
                t_midpoint=t_midpoint,
                t_end=t_end,
            )
        )

    integrator = BohmianTrajectoryIntegrator(
        mass=mass,
        dt=dt,
        hbar=hbar,
    )

    trajectory = integrator.integrate(
        initial_position=1.0,
        intervals=intervals,
    )

    assert trajectory.shape == (3,)
    assert np.all(np.isfinite(trajectory))


def test_bohmian_integrator_warns_on_si_scale_time_discontinuity():
    """A real temporal gap at SI scales must trigger a warning."""
    dt = 1.73e-14
    gap = 1.0e-15
    x = np.linspace(-15.0, 15.0, 3001)

    intervals = []

    for t_start in (0.0, dt + gap):
        t_midpoint = t_start + dt / 2.0
        t_end = t_start + dt

        intervals.append(
            EvolutionInterval(
                initial_state=free_gaussian_wavefunction(x, t_start),
                midpoint_state=free_gaussian_wavefunction(x, t_midpoint),
                final_state=free_gaussian_wavefunction(x, t_end),
                t_start=t_start,
                t_midpoint=t_midpoint,
                t_end=t_end,
            )
        )

    integrator = BohmianTrajectoryIntegrator(
        mass=1.0,
        dt=dt,
        hbar=1.0,
    )

    with pytest.warns(RuntimeWarning, match="not temporally continuous"):
        integrator.integrate(
            initial_position=1.0,
            intervals=intervals,
        )
