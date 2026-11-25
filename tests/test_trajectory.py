"""Tests for Bohmian trajectory integration."""

import numpy as np
import pytest

from quantumforge.dynamics.interval import EvolutionInterval
from quantumforge.dynamics.trajectory import (
    BohmianTrajectoryIntegrator,
    BohmianTrajectoryIntegrator1D,
)
from quantumforge.physics.wavefunction import WaveFunction


@pytest.mark.parametrize(
    ("mass", "dt", "hbar"),
    [
        (0.0, 0.1, 1.0),
        (-1.0, 0.1, 1.0),
        (np.inf, 0.1, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, -0.1, 1.0),
        (1.0, np.nan, 1.0),
        (1.0, 0.1, 0.0),
        (1.0, 0.1, -1.0),
        (1.0, 0.1, np.inf),
    ],
)
def test_integrator_rejects_invalid_parameters(mass, dt, hbar):
    with pytest.raises(ValueError):
        BohmianTrajectoryIntegrator1D(
            mass=mass,
            dt=dt,
            hbar=hbar,
        )


def test_integrator_accepts_valid_parameters():
    integrator = BohmianTrajectoryIntegrator1D(
        mass=1.0,
        dt=0.01,
        hbar=1.0,
    )

    assert integrator.mass == 1.0
    assert integrator.dt == 0.01
    assert integrator.hbar == 1.0


def test_integrates_constant_velocity_field_in_2d():
    from quantumforge.dynamics.trajectory import BohmianTrajectoryIntegrator

    x = np.linspace(-5.0, 5.0, 101)
    y = np.linspace(-5.0, 5.0, 101)

    grids = np.meshgrid(y, x, indexing="ij")
    y_grid, x_grid = grids

    kx = 0.4
    ky = 0.3

    psi = np.exp(1j * (kx * x_grid + ky * y_grid))
    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    vx = np.sinc(kx * dx / np.pi) * kx
    vy = np.sinc(ky * dy / np.pi) * ky

    dt = 0.01
    steps = 10

    intervals = [
        (wavefunction, wavefunction, wavefunction)
        for _ in range(steps)
    ]

    integrator = BohmianTrajectoryIntegrator(
        mass=1.0,
        dt=dt,
        hbar=1.0,
    )

    initial_position = np.array([0.0, 0.0])

    trajectory = integrator.integrate(
        initial_position,
        intervals,
    )

    expected = initial_position + steps * dt * np.array([vx, vy])

    assert trajectory.shape == (steps + 1, 2)

    np.testing.assert_allclose(
        trajectory[-1],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_integrates_constant_velocity_field_in_3d():
    from quantumforge.dynamics.trajectory import BohmianTrajectoryIntegrator

    x = np.linspace(-5.0, 5.0, 101)
    y = np.linspace(-5.0, 5.0, 101)
    z = np.linspace(-5.0, 5.0, 101)

    grids = np.meshgrid(z, y, x, indexing="ij")
    z_grid, y_grid, x_grid = grids

    kx = 0.4
    ky = 0.3
    kz = 0.2

    psi = np.exp(
        1j * (kx * x_grid + ky * y_grid + kz * z_grid)
    )

    wavefunction = WaveFunction(x=x, y=y, z=z, psi=psi)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    velocity = np.array([
        np.sinc(kx * dx / np.pi) * kx,
        np.sinc(ky * dy / np.pi) * ky,
        np.sinc(kz * dz / np.pi) * kz,
    ])

    dt = 0.01
    steps = 10

    intervals = [
        (wavefunction, wavefunction, wavefunction)
        for _ in range(steps)
    ]

    integrator = BohmianTrajectoryIntegrator(
        mass=1.0,
        dt=dt,
        hbar=1.0,
    )

    initial_position = np.array([0.0, 0.0, 0.0])

    trajectory = integrator.integrate(
        initial_position,
        intervals,
    )

    expected = initial_position + steps * dt * velocity

    assert trajectory.shape == (steps + 1, 3)

    np.testing.assert_allclose(
        trajectory[-1],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_integrates_constant_velocity_field_in_4d():
    from quantumforge.dynamics.trajectory import BohmianTrajectoryIntegrator

    coordinates = tuple(
        np.linspace(-5.0, 5.0, 11) for _ in range(4)
    )
    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    spatial_grids = tuple(reversed(grids))

    wave_numbers = np.array([0.4, 0.3, 0.2, 0.1])

    phase = sum(
        k * grid
        for k, grid in zip(wave_numbers, spatial_grids, strict=True)
    )

    psi = np.exp(1j * phase)
    wavefunction = WaveFunction(coordinates=coordinates, psi=psi)

    dx = coordinates[0][1] - coordinates[0][0]
    velocity = wave_numbers * np.sinc(wave_numbers * dx / np.pi)

    dt = 0.01
    steps = 5

    intervals = [
        (wavefunction, wavefunction, wavefunction)
        for _ in range(steps)
    ]

    integrator = BohmianTrajectoryIntegrator(
        mass=1.0,
        dt=dt,
        hbar=1.0,
    )

    initial_position = np.zeros(4)

    trajectory = integrator.integrate(
        initial_position,
        intervals,
    )

    expected = initial_position + steps * dt * velocity

    assert trajectory.shape == (steps + 1, 4)

    np.testing.assert_allclose(
        trajectory[-1],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_integrates_constant_velocity_field_in_5d():
    from quantumforge.dynamics.trajectory import BohmianTrajectoryIntegrator

    coordinates = tuple(
        np.linspace(-5.0, 5.0, 11) for _ in range(5)
    )
    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    spatial_grids = tuple(reversed(grids))

    wave_numbers = np.array([0.4, 0.3, 0.2, 0.1, 0.05])

    phase = sum(
        k * grid
        for k, grid in zip(wave_numbers, spatial_grids, strict=True)
    )

    psi = np.exp(1j * phase)
    wavefunction = WaveFunction(coordinates=coordinates, psi=psi)

    dx = coordinates[0][1] - coordinates[0][0]
    velocity = wave_numbers * np.sinc(wave_numbers * dx / np.pi)

    dt = 0.01
    steps = 5

    intervals = [
        (wavefunction, wavefunction, wavefunction)
        for _ in range(steps)
    ]

    integrator = BohmianTrajectoryIntegrator(
        mass=1.0,
        dt=dt,
        hbar=1.0,
    )

    initial_position = np.zeros(5)

    trajectory = integrator.integrate(
        initial_position,
        intervals,
    )

    expected = initial_position + steps * dt * velocity

    assert trajectory.shape == (steps + 1, 5)

    np.testing.assert_allclose(
        trajectory[-1],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_rejects_interval_with_invalid_wavefunction_type():
    integrator = BohmianTrajectoryIntegrator1D(
        mass=1.0,
        dt=0.01,
        hbar=1.0,
    )

    x = np.linspace(-5.0, 5.0, 101)
    psi = np.ones_like(x, dtype=complex)
    wavefunction = WaveFunction(x=x, psi=psi)

    intervals = [
        (wavefunction, "invalid", wavefunction),
    ]

    with pytest.raises(TypeError):
        integrator.integrate(0.0, intervals)


def test_integrates_independent_intervals_with_warning():
    x = np.linspace(-5.0, 5.0, 101)

    initial_state = WaveFunction(
        x=x,
        psi=np.ones_like(x, dtype=complex),
    )
    independent_state = WaveFunction(
        x=x,
        psi=np.exp(0.5j * x),
    )

    intervals = [
        EvolutionInterval(
            initial_state,
            initial_state,
            initial_state,
            t_start=0.0,
            t_midpoint=0.005,
            t_end=0.01,
        ),
        EvolutionInterval(
            independent_state,
            independent_state,
            independent_state,
            t_start=0.02,
            t_midpoint=0.025,
            t_end=0.03,
        ),
    ]

    integrator = BohmianTrajectoryIntegrator1D(
        mass=1.0,
        dt=0.01,
        hbar=1.0,
    )

    with pytest.warns(RuntimeWarning, match="not temporally continuous"):
        trajectory = integrator.integrate(0.0, intervals)

    assert trajectory.shape == (3,)
    assert np.all(np.isfinite(trajectory))


def test_integrator_uses_interval_duration():
    x = np.linspace(-5.0, 5.0, 101)
    psi = np.exp(1j * x)

    state = WaveFunction(x=x, psi=psi)

    intervals = [
        EvolutionInterval(
            state,
            state,
            state,
            t_start=0.0,
            t_midpoint=0.1,
            t_end=0.2,
        ),
        EvolutionInterval(
            state,
            state,
            state,
            t_start=0.2,
            t_midpoint=0.35,
            t_end=0.5,
        ),
    ]

    integrator = BohmianTrajectoryIntegrator(
        mass=1.0,
        dt=0.01,
        hbar=1.0,
    )

    trajectory = integrator.integrate(0.0, intervals)

    dx = x[1] - x[0]
    expected_velocity = np.sin(dx) / dx

    np.testing.assert_allclose(
        trajectory,
        [
            0.0,
            0.2 * expected_velocity,
            0.5 * expected_velocity,
        ],
        atol=1e-10,
    )