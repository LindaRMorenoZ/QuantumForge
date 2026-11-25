"""Tests for analytical quantum-mechanical reference solutions."""

import numpy as np
import pytest
from analytic_references import free_gaussian_wavefunction

from quantumforge.dynamics.trajectory import BohmianTrajectoryIntegrator
from quantumforge.physics.constants import SI_CONSTANTS


def test_free_gaussian_is_normalized_at_initial_time():
    mass = SI_CONSTANTS.electron_mass
    sigma = 1e-9
    x = np.linspace(-12.0 * sigma, 12.0 * sigma, 4801)

    wavefunction = free_gaussian_wavefunction(
        x=x,
        time=0.0,
        mass=mass,
        sigma=sigma,
    )

    probability = np.trapezoid(wavefunction.density, x)

    assert probability == pytest.approx(1.0, abs=1e-10)


def test_free_gaussian_width_matches_analytical_spreading():
    mass = SI_CONSTANTS.electron_mass
    hbar = SI_CONSTANTS.hbar
    sigma = 1e-9

    characteristic_time = 2.0 * mass * sigma**2 / hbar
    x = np.linspace(-12.0 * sigma, 12.0 * sigma, 4801)

    wavefunction = free_gaussian_wavefunction(
        x=x,
        time=characteristic_time,
        mass=mass,
        sigma=sigma,
    )

    probability = np.trapezoid(wavefunction.density, x)
    mean = np.trapezoid(x * wavefunction.density, x) / probability
    variance = (
        np.trapezoid((x - mean) ** 2 * wavefunction.density, x)
        / probability
    )

    assert np.sqrt(variance) == pytest.approx(
        np.sqrt(2.0) * sigma,
        rel=1e-8,
    )


def test_free_gaussian_center_remains_stationary():
    mass = SI_CONSTANTS.electron_mass
    sigma = 1e-9
    x = np.linspace(-12.0 * sigma, 12.0 * sigma, 4801)

    characteristic_time = 2.0 * mass * sigma**2 / SI_CONSTANTS.hbar

    for time in (0.0, characteristic_time / 2.0, characteristic_time):
        wavefunction = free_gaussian_wavefunction(
            x=x,
            time=time,
            mass=mass,
            sigma=sigma,
        )

        probability = np.trapezoid(wavefunction.density, x)
        mean = np.trapezoid(x * wavefunction.density, x) / probability

        assert mean == pytest.approx(0.0, abs=1e-15)


def test_free_gaussian_bohmian_trajectory_matches_analytical_solution():
    from quantumforge.numerics.schrodinger import SchrodingerSolver
    from quantumforge.physics.constants import SI_CONSTANTS
    from quantumforge.physics.potential import Potential
    from quantumforge.physics.wavefunction import WaveFunction

    mass = SI_CONSTANTS.electron_mass
    hbar = SI_CONSTANTS.hbar
    sigma = 1e-9

    characteristic_time = 2.0 * mass * sigma**2 / hbar
    subdivisions = 200
    dt = characteristic_time / subdivisions

    x = np.linspace(-12.0 * sigma, 12.0 * sigma, 4801)

    initial = free_gaussian_wavefunction(
        x=x,
        time=0.0,
        mass=mass,
        sigma=sigma,
    )

    initial_psi = initial.psi.copy()
    initial_psi[[0, -1]] = 0.0
    initial = WaveFunction(x=x, psi=initial_psi)

    solver = SchrodingerSolver(
        mass=mass,
        potential=Potential(values=0.0),
        dt=dt,
        hbar=hbar,
    )

    intervals = solver.evolve_with_midpoints(
        initial,
        steps=subdivisions,
    )

    initial_position = 2.0 * sigma

    integrator = BohmianTrajectoryIntegrator(
        mass=mass,
        dt=dt,
        hbar=hbar,
    )

    trajectory = integrator.integrate(
        initial_position,
        intervals,
    )

    analytical_position = initial_position * np.sqrt(2.0)

    assert trajectory[-1] == pytest.approx(
        analytical_position,
        rel=5e-3,
    )


