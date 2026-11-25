import numpy as np
import pytest

from quantumforge.physics.velocity import bohmian_velocity
from quantumforge.physics.wavefunction import WaveFunction


def test_bohmian_velocity_for_plane_wave():
    hbar = 1.0
    mass = 2.0
    momentum = 3.0

    x = np.linspace(-10.0, 10.0, 10001)
    psi = np.exp(1j * momentum * x / hbar)

    wavefunction = WaveFunction(x=x, psi=psi)

    velocity = bohmian_velocity(
        wavefunction,
        mass=mass,
        hbar=hbar,
    )

    expected = momentum / mass

    np.testing.assert_allclose(
        velocity[1:-1],
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_bohmian_velocity_for_real_wavefunction():
    x = np.linspace(-5.0, 5.0, 501)
    psi = np.exp(-x**2)

    wavefunction = WaveFunction(x=x, psi=psi)

    velocity = bohmian_velocity(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    np.testing.assert_allclose(velocity, 0.0, atol=1e-12)


def test_bohmian_velocity_is_undefined_at_wavefunction_nodes():
    x = np.linspace(-np.pi, np.pi, 1001)
    psi = np.sin(x).astype(complex)

    wavefunction = WaveFunction(x=x, psi=psi)

    velocity = bohmian_velocity(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert np.isnan(velocity[500])


@pytest.mark.parametrize("mass", [0.0, -1.0, np.inf, np.nan])
def test_bohmian_velocity_rejects_invalid_mass(mass):
    x = np.linspace(-1.0, 1.0, 10)
    psi = np.ones(10, dtype=complex)

    wavefunction = WaveFunction(x=x, psi=psi)

    with pytest.raises(ValueError):
        bohmian_velocity(wavefunction, mass=mass)



def test_two_dimensional_plane_wave_velocity():
    x = np.linspace(-10.0, 10.0, 10001)
    y = np.linspace(-10.0, 10.0, 10001)
    xx, yy = np.meshgrid(x, y)

    kx = 1.5
    ky = 2.0
    mass = 1.0
    hbar = 1.0

    psi = np.exp(1j * (kx * xx + ky * yy))

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    velocity_x, velocity_y = bohmian_velocity(
        wavefunction,
        mass=mass,
        hbar=hbar,
    )

    assert np.allclose(velocity_x, kx, rtol=1e-4)
    assert np.allclose(velocity_y, ky, rtol=1e-4)


def test_two_dimensional_real_wavefunction_has_zero_velocity():
    x = np.linspace(-5.0, 5.0, 101)
    y = np.linspace(-5.0, 5.0, 101)
    xx, yy = np.meshgrid(x, y)

    psi = np.exp(-(xx**2 + yy**2))

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    velocity_x, velocity_y = bohmian_velocity(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert np.allclose(velocity_x, 0.0)
    assert np.allclose(velocity_y, 0.0)


def test_two_dimensional_wavefunction_node_has_nan_velocity():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)

    psi = np.ones((5, 5), dtype=complex)
    psi[2, 2] = 0.0

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    velocity_x, velocity_y = bohmian_velocity(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert np.isnan(velocity_x[2, 2])
    assert np.isnan(velocity_y[2, 2])


def test_two_dimensional_velocity_arrays_are_read_only():
    x = np.linspace(-1.0, 1.0, 11)
    y = np.linspace(-1.0, 1.0, 11)
    psi = np.ones((11, 11), dtype=complex)

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    velocity_x, velocity_y = bohmian_velocity(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert not velocity_x.flags.writeable
    assert not velocity_y.flags.writeable


def test_four_dimensional_bohmian_velocity():
    coordinates = tuple(
        np.linspace(-1.0, 1.0, 5) for _ in range(4)
    )

    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    psi = np.exp(1j * sum(grids))

    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=psi,
    )

    velocity = bohmian_velocity(wavefunction, mass=1.0, hbar=1.0)

    assert len(velocity) == 4

    for component in velocity:
        np.testing.assert_allclose(
            component,
            np.sinc(0.5 / np.pi),
            rtol=1e-12,
            atol=1e-12,
        )


def test_five_dimensional_bohmian_velocity():
    coordinates = tuple(
        np.linspace(-1.0, 1.0, 3) for _ in range(5)
    )

    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    psi = np.exp(1j * sum(grids))

    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=psi,
    )

    velocity = bohmian_velocity(wavefunction, mass=1.0, hbar=1.0)

    assert len(velocity) == 5

    for component in velocity:
        np.testing.assert_allclose(
            component,
            np.sinc(1.0 / np.pi),
            rtol=1e-12,
            atol=1e-12,
        )


def test_four_dimensional_bohmian_velocity_is_undefined_at_nodes():
    coordinates = tuple(
        np.linspace(-1.0, 1.0, 3) for _ in range(4)
    )

    psi = np.ones((3, 3, 3, 3), dtype=complex)
    psi[1, 1, 1, 1] = 0.0

    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=psi,
    )

    velocity = bohmian_velocity(wavefunction, mass=1.0, hbar=1.0)

    for component in velocity:
        assert np.isnan(component[1, 1, 1, 1]).all()
