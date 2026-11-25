import numpy as np
import pytest

from quantumforge.physics.current import probability_current
from quantumforge.physics.wavefunction import WaveFunction


def test_probability_current_for_plane_wave():
    hbar = 1.0
    mass = 2.0
    momentum = 3.0

    x = np.linspace(-10.0, 10.0, 10001)
    psi = np.exp(1j * momentum * x / hbar)

    wavefunction = WaveFunction(x=x, psi=psi)

    current = probability_current(
        wavefunction,
        mass=mass,
        hbar=hbar,
    )

    expected = np.full_like(x, momentum / mass)

    np.testing.assert_allclose(
        current[1:-1],
        expected[1:-1],
        rtol=1e-5,
        atol=1e-5,
    )


def test_probability_current_for_real_wavefunction():
    x = np.linspace(-5.0, 5.0, 501)
    psi = np.exp(-x**2)

    wavefunction = WaveFunction(x=x, psi=psi)

    current = probability_current(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    np.testing.assert_allclose(current, 0.0, atol=1e-12)


@pytest.mark.parametrize("mass", [0.0, -1.0, np.inf, np.nan])
def test_probability_current_rejects_invalid_mass(mass):
    x = np.linspace(-1.0, 1.0, 10)
    psi = np.ones(10, dtype=complex)

    wavefunction = WaveFunction(x=x, psi=psi)

    with pytest.raises(ValueError):
        probability_current(wavefunction, mass=mass)


@pytest.mark.parametrize("hbar", [0.0, -1.0, np.inf, np.nan])
def test_probability_current_rejects_invalid_hbar(hbar):
    x = np.linspace(-1.0, 1.0, 10)
    psi = np.ones(10, dtype=complex)

    wavefunction = WaveFunction(x=x, psi=psi)

    with pytest.raises(ValueError):
        probability_current(
            wavefunction,
            mass=1.0,
            hbar=hbar,
        )



def test_two_dimensional_plane_wave_current():
    x = np.linspace(-10.0, 10.0, 10001)
    y = np.linspace(-10.0, 10.0, 10001)
    xx, yy = np.meshgrid(x, y)

    kx = 1.5
    ky = 2.0
    mass = 1.0
    hbar = 1.0

    psi = np.exp(1j * (kx * xx + ky * yy))

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    current_x, current_y = probability_current(
        wavefunction,
        mass=mass,
        hbar=hbar,
    )

    assert np.allclose(current_x, kx, rtol=1e-4)
    assert np.allclose(current_y, ky, rtol=1e-4)


def test_two_dimensional_real_wavefunction_has_zero_current():
    x = np.linspace(-5.0, 5.0, 101)
    y = np.linspace(-5.0, 5.0, 101)
    xx, yy = np.meshgrid(x, y)

    psi = np.exp(-(xx**2 + yy**2))

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    current_x, current_y = probability_current(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert np.allclose(current_x, 0.0)
    assert np.allclose(current_y, 0.0)


def test_two_dimensional_current_arrays_are_read_only():
    x = np.linspace(-1.0, 1.0, 11)
    y = np.linspace(-1.0, 1.0, 11)
    psi = np.ones((11, 11), dtype=complex)

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    current_x, current_y = probability_current(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert not current_x.flags.writeable
    assert not current_y.flags.writeable


def test_four_dimensional_probability_current():
    coordinates = tuple(
        np.linspace(-1.0, 1.0, 5) for _ in range(4)
    )

    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    psi = np.exp(1j * sum(grids))

    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=psi,
    )

    current = probability_current(wavefunction, mass=1.0, hbar=1.0)

    assert len(current) == 4

    for component in current:
        np.testing.assert_allclose(
            component,
            np.sinc(0.5 / np.pi) * np.abs(psi) ** 2,
            rtol=1e-12,
            atol=1e-12,
        )


def test_five_dimensional_probability_current():
    coordinates = tuple(
        np.linspace(-1.0, 1.0, 3) for _ in range(5)
    )

    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    psi = np.exp(1j * sum(grids))

    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=psi,
    )

    current = probability_current(wavefunction, mass=1.0, hbar=1.0)

    assert len(current) == 5

    for component in current:
        np.testing.assert_allclose(
            component,
            np.sinc(1.0 / np.pi) * np.abs(psi) ** 2,
            rtol=1e-12,
            atol=1e-12,
        )


