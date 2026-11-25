import sys
from pathlib import Path

import numpy as np
import pytest

# Make the src-layout package importable when tests are run from the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantumforge.physics.quantum_potential import quantum_potential
from quantumforge.physics.wavefunction import WaveFunction


def test_quantum_potential_for_gaussian_wavefunction():
    hbar = 1.0
    mass = 2.0
    sigma = 1.0

    x = np.linspace(-8.0, 8.0, 10001)
    psi = np.exp(-x**2 / (4.0 * sigma**2))

    wavefunction = WaveFunction(x=x, psi=psi)

    potential = quantum_potential(
        wavefunction,
        mass=mass,
        hbar=hbar,
    )

    expected = (
        hbar**2 / (2.0 * mass)
        * (1.0 / (2.0 * sigma**2) - x**2 / (4.0 * sigma**4))
    )

    np.testing.assert_allclose(
        potential[10:-10],
        expected[10:-10],
        rtol=1e-4,
        atol=1e-4,
    )


def test_quantum_potential_for_constant_amplitude():
    x = np.linspace(-5.0, 5.0, 501)
    psi = np.ones(501, dtype=complex)

    wavefunction = WaveFunction(x=x, psi=psi)

    potential = quantum_potential(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    np.testing.assert_allclose(potential, 0.0, atol=1e-12)


def test_quantum_potential_is_undefined_at_exact_nodes():
    x = np.linspace(-np.pi, np.pi, 1001)
    psi = np.sin(x).astype(complex)

    wavefunction = WaveFunction(x=x, psi=psi)

    potential = quantum_potential(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert np.isnan(potential[500])


@pytest.mark.parametrize("mass", [0.0, -1.0, np.inf, np.nan])
def test_quantum_potential_rejects_invalid_mass(mass: float):
    x = np.linspace(-1.0, 1.0, 10)
    psi = np.ones(10, dtype=complex)

    wavefunction = WaveFunction(x=x, psi=psi)

    with pytest.raises(ValueError):
        quantum_potential(wavefunction, mass=mass)


@pytest.mark.parametrize("hbar", [0.0, -1.0, np.inf, np.nan])
def test_quantum_potential_rejects_invalid_hbar(hbar: float):
    x = np.linspace(-1.0, 1.0, 10)
    psi = np.ones(10, dtype=complex)

    wavefunction = WaveFunction(x=x, psi=psi)

    with pytest.raises(ValueError):
        quantum_potential(
            wavefunction,
            mass=1.0,
            hbar=hbar,
        )



def test_two_dimensional_gaussian_quantum_potential():
    x = np.linspace(-5.0, 5.0, 1001)
    y = np.linspace(-5.0, 5.0, 1001)
    xx, yy = np.meshgrid(x, y)

    alpha = 0.5
    mass = 1.0
    hbar = 1.0

    psi = np.exp(-alpha * (xx**2 + yy**2))

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    potential = quantum_potential(
        wavefunction,
        mass=mass,
        hbar=hbar,
    )

    expected = -(
        hbar**2 / (2.0 * mass)
    ) * (
        4.0 * alpha**2 * (xx**2 + yy**2) - 4.0 * alpha
    )


    mask= (np.abs(xx) <= 2.0) & (np.abs(yy) <= 2.0)
    
    assert np.allclose(
        potential[mask],
        expected[mask],
        rtol=2e-4,
        atol=2.5e-4,
    )


def test_two_dimensional_constant_amplitude_has_zero_quantum_potential():
    x = np.linspace(-5.0, 5.0, 101)
    y = np.linspace(-5.0, 5.0, 101)

    psi = np.ones((101, 101), dtype=complex)

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    potential = quantum_potential(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert np.allclose(potential, 0.0)


def test_two_dimensional_wavefunction_node_has_nan_quantum_potential():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)

    psi = np.ones((5, 5), dtype=complex)
    psi[2, 2] = 0.0

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    potential = quantum_potential(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert np.isnan(potential[2, 2])


def test_two_dimensional_quantum_potential_is_read_only():
    x = np.linspace(-1.0, 1.0, 11)
    y = np.linspace(-1.0, 1.0, 11)

    psi = np.ones((11, 11), dtype=complex)

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    potential = quantum_potential(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    assert not potential.flags.writeable


def test_four_dimensional_quantum_potential():
    coordinates = tuple(
        np.linspace(-4.0, 4.0, 13) for _ in range(4)
    )
    sigma = 5.0

    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    radius_squared = sum(grid**2 for grid in grids)

    psi = np.exp(
        -radius_squared / (2.0 * sigma**2)
    ).astype(complex)

    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=psi,
    )

    potential = quantum_potential(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    expected = 0.5 * (
        4 / sigma**2
        - radius_squared / sigma**4
    )

    np.testing.assert_allclose(
        potential[2:-2, 2:-2, 2:-2, 2:-2],
        expected[2:-2, 2:-2, 2:-2, 2:-2],
        rtol=2e-3,
        atol=2e-3,
    )


def test_five_dimensional_quantum_potential():
    coordinates = tuple(
        np.linspace(-4.0, 4.0, 13) for _ in range(5)
    )
    sigma = 5.0

    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    radius_squared = sum(grid**2 for grid in grids)

    psi = np.exp(
        -radius_squared / (2.0 * sigma**2)
    ).astype(complex)
    
    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=psi,
    )

    potential = quantum_potential(
        wavefunction,
        mass=1.0,
        hbar=1.0,
    )

    expected = 0.5 * (
        5 / sigma**2
        - radius_squared / sigma**4
    )

    np.testing.assert_allclose(
        potential[2:-2, 2:-2, 2:-2, 2:-2, 2:-2],
        expected[2:-2, 2:-2, 2:-2, 2:-2, 2:-2],
        rtol=1e-2,
        atol=1e-2,
    )