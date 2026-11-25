import numpy as np
import pytest

from quantumforge.physics.wavefunction import WaveFunction


def test_probability_density():
    x = np.array([-1.0, 0.0, 1.0])
    psi = np.array([1 + 1j, 2 + 0j, 0 + 1j])

    wavefunction = WaveFunction(x=x, psi=psi)

    expected = np.array([2.0, 4.0, 1.0])

    np.testing.assert_allclose(wavefunction.density, expected)


def test_wavefunction_rejects_different_array_sizes():
    x = np.array([0.0, 1.0])
    psi = np.array([1.0 + 0j])

    with pytest.raises(ValueError):
        WaveFunction(x=x, psi=psi)


def test_wavefunction_rejects_non_increasing_coordinates():
    x = np.array([0.0, 2.0, 1.0])
    psi = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        WaveFunction(x=x, psi=psi)


def test_wavefunction_arrays_are_read_only():
    x = np.array([0.0, 1.0])
    psi = np.array([1.0 + 0j, 2.0 + 0j])

    wavefunction = WaveFunction(x=x, psi=psi)

    with pytest.raises(ValueError):
        wavefunction.psi[0] = 3.0



def test_two_dimensional_wavefunction():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-2.0, 2.0, 3)
    psi = np.ones((3, 5), dtype=complex)

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    assert wavefunction.ndim == 2
    assert wavefunction.psi.shape == (3, 5)
    assert wavefunction.density.shape == (3, 5)


def test_two_dimensional_wavefunction_arrays_are_read_only():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-2.0, 2.0, 3)
    psi = np.ones((3, 5), dtype=complex)

    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    assert not wavefunction.x.flags.writeable
    assert not wavefunction.y.flags.writeable
    assert not wavefunction.psi.flags.writeable


def test_two_dimensional_wavefunction_rejects_wrong_shape():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-2.0, 2.0, 3)
    psi = np.ones((5, 3), dtype=complex)

    with pytest.raises(ValueError, match="shape"):
        WaveFunction(x=x, y=y, psi=psi)


def test_two_dimensional_wavefunction_rejects_non_increasing_y():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.array([-1.0, 0.0, 0.0])
    psi = np.ones((3, 5), dtype=complex)

    with pytest.raises(ValueError, match="strictly increasing"):
        WaveFunction(x=x, y=y, psi=psi)


def test_one_dimensional_wavefunction_rejects_two_dimensional_psi():
    x = np.linspace(-1.0, 1.0, 5)
    psi = np.ones((3, 5), dtype=complex)

    with pytest.raises(ValueError, match="one-dimensional"):
        WaveFunction(x=x, psi=psi)


def test_three_dimensional_wavefunction():
    x = np.linspace(-2.0, 2.0, 11)
    y = np.linspace(-2.0, 2.0, 13)
    z = np.linspace(-2.0, 2.0, 15)

    psi = np.ones((len(z), len(y), len(x)), dtype=complex)

    wavefunction = WaveFunction(
        x=x,
        y=y,
        z=z,
        psi=psi,
    )

    assert wavefunction.ndim == 3
    assert wavefunction.psi.shape == (15, 13, 11)
    assert wavefunction.density.shape == (15, 13, 11)
    assert wavefunction.z is not None
    assert not wavefunction.z.flags.writeable


def test_three_dimensional_wavefunction_rejects_invalid_shape():
    x = np.linspace(-2.0, 2.0, 11)
    y = np.linspace(-2.0, 2.0, 13)
    z = np.linspace(-2.0, 2.0, 15)

    psi = np.ones((15, 13, 10), dtype=complex)

    with pytest.raises(ValueError, match="shape"):
        WaveFunction(x=x, y=y, z=z, psi=psi)


def test_three_dimensional_wavefunction_rejects_z_without_y():
    x = np.linspace(-2.0, 2.0, 11)
    z = np.linspace(-2.0, 2.0, 15)
    psi = np.ones(11, dtype=complex)

    with pytest.raises(ValueError, match="without y"):
        WaveFunction(x=x, z=z, psi=psi)


def test_four_dimensional_wavefunction():
    coordinates = tuple(
        np.linspace(-1.0, 1.0, 3) for _ in range(4)
    )
    psi = np.ones((3, 3, 3, 3), dtype=complex)

    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=psi,
    )

    assert wavefunction.ndim == 4
    assert wavefunction.psi.shape == (3, 3, 3, 3)
    assert len(wavefunction.coordinates) == 4
    assert wavefunction.density.shape == psi.shape


def test_five_dimensional_wavefunction():
    coordinates = tuple(
        np.linspace(-1.0, 1.0, 2) for _ in range(5)
    )
    psi = np.ones((2, 2, 2, 2, 2), dtype=complex)

    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=psi,
    )

    assert wavefunction.ndim == 5
    assert wavefunction.psi.shape == (2, 2, 2, 2, 2)
    assert len(wavefunction.coordinates) == 5


