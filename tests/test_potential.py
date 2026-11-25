
import numpy as np
import pytest

from quantumforge.physics.potential import Potential


def test_constant_potential():
    potential = Potential(values=2.5)

    assert potential.ndim == 0
    assert potential.values == 2.5


def test_one_dimensional_potential():
    x = np.linspace(-5.0, 5.0, 101)
    values = 0.5 * x**2

    potential = Potential(values=values, x=x)

    assert potential.ndim == 1
    assert np.array_equal(potential.x, x)
    assert np.array_equal(potential.values, values)


def test_two_dimensional_potential():
    x = np.linspace(-5.0, 5.0, 51)
    y = np.linspace(-3.0, 3.0, 31)
    xx, yy = np.meshgrid(x, y)

    values = xx**2 + yy**2

    potential = Potential(values=values, x=x, y=y)

    assert potential.ndim == 2
    assert potential.values.shape == (len(y), len(x))
    assert np.array_equal(potential.values, values)


def test_potential_values_are_read_only():
    x = np.linspace(-1.0, 1.0, 11)
    potential = Potential(values=x**2, x=x)

    assert not potential.values.flags.writeable
    assert not potential.x.flags.writeable


def test_two_dimensional_coordinates_are_read_only():
    x = np.linspace(-1.0, 1.0, 11)
    y = np.linspace(-1.0, 1.0, 11)
    values = np.ones((11, 11))

    potential = Potential(values=values, x=x, y=y)

    assert not potential.x.flags.writeable
    assert not potential.y.flags.writeable


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([[1.0, -np.inf]]),
    ],
)
def test_non_finite_potential_values_are_rejected(values):
    with pytest.raises(ValueError, match="finite"):
        Potential(values=values, x=np.arange(values.shape[-1]))


def test_potential_rejects_three_dimensions():
    values = np.ones((3, 3, 3))

    with pytest.raises(ValueError, match="scalar, 1D, or 2D"):
        Potential(values=values)


def test_array_potential_requires_coordinates():
    with pytest.raises(ValueError, match="coordinates"):
        Potential(values=np.array([1.0, 2.0]))


def test_y_cannot_be_provided_without_x():
    with pytest.raises(ValueError, match="without x"):
        Potential(values=1.0, y=np.array([0.0, 1.0]))


def test_one_dimensional_potential_rejects_wrong_shape():
    x = np.linspace(-1.0, 1.0, 11)
    values = np.ones(10)

    with pytest.raises(ValueError, match="match x"):
        Potential(values=values, x=x)


def test_two_dimensional_potential_rejects_wrong_shape():
    x = np.linspace(-1.0, 1.0, 11)
    y = np.linspace(-1.0, 1.0, 9)
    values = np.ones((11, 9))

    with pytest.raises(ValueError, match="shape"):
        Potential(values=values, x=x, y=y)


def test_coordinates_must_be_strictly_increasing():
    x = np.array([-1.0, 0.0, 0.0, 1.0])
    values = np.ones(4)

    with pytest.raises(ValueError, match="strictly increasing"):
        Potential(values=values, x=x)


def test_spatially_constant_potential_property():
    constant = Potential(values=2.5)

    x = np.linspace(-1.0, 1.0, 11)
    spatial = Potential(values=x**2, x=x)

    assert constant.is_spatially_constant
    assert not spatial.is_spatially_constant



def test_constant_potential_is_compatible_with_any_wavefunction():
    from quantumforge.physics.wavefunction import WaveFunction

    potential = Potential(values=2.5)
    x = np.linspace(-1.0, 1.0, 11)
    wavefunction = WaveFunction(x=x, psi=np.ones(11, dtype=complex))

    potential.validate_compatibility(wavefunction)


def test_compatible_one_dimensional_potential():
    from quantumforge.physics.wavefunction import WaveFunction

    x = np.linspace(-1.0, 1.0, 11)
    potential = Potential(values=x**2, x=x)
    wavefunction = WaveFunction(x=x, psi=np.ones(11, dtype=complex))

    potential.validate_compatibility(wavefunction)


def test_incompatible_dimensions_are_rejected():
    from quantumforge.physics.wavefunction import WaveFunction

    x = np.linspace(-1.0, 1.0, 11)
    y = np.linspace(-1.0, 1.0, 9)

    potential = Potential(values=x**2, x=x)
    wavefunction = WaveFunction(
        x=x,
        y=y,
        psi=np.ones((9, 11), dtype=complex),
    )

    with pytest.raises(ValueError, match="dimensions"):
        potential.validate_compatibility(wavefunction)


def test_incompatible_x_coordinates_are_rejected():
    from quantumforge.physics.wavefunction import WaveFunction

    x = np.linspace(-1.0, 1.0, 11)
    different_x = np.linspace(-1.1, 1.1, 11)

    potential = Potential(values=x**2, x=x)
    wavefunction = WaveFunction(
        x=different_x,
        psi=np.ones(11, dtype=complex),
    )

    with pytest.raises(ValueError, match="x coordinates"):
        potential.validate_compatibility(wavefunction)


def test_incompatible_y_coordinates_are_rejected():
    from quantumforge.physics.wavefunction import WaveFunction

    x = np.linspace(-1.0, 1.0, 11)
    y = np.linspace(-1.0, 1.0, 9)
    different_y = np.linspace(-1.1, 1.1, 9)

    potential = Potential(
        values=np.ones((9, 11)),
        x=x,
        y=y,
    )
    wavefunction = WaveFunction(
        x=x,
        y=different_y,
        psi=np.ones((9, 11), dtype=complex),
    )

    with pytest.raises(ValueError, match="y coordinates"):
        potential.validate_compatibility(wavefunction)


def test_four_dimensional_potential():
    coordinates = tuple(
        np.linspace(-1.0, 1.0, 5) for _ in range(4)
    )
    values = np.ones((5, 5, 5, 5))

    potential = Potential(
        values=values,
        coordinates=coordinates,
    )

    assert potential.ndim == 4

    for actual, expected in zip(
        potential.coordinates,
        coordinates,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(potential.x, coordinates[0])
    np.testing.assert_array_equal(potential.y, coordinates[1])


def test_four_dimensional_potential_compatibility():
    from quantumforge.physics.wavefunction import WaveFunction

    coordinates = tuple(
        np.linspace(-1.0, 1.0, 5) for _ in range(4)
    )
    values = np.ones((5, 5, 5, 5))

    potential = Potential(
        values=values,
        coordinates=coordinates,
    )
    wavefunction = WaveFunction(
        coordinates=coordinates,
        psi=np.ones((5, 5, 5, 5), dtype=complex),
    )

    potential.validate_compatibility(wavefunction)


def test_five_dimensional_potential():
    coordinates = tuple(
        np.linspace(-1.0, 1.0, 4) for _ in range(5)
    )
    values = np.ones((4, 4, 4, 4, 4))

    potential = Potential(
        values=values,
        coordinates=coordinates,
    )

    assert potential.ndim == 5
    assert len(potential.coordinates) == 5


def test_incompatible_fourth_coordinate_is_rejected():
    from quantumforge.physics.wavefunction import WaveFunction

    coordinates = tuple(
        np.linspace(-1.0, 1.0, 5) for _ in range(4)
    )
    different_coordinates = (
        *coordinates[:3],
        np.linspace(-1.1, 1.1, 5),
    )

    potential = Potential(
        values=np.ones((5, 5, 5, 5)),
        coordinates=coordinates,
    )
    wavefunction = WaveFunction(
        coordinates=different_coordinates,
        psi=np.ones((5, 5, 5, 5), dtype=complex),
    )

    with pytest.raises(ValueError, match="axis 3 coordinates"):
        potential.validate_compatibility(wavefunction)
