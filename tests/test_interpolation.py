import numpy as np
import pytest

from quantumforge.dynamics.interpolation import interpolate_velocity_1d
from quantumforge.physics.wavefunction import WaveFunction


def create_wavefunction():
    x = np.linspace(0.0, 4.0, 5)
    psi = np.ones(5, dtype=complex)

    return WaveFunction(x=x, psi=psi)


def test_interpolates_velocity_at_grid_points():
    wavefunction = create_wavefunction()
    velocity = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    result = interpolate_velocity_1d(
        wavefunction,
        velocity,
        position=2.0,
    )

    assert result == 2.0


def test_interpolates_velocity_between_grid_points():
    wavefunction = create_wavefunction()
    velocity = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    result = interpolate_velocity_1d(
        wavefunction,
        velocity,
        position=1.5,
    )

    assert result == 1.5


def test_returns_nan_outside_spatial_domain():
    wavefunction = create_wavefunction()
    velocity = np.ones(5)

    result = interpolate_velocity_1d(
        wavefunction,
        velocity,
        position=4.5,
    )

    assert np.isnan(result)


def test_returns_nan_when_interpolation_crosses_undefined_velocity():
    wavefunction = create_wavefunction()
    velocity = np.array([0.0, 1.0, np.nan, 3.0, 4.0])

    result = interpolate_velocity_1d(
        wavefunction,
        velocity,
        position=1.5,
    )

    assert np.isnan(result)


def test_rejects_non_finite_position():
    wavefunction = create_wavefunction()
    velocity = np.ones(5)

    with pytest.raises(ValueError, match="Position must be finite"):
        interpolate_velocity_1d(
            wavefunction,
            velocity,
            position=np.nan,
        )


def test_rejects_velocity_with_incompatible_shape():
    wavefunction = create_wavefunction()
    velocity = np.ones(4)

    with pytest.raises(ValueError, match="Velocity shape"):
        interpolate_velocity_1d(
            wavefunction,
            velocity,
            position=1.5,
        )


def test_interpolate_velocity_2d():
    x = np.linspace(-2.0, 2.0, 5)
    y = np.linspace(-3.0, 3.0, 7)

    grids = np.meshgrid(y, x, indexing="ij")
    y_grid, x_grid = grids

    psi = np.ones_like(x_grid, dtype=complex)
    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    vx = 2.0 * x_grid + 3.0 * y_grid
    vy = -x_grid + 4.0 * y_grid

    velocity = (vx, vy)
    position = (0.3, -0.4)

    result = interpolate_velocity_1d(
        wavefunction,
        velocity,
        position,
    )

    expected = np.array([
        2.0 * position[0] + 3.0 * position[1],
        -position[0] + 4.0 * position[1],
    ])

    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_interpolate_velocity_3d():
    x = np.linspace(-2.0, 2.0, 5)
    y = np.linspace(-3.0, 3.0, 7)
    z = np.linspace(-1.0, 1.0, 5)

    grids = np.meshgrid(z, y, x, indexing="ij")
    z_grid, y_grid, x_grid = grids

    psi = np.ones_like(x_grid, dtype=complex)
    wavefunction = WaveFunction(x=x, y=y, z=z, psi=psi)

    vx = x_grid + 2.0 * y_grid + 3.0 * z_grid
    vy = -2.0 * x_grid + y_grid - z_grid
    vz = 4.0 * x_grid + 0.5 * y_grid + 2.0 * z_grid

    velocity = (vx, vy, vz)
    position = (0.3, -0.4, 0.2)

    result = interpolate_velocity_1d(
        wavefunction,
        velocity,
        position,
    )

    expected = np.array([
        position[0] + 2.0 * position[1] + 3.0 * position[2],
        -2.0 * position[0] + position[1] - position[2],
        4.0 * position[0] + 0.5 * position[1] + 2.0 * position[2],
    ])

    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_interpolate_velocity_4d():
    coordinates = tuple(
        np.linspace(-2.0, 2.0, 5) for _ in range(4)
    )
    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    w_grid, z_grid, y_grid, x_grid = grids

    psi = np.ones_like(x_grid, dtype=complex)
    wavefunction = WaveFunction(coordinates=coordinates, psi=psi)

    vx = x_grid + 2.0 * y_grid + z_grid + 3.0 * w_grid
    vy = 2.0 * x_grid - y_grid + z_grid - w_grid
    vz = -x_grid + 0.5 * y_grid + 2.0 * z_grid + w_grid
    vw = 3.0 * x_grid + y_grid - z_grid + 2.0 * w_grid

    velocity = (vx, vy, vz, vw)
    position = (0.3, -0.4, 0.2, 0.1)

    result = interpolate_velocity_1d(
        wavefunction,
        velocity,
        position,
    )

    expected = np.array([
        position[0] + 2.0 * position[1] + position[2] + 3.0 * position[3],
        2.0 * position[0] - position[1] + position[2] - position[3],
        -position[0] + 0.5 * position[1] + 2.0 * position[2] + position[3],
        3.0 * position[0] + position[1] - position[2] + 2.0 * position[3],
    ])

    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_interpolate_velocity_5d():
    coordinates = tuple(
        np.linspace(-2.0, 2.0, 5) for _ in range(5)
    )
    grids = np.meshgrid(*reversed(coordinates), indexing="ij")
    v_grid, w_grid, z_grid, y_grid, x_grid = grids

    psi = np.ones_like(x_grid, dtype=complex)
    wavefunction = WaveFunction(coordinates=coordinates, psi=psi)

    velocity = (
        x_grid + y_grid + z_grid + w_grid + v_grid,
        2.0 * x_grid - y_grid + z_grid - w_grid + v_grid,
        -x_grid + 2.0 * y_grid - z_grid + w_grid - v_grid,
        3.0 * x_grid + y_grid + 2.0 * z_grid - w_grid + v_grid,
        x_grid - y_grid + z_grid + 2.0 * w_grid - 3.0 * v_grid,
    )

    position = (0.3, -0.4, 0.2, 0.1, -0.3)

    result = interpolate_velocity_1d(
        wavefunction,
        velocity,
        position,
    )

    x, y, z, w, v = position

    expected = np.array([
        x + y + z + w + v,
        2.0 * x - y + z - w + v,
        -x + 2.0 * y - z + w - v,
        3.0 * x + y + 2.0 * z - w + v,
        x - y + z + 2.0 * w - 3.0 * v,
    ])

    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_interpolate_velocity_returns_nan_outside_domain():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)

    grids = np.meshgrid(y, x, indexing="ij")
    y_grid, x_grid = grids

    psi = np.ones_like(x_grid, dtype=complex)
    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    velocity = (
        np.ones_like(psi),
        2.0 * np.ones_like(psi),
    )

    result = interpolate_velocity_1d(
        wavefunction,
        velocity,
        (1.5, 0.0),
    )

    assert np.all(np.isnan(result))


def test_interpolate_velocity_returns_nan_for_undefined_vertex():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)

    grids = np.meshgrid(y, x, indexing="ij")
    y_grid, x_grid = grids

    psi = np.ones((len(y), len(x)), dtype=complex)
    wavefunction = WaveFunction(x=x, y=y, psi=psi)

    vx = np.ones(psi.shape, dtype=float)
    vy = 2.0 * np.ones(psi.shape, dtype=float)

    vx[2, 2] = np.nan

    result = interpolate_velocity_1d(
        wavefunction,
        (vx, vy),
        (0.1, 0.1),
    )

    assert np.all(np.isnan(result))   