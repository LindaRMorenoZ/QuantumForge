
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from quantumforge.numerics.operators import hamiltonian_1d, laplacian_1d
from quantumforge.physics.potential import Potential


def test_laplacian_returns_sparse_csr_matrix():
    x = np.linspace(-5.0, 5.0, 101)

    operator = laplacian_1d(x)

    assert isinstance(operator, csr_matrix)
    assert operator.shape == (99, 99)


def test_laplacian_diagonal_elements():
    x = np.linspace(-5.0, 5.0, 11)
    dx = x[1] - x[0]

    operator = laplacian_1d(x).toarray()

    expected = -2.0 / dx**2

    assert np.allclose(np.diag(operator), expected)


def test_laplacian_off_diagonal_elements():
    x = np.linspace(-5.0, 5.0, 11)
    dx = x[1] - x[0]

    operator = laplacian_1d(x).toarray()

    expected = 1.0 / dx**2

    assert np.allclose(np.diag(operator, k=1), expected)
    assert np.allclose(np.diag(operator, k=-1), expected)


def test_laplacian_is_symmetric():
    x = np.linspace(-5.0, 5.0, 101)

    operator = laplacian_1d(x)

    difference = operator - operator.T

    assert difference.nnz == 0


def test_laplacian_rejects_non_uniform_grid():
    x = np.array([0.0, 1.0, 2.0, 4.0, 5.0])

    with pytest.raises(ValueError, match="uniformly spaced"):
        laplacian_1d(x)


def test_laplacian_rejects_non_increasing_grid():
    x = np.array([0.0, 1.0, 2.0, 1.5, 4.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        laplacian_1d(x)


def test_laplacian_rejects_insufficient_grid_points():
    x = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="at least three"):
        laplacian_1d(x)


def test_laplacian_rejects_non_finite_coordinates():
    x = np.array([0.0, 1.0, np.nan, 3.0])

    with pytest.raises(ValueError, match="finite"):
        laplacian_1d(x)


def test_hamiltonian_is_sparse_and_has_correct_shape():
    x = np.linspace(-5.0, 5.0, 101)
    potential = Potential(values=0.0)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    assert isinstance(hamiltonian, csr_matrix)
    assert hamiltonian.shape == (99, 99)


def test_hamiltonian_is_hermitian():
    x = np.linspace(-5.0, 5.0, 101)
    potential = Potential(values=0.5 * x**2, x=x)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    difference = hamiltonian - hamiltonian.getH()

    assert difference.nnz == 0


def test_constant_potential_shifts_hamiltonian_diagonal():
    x = np.linspace(-5.0, 5.0, 11)
    mass = 1.0
    hbar = 1.0
    constant_energy = 2.5

    free_hamiltonian = hamiltonian_1d(
        x=x,
        mass=mass,
        potential=Potential(values=0.0),
        hbar=hbar,
    )

    shifted_hamiltonian = hamiltonian_1d(
        x=x,
        mass=mass,
        potential=Potential(values=constant_energy),
        hbar=hbar,
    )

    difference = shifted_hamiltonian - free_hamiltonian

    assert np.allclose(
        difference.toarray(),
        constant_energy * np.eye(x.size - 2),
    )


def test_hamiltonian_rejects_incompatible_potential():
    x = np.linspace(-5.0, 5.0, 11)
    different_x = np.linspace(-4.0, 4.0, 11)

    potential = Potential(
        values=np.ones(11),
        x=different_x,
    )

    with pytest.raises(ValueError, match="x coordinates"):
        hamiltonian_1d(
            x=x,
            mass=1.0,
            potential=potential,
            hbar=1.0,
        )


@pytest.mark.parametrize(
    ("mass", "hbar"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
    ],
)
def test_hamiltonian_rejects_invalid_physical_parameters(mass, hbar):
    x = np.linspace(-5.0, 5.0, 11)

    with pytest.raises(ValueError):
        hamiltonian_1d(
            x=x,
            mass=mass,
            potential=Potential(values=0.0),
            hbar=hbar,
        )



def test_hamiltonian_ground_state_energy_converges_spatially():
    from scipy.linalg import eigvalsh

    from quantumforge.numerics.operators import hamiltonian_1d

    hbar = 1.0
    mass = 1.0
    length = 1.0
    exact_energy = np.pi**2 * hbar**2 / (2.0 * mass * length**2)

    errors = []

    for number_of_points in (21, 41, 81):
        x = np.linspace(0.0, length, number_of_points)

        hamiltonian = hamiltonian_1d(
            x=x,
            mass=mass,
            potential=Potential(0.0),
            hbar=hbar,
        ).toarray()

        numerical_energy = eigvalsh(hamiltonian)[0]
        errors.append(abs(numerical_energy - exact_energy))

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]

    assert np.isclose(
        errors[0] / errors[1],
        4.0,
        rtol=0.03,
    )
    assert np.isclose(
        errors[1] / errors[2],
        4.0,
        rtol=0.03,
    )


def test_free_hamiltonian_is_positive_definite():
    from scipy.linalg import eigvalsh

    x = np.linspace(-5.0, 5.0, 101)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=Potential(0.0),
        hbar=1.0,
    ).toarray()

    eigenvalues = eigvalsh(hamiltonian)

    assert np.all(eigenvalues > 0.0)

    
def test_laplacian_2d_shape_and_symmetry():
    from quantumforge.numerics.operators import laplacian_nd

    x = np.linspace(-1.0, 1.0, 7)
    y = np.linspace(-1.0, 1.0, 9)

    operator = laplacian_nd((x, y))

    expected_size = (x.size - 2) * (y.size - 2)

    assert operator.shape == (expected_size, expected_size)
    assert (operator - operator.T).nnz == 0


def test_laplacian_5d_shape_and_symmetry():
    from quantumforge.numerics.operators import laplacian_nd

    coordinates = tuple(
        np.linspace(-1.0, 1.0, 4) for _ in range(5)
    )

    operator = laplacian_nd(coordinates)

    expected_size = 2**5

    assert operator.shape == (expected_size, expected_size)
    assert (operator - operator.T).nnz == 0


def test_laplacian_nd_has_negative_eigenvalues():
    from scipy.sparse.linalg import eigsh

    from quantumforge.numerics.operators import laplacian_nd

    coordinates = (
        np.linspace(-1.0, 1.0, 7),
        np.linspace(-1.0, 1.0, 7),
    )

    operator = laplacian_nd(coordinates)

    largest_eigenvalue = eigsh(
        operator,
        k=1,
        which="LA",
        return_eigenvectors=False,
    )[0]

    assert largest_eigenvalue < 0.0


def test_hamiltonian_2d_shape_and_hermiticity():
    from quantumforge.numerics.operators import hamiltonian_nd
    from quantumforge.physics.potential import Potential

    x = np.linspace(-1.0, 1.0, 7)
    y = np.linspace(-1.0, 1.0, 9)

    potential = Potential(
        values=np.zeros((9, 7)),
        coordinates=(x, y),
    )

    operator = hamiltonian_nd(
        (x, y),
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    expected_size = (x.size - 2) * (y.size - 2)

    assert operator.shape == (expected_size, expected_size)
    assert (operator - operator.getH()).nnz == 0


def test_hamiltonian_3d_with_constant_potential():
    from quantumforge.numerics.operators import hamiltonian_nd
    from quantumforge.physics.potential import Potential

    coordinates = tuple(
        np.linspace(-1.0, 1.0, 5) for _ in range(3)
    )

    potential = Potential(values=2.0)

    operator = hamiltonian_nd(
        coordinates,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    expected_size = 3**3

    assert operator.shape == (expected_size, expected_size)
    np.testing.assert_allclose(
        operator.diagonal() - 2.0,
        operator.diagonal() - 2.0,
    )


def test_hamiltonian_5d_shape():
    from quantumforge.numerics.operators import hamiltonian_nd
    from quantumforge.physics.potential import Potential

    coordinates = tuple(
        np.linspace(-1.0, 1.0, 4) for _ in range(5)
    )

    potential = Potential(values=0.0)

    operator = hamiltonian_nd(
        coordinates,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    expected_size = 2**5

    assert operator.shape == (expected_size, expected_size) 


def test_hamiltonian_nd_has_positive_eigenvalues():
    from scipy.sparse.linalg import eigsh

    from quantumforge.numerics.operators import hamiltonian_nd
    from quantumforge.physics.potential import Potential

    coordinates = (
        np.linspace(-1.0, 1.0, 7),
        np.linspace(-1.0, 1.0, 7),
    )

    operator = hamiltonian_nd(
        coordinates,
        mass=1.0,
        potential=Potential(values=0.0),
        hbar=1.0,
    )

    smallest_eigenvalue = eigsh(
        operator,
        k=1,
        which="SA",
        return_eigenvectors=False,
    )[0]

    assert smallest_eigenvalue > 0.0   


def test_hamiltonian_nd_matches_hamiltonian_1d():
    from quantumforge.numerics.operators import (
        hamiltonian_1d,
        hamiltonian_nd,
    )
    from quantumforge.physics.potential import Potential

    x = np.linspace(-2.0, 2.0, 21)
    potential = Potential(values=0.5 * x**2, x=x)

    hamiltonian_1d_operator = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    hamiltonian_nd_operator = hamiltonian_nd(
        coordinates=(x,),
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    difference = hamiltonian_1d_operator - hamiltonian_nd_operator

    assert difference.nnz == 0        