
"""Finite-difference spatial operators."""

from functools import reduce

import numpy as np
from scipy.sparse import diags, kronsum

from quantumforge.physics.constants import SI_CONSTANTS
from quantumforge.physics.potential import Potential


def _validate_physical_parameters(mass: float, hbar: float) -> None:
    """Validate mass and reduced Planck constant."""
    if (
        not isinstance(mass, (int, float, np.integer, np.floating))
        or isinstance(mass, (bool, np.bool_))
        or not np.isfinite(mass)
        or mass <= 0
    ):
        raise ValueError("mass must be a positive finite real value.")

    if (
        not isinstance(hbar, (int, float, np.integer, np.floating))
        or isinstance(hbar, (bool, np.bool_))
        or not np.isfinite(hbar)
        or hbar <= 0
    ):
        raise ValueError("hbar must be a positive finite real value.")

def laplacian_1d(x: np.ndarray):
    """Construct the 1D finite-difference Laplacian.

    Uses a uniform spatial grid and second-order central differences.
    Homogeneous Dirichlet boundary conditions are assumed.

    Returns the operator acting on interior grid points only.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must contain at least three grid points.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")

    differences = np.diff(x)

    if not np.all(differences > 0):
        raise ValueError("x must be strictly increasing.")

    dx = differences[0]

    if not np.allclose(differences, dx, rtol=1e-10, atol=1e-14):
        raise ValueError("x must be uniformly spaced.")

    number_of_interior_points = x.size - 2

    main_diagonal = -2.0 * np.ones(number_of_interior_points)
    off_diagonal = np.ones(number_of_interior_points - 1)

    operator = diags(
        [
            off_diagonal,
            main_diagonal,
            off_diagonal,
        ],
        offsets=[-1, 0, 1],
        format="csr",
    )

    return operator / dx**2

def laplacian_nd(coordinates: tuple[np.ndarray, ...]):
    """Construct the finite-difference Laplacian in arbitrary dimensions.

    Uses a uniform spatial grid and second-order central differences
    along each coordinate axis.

    Homogeneous Dirichlet boundary conditions are assumed.
    The operator acts on interior grid points only.

    Coordinates are provided in the order (x, y, z, ...).
    """
    if not coordinates:
        raise ValueError("At least one coordinate axis is required.")

    operators = [
        laplacian_1d(np.asarray(axis, dtype=float))
        for axis in reversed(coordinates)
    ]

    return reduce(kronsum, operators).tocsr()

def hamiltonian_1d(
    x: np.ndarray,
    mass: float,
    potential: Potential,
    hbar: float = SI_CONSTANTS.hbar,
):
    """Construct the 1D discrete Hamiltonian.

    Uses a second-order finite-difference kinetic operator and a
    diagonal potential-energy operator.

    Homogeneous Dirichlet boundary conditions are assumed.
    The Hamiltonian acts on interior grid points only.
    """
    _validate_physical_parameters(mass, hbar)
    x = np.asarray(x, dtype=float)

    potential.validate_coordinates((x,))

    if potential.is_spatially_constant:
        potential_values = np.full(x.size, potential.values)
    else:
        potential_values = potential.values

    laplacian = laplacian_1d(x)

    kinetic = -(hbar**2 / (2.0 * mass)) * laplacian

    potential_operator = diags(
        potential_values[1:-1],
        offsets=0,
        format="csr",
    )

    return kinetic + potential_operator

def hamiltonian_nd(
    coordinates: tuple[np.ndarray, ...],
    mass: float,
    potential: Potential,
    hbar: float = SI_CONSTANTS.hbar,
):
    """Construct the discrete Hamiltonian in arbitrary dimensions.

    Uses a second-order finite-difference kinetic operator and a
    diagonal potential-energy operator.

    Homogeneous Dirichlet boundary conditions are assumed.
    The Hamiltonian acts on interior grid points only.

    Coordinates are provided in the order (x, y, z, ...).
    """
    _validate_physical_parameters(mass, hbar)

    potential.validate_coordinates(coordinates)
    coordinates = tuple(
        np.asarray(axis, dtype=float) for axis in coordinates
    )

    if potential.is_spatially_constant:
        shape = tuple(axis.size for axis in reversed(coordinates))
        potential_values = np.full(shape, potential.values)
    else:
        potential_values = potential.values

    interior_slices = tuple(
        slice(1, -1) for _ in coordinates
    )

    interior_potential = potential_values[interior_slices].ravel()

    laplacian = laplacian_nd(coordinates)

    kinetic = -(hbar**2 / (2.0 * mass)) * laplacian

    potential_operator = diags(
        interior_potential,
        offsets=0,
        format="csr",
    )

    return kinetic + potential_operator
