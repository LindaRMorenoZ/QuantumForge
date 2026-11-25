
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse.linalg._dsolve import factorized

import quantumforge.numerics.schrodinger as schrodinger_module
from quantumforge.numerics.schrodinger import SchrodingerSolver
from quantumforge.physics.potential import Potential
from quantumforge.physics.wavefunction import WaveFunction


def create_gaussian_wavefunction():
    x = np.linspace(-10.0, 10.0, 1001)

    sigma = 1.0
    k0 = 2.0

    psi = np.exp(-(x**2) / (4.0 * sigma**2)) * np.exp(1j * k0 * x)

    psi[0] = 0.0
    psi[-1] = 0.0

    dx = x[1] - x[0]
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    psi /= norm

    return WaveFunction(x=x, psi=psi)


def test_schrodinger_solver_preserves_wavefunction_grid():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    result = solver.step(wavefunction)

    assert np.array_equal(result.x, wavefunction.x)
    assert result.psi.shape == wavefunction.psi.shape


def test_schrodinger_solver_preserves_dirichlet_boundaries():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    result = solver.evolve(wavefunction, steps=10)

    assert result.psi[0] == 0.0
    assert result.psi[-1] == 0.0


def test_crank_nicolson_preserves_probability_norm():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.5 * wavefunction.x**2,
                            x=wavefunction.x),
        dt=0.001,
        hbar=1.0,
    )

    initial_norm = np.sum(np.abs(wavefunction.psi) ** 2)
    result = solver.evolve(wavefunction, steps=100)
    final_norm = np.sum(np.abs(result.psi) ** 2)

    assert np.isclose(final_norm, initial_norm, rtol=1e-10, atol=1e-12)


def test_evolve_zero_steps_returns_original_wavefunction():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    result = solver.evolve(wavefunction, steps=0)

    assert result is wavefunction


def test_evolve_multiple_steps_matches_repeated_single_steps():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    repeated = wavefunction

    for _ in range(5):
        repeated = solver.step(repeated)

    evolved = solver.evolve(wavefunction, steps=5)

    assert np.allclose(evolved.psi, repeated.psi)


def test_solver_rejects_two_dimensional_wavefunction():
    x = np.linspace(-5.0, 5.0, 21)
    y = np.linspace(-5.0, 5.0, 21)

    wavefunction = WaveFunction(
        x=x,
        y=y,
        psi=np.ones((21, 21), dtype=complex),
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    with pytest.raises(ValueError, match="1D only"):
        solver.step(wavefunction)


@pytest.mark.parametrize(
    ("mass", "potential", "dt", "hbar"),
    [
        (0.0, Potential(values=0.0), 0.001, 1.0),
        (1.0, Potential(values=0.0), 0.0, 1.0),
        (1.0, Potential(values=0.0), 0.001, 0.0),
    ],
)
def test_solver_rejects_invalid_parameters(mass, potential, dt, hbar):
    with pytest.raises(ValueError):
        SchrodingerSolver(
            mass=mass,
            potential=potential,
            dt=dt,
            hbar=hbar,
        )


@pytest.mark.parametrize("steps", [-1, 1.5, "10"])
def test_evolve_rejects_invalid_step_counts(steps):
    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    with pytest.raises(ValueError):
        solver.evolve(create_gaussian_wavefunction(), steps=steps)


def test_evolve_factorizes_matrix_only_once():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(0.0),
        dt=0.001,
        hbar=1.0,
    )

    with patch.object(
        schrodinger_module,
        "factorized",
        wraps=schrodinger_module.factorized,
    ) as factorized_mock:
        solver.evolve(wavefunction, steps=5)

    assert factorized_mock.call_count == 1


def test_crank_nicolson_matches_discrete_eigenstate_evolution():
    from scipy.linalg import eigh

    from quantumforge.numerics.operators import hamiltonian_1d

    x = np.linspace(-5.0, 5.0, 101)
    mass = 1.0
    hbar = 1.0
    dt = 0.001
    steps = 20

    solver = SchrodingerSolver(
        mass=mass,
        potential=Potential(0.0),
        dt=dt,
        hbar=hbar,
    )

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=mass,
        potential=Potential(0.0),
        hbar=hbar,
    ).toarray()

    energies, eigenvectors = eigh(hamiltonian)

    energy = energies[0]
    eigenstate = eigenvectors[:, 0].astype(complex)

    psi = np.zeros(x.size, dtype=complex)
    psi[1:-1] = eigenstate

    wavefunction = WaveFunction(x=x, psi=psi)

    evolved = solver.evolve(wavefunction, steps=steps)

    coefficient = 1j * energy * dt / (2.0 * hbar)
    single_step_factor = (1.0 - coefficient) / (1.0 + coefficient)
    expected_interior = single_step_factor**steps * eigenstate

    assert np.allclose(
        evolved.psi[1:-1],
        expected_interior,
        rtol=1e-10,
        atol=1e-12,
    )


def test_crank_nicolson_has_second_order_temporal_convergence():
    from scipy.linalg import eigh

    from quantumforge.numerics.operators import hamiltonian_1d

    x = np.linspace(-5.0, 5.0, 101)
    mass = 1.0
    hbar = 1.0
    total_time = 0.1

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=mass,
        potential=Potential(0.0),
        hbar=hbar,
    ).toarray()

    energies, eigenvectors = eigh(hamiltonian)

    energy = energies[0]
    eigenstate = eigenvectors[:, 0].astype(complex)

    psi = np.zeros(x.size, dtype=complex)
    psi[1:-1] = eigenstate

    wavefunction = WaveFunction(x=x, psi=psi)

    def evolution_error(dt):
        steps = round(total_time / dt)

        solver = SchrodingerSolver(
            mass=mass,
            potential=Potential(0.0),
            dt=dt,
            hbar=hbar,
        )

        evolved = solver.evolve(wavefunction, steps=steps)

        exact = np.exp(-1j * energy * total_time / hbar) * eigenstate

        return np.linalg.norm(evolved.psi[1:-1] - exact)

    coarse_error = evolution_error(0.01)
    fine_error = evolution_error(0.005)

    assert coarse_error > 0.0
    assert fine_error < coarse_error
    assert np.isclose(
        coarse_error / fine_error,
        4.0,
        rtol=0.02,
    )    


def test_schrodinger_solver_is_available_from_numerics_package():
    from quantumforge.numerics import SchrodingerSolver as PublicSolver

    assert PublicSolver is SchrodingerSolver  


def test_two_half_steps_match_one_full_step():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    rhs_half, solve_half = solver._prepare_evolution(
        wavefunction,
        dt=solver.dt / 2.0,
    )

    midpoint = solver._advance(
        wavefunction,
        rhs_half,
        solve_half,
    )

    final_from_half_steps = solver._advance(
        midpoint,
        rhs_half,
        solve_half,
    )

    final_from_full_step = solver.step(wavefunction)

    np.testing.assert_allclose(
        final_from_half_steps.psi,
        final_from_full_step.psi,
        rtol=1e-4,
        atol=2e-9,
    )


def test_evolve_states_returns_all_time_steps():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    states = solver.evolve_states(wavefunction, steps=5)

    assert len(states) == 6
    assert states[0] is wavefunction

    expected = solver.evolve(wavefunction, steps=5)

    np.testing.assert_allclose(
        states[-1].psi,
        expected.psi,
        rtol=1e-12,
        atol=1e-14,
    )    


def test_midpoint_matches_half_time_step_evolution():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    rhs_half, solve_half = solver._prepare_evolution(
        wavefunction,
        dt=solver.dt / 2.0,
    )

    midpoint = solver._midpoint(
        wavefunction,
        rhs_half,
        solve_half,
    )

    expected = solver._advance(
        wavefunction,
        rhs_half,
        solve_half,
    )

    np.testing.assert_allclose(
        midpoint.psi,
        expected.psi,
        rtol=1e-13,
        atol=1e-15,
    )


def test_evolve_states_factorizes_matrix_only_once():
    from unittest.mock import patch

    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    with patch(
        "quantumforge.numerics.schrodinger.factorized",
        wraps=factorized,
    ) as mocked_factorized:
        states = solver.evolve_states(wavefunction, steps=5)

    assert len(states) == 6
    assert mocked_factorized.call_count == 1


def test_evolve_with_midpoints_returns_correct_number_of_intervals():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    intervals = solver.evolve_with_midpoints(
        wavefunction,
        steps=5,
    )

    assert len(intervals) == 5

    for initial, midpoint, final in intervals:
        assert isinstance(initial, WaveFunction)
        assert isinstance(midpoint, WaveFunction)
        assert isinstance(final, WaveFunction)

    for first, second in zip(intervals[:-1], intervals[1:], strict=False):
        np.testing.assert_array_equal(
            first[2].psi,
            second[0].psi,
        )


def test_evolve_with_midpoints_uses_correct_time_steps():
    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    intervals = solver.evolve_with_midpoints(
        wavefunction,
        steps=2,
    )

    initial, midpoint, final = intervals[0]

    rhs_half, solve_half = solver._prepare_evolution(
        initial,
        dt=solver.dt / 2.0,
    )
    expected_midpoint = solver._advance(
        initial,
        rhs_half,
        solve_half,
    )

    np.testing.assert_allclose(
        midpoint.psi,
        expected_midpoint.psi,
        rtol=1e-13,
        atol=1e-15,
    )

    expected_final = solver.step(initial)

    np.testing.assert_allclose(
        final.psi,
        expected_final.psi,
        rtol=1e-13,
        atol=1e-15,
    )


def test_evolve_with_midpoints_factorizes_each_time_step_once():
    from unittest.mock import patch

    wavefunction = create_gaussian_wavefunction()

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    with patch(
        "quantumforge.numerics.schrodinger.factorized",
        wraps=factorized,
    ) as mocked_factorized:
        intervals = solver.evolve_with_midpoints(
            wavefunction,
            steps=5,
        )

    assert len(intervals) == 5
    assert mocked_factorized.call_count == 2


