from unittest.mock import patch

import numpy as np
import pytest
from analytic_references import free_gaussian_wavefunction
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg._dsolve import factorized

import quantumforge.numerics.schrodinger as schrodinger_module
from quantumforge.numerics.schrodinger import SchrodingerSolver
from quantumforge.physics.constants import SI_CONSTANTS
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
        potential=Potential(values=0.5 * wavefunction.x**2, x=wavefunction.x),
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


def test_solver_accepts_two_dimensional_wavefunction():
    x = np.linspace(-5.0, 5.0, 21)
    y = np.linspace(-5.0, 5.0, 21)

    psi = np.ones((21, 21), dtype=complex)
    psi[[0, -1], :] = 0.0
    psi[:, [0, -1]] = 0.0

    wavefunction = WaveFunction(
        x=x,
        y=y,
        psi=psi,
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.001,
        hbar=1.0,
    )

    evolved = solver.step(wavefunction)

    assert evolved.ndim == 2
    assert evolved.psi.shape == wavefunction.psi.shape
    assert not np.allclose(evolved.psi, wavefunction.psi)


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


def test_free_gaussian_evolution_preserves_probability_in_si_units():

    mass = SI_CONSTANTS.electron_mass
    hbar = SI_CONSTANTS.hbar
    sigma = 1e-9

    characteristic_time = 2.0 * mass * sigma**2 / hbar
    dt = characteristic_time / 100.0

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
    initial_probability = np.trapezoid(initial.density, x)

    solver = SchrodingerSolver(
        mass=mass,
        potential=Potential(values=0.0),
        dt=dt,
        hbar=hbar,
    )

    final = solver.evolve(initial, steps=100)
    final_probability = np.trapezoid(final.density, x)

    assert final_probability == pytest.approx(
        initial_probability,
        rel=1e-8,
    )


def test_free_gaussian_evolution_matches_analytical_width_in_si_units():
    mass = SI_CONSTANTS.electron_mass
    hbar = SI_CONSTANTS.hbar
    sigma = 1e-9

    characteristic_time = 2.0 * mass * sigma**2 / hbar
    dt = characteristic_time / 100.0

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

    final = solver.evolve(initial, steps=100)

    probability = np.trapezoid(final.density, x)
    mean = np.trapezoid(x * final.density, x) / probability
    variance = np.trapezoid((x - mean) ** 2 * final.density, x) / probability

    numerical_width = np.sqrt(variance)
    analytical_width = np.sqrt(2.0) * sigma

    assert numerical_width == pytest.approx(
        analytical_width,
        rel=1e-3,
    )


def test_free_gaussian_evolution_matches_analytical_wavefunction():
    mass = SI_CONSTANTS.electron_mass
    hbar = SI_CONSTANTS.hbar
    sigma = 1e-9

    characteristic_time = 2.0 * mass * sigma**2 / hbar
    dt = characteristic_time / 100.0

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

    numerical = solver.evolve(initial, steps=100)
    analytical = free_gaussian_wavefunction(
        x=x,
        time=characteristic_time,
        mass=mass,
        sigma=sigma,
    )

    overlap = np.trapezoid(
        np.conjugate(analytical.psi) * numerical.psi,
        x,
    )
    phase = np.angle(overlap)

    aligned_numerical = numerical.psi * np.exp(-1j * phase)

    error = np.sqrt(
        np.trapezoid(
            np.abs(aligned_numerical - analytical.psi) ** 2,
            x,
        )
    )
    reference_norm = np.sqrt(np.trapezoid(analytical.density, x))

    relative_error = error / reference_norm

    assert relative_error < 1e-3


def test_free_gaussian_evolution_has_second_order_temporal_convergence():
    mass = SI_CONSTANTS.electron_mass
    hbar = SI_CONSTANTS.hbar
    sigma = 1e-9

    characteristic_time = 2.0 * mass * sigma**2 / hbar
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

    analytical = free_gaussian_wavefunction(
        x=x,
        time=characteristic_time,
        mass=mass,
        sigma=sigma,
    )

    errors = []

    for subdivisions in (50, 100, 200):
        dt = characteristic_time / subdivisions

        solver = SchrodingerSolver(
            mass=mass,
            potential=Potential(values=0.0),
            dt=dt,
            hbar=hbar,
        )

        numerical = solver.evolve(initial, steps=subdivisions)

        overlap = np.trapezoid(
            np.conjugate(analytical.psi) * numerical.psi,
            x,
        )
        phase = np.angle(overlap)
        aligned = numerical.psi * np.exp(-1j * phase)

        error = np.sqrt(
            np.trapezoid(
                np.abs(aligned - analytical.psi) ** 2,
                x,
            )
        )
        errors.append(error)

    coarse_ratio = errors[0] / errors[1]
    fine_ratio = errors[1] / errors[2]

    assert coarse_ratio > 3.0
    assert fine_ratio > 3.0


def test_free_gaussian_evolution_has_second_order_spatial_convergence():
    mass = SI_CONSTANTS.electron_mass
    hbar = SI_CONSTANTS.hbar
    sigma = 1e-9

    characteristic_time = 2.0 * mass * sigma**2 / hbar
    subdivisions = 1600
    dt = characteristic_time / subdivisions

    errors = []

    for points in (1201, 2401, 4801):
        x = np.linspace(-12.0 * sigma, 12.0 * sigma, points)

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

        numerical = solver.evolve(initial, steps=subdivisions)

        analytical = free_gaussian_wavefunction(
            x=x,
            time=characteristic_time,
            mass=mass,
            sigma=sigma,
        )

        overlap = np.trapezoid(
            np.conjugate(analytical.psi) * numerical.psi,
            x,
        )
        phase = np.angle(overlap)
        aligned = numerical.psi * np.exp(-1j * phase)

        error = np.sqrt(np.trapezoid(np.abs(aligned - analytical.psi) ** 2, x))
        errors.append(error)

    coarse_ratio = errors[0] / errors[1]
    fine_ratio = errors[1] / errors[2]

    assert coarse_ratio > 3.0
    assert fine_ratio > 3.0


def test_schrodinger_solver_preserves_probability_in_2d():
    x = np.linspace(-5.0, 5.0, 51)
    y = np.linspace(-5.0, 5.0, 61)

    X, Y = np.meshgrid(x, y, indexing="xy")

    sigma = 1.0
    psi = np.exp(-(X**2 + Y**2) / (4.0 * sigma**2))
    psi = psi.astype(complex)

    psi[[0, -1], :] = 0.0
    psi[:, [0, -1]] = 0.0

    initial = WaveFunction(
        coordinates=(x, y),
        psi=psi,
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.01,
        hbar=1.0,
    )

    numerical = solver.evolve(initial, steps=20)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    initial_probability = np.sum(np.abs(initial.psi) ** 2) * dx * dy
    final_probability = np.sum(np.abs(numerical.psi) ** 2) * dx * dy

    np.testing.assert_allclose(
        final_probability,
        initial_probability,
        rtol=1e-12,
        atol=1e-12,
    )


def test_schrodinger_solver_evolves_wavefunction_in_2d():
    x = np.linspace(-5.0, 5.0, 51)
    y = np.linspace(-5.0, 5.0, 61)

    X, Y = np.meshgrid(x, y, indexing="xy")

    psi = np.exp(-(X**2 + Y**2) / 4.0).astype(complex)
    psi[[0, -1], :] = 0.0
    psi[:, [0, -1]] = 0.0

    initial = WaveFunction(
        coordinates=(x, y),
        psi=psi,
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.01,
        hbar=1.0,
    )

    numerical = solver.evolve(initial, steps=20)

    assert numerical.ndim == 2
    assert numerical.psi.shape == initial.psi.shape
    assert not np.allclose(numerical.psi, initial.psi)


def test_schrodinger_solver_evolve_with_midpoints_in_2d():
    x = np.linspace(-5.0, 5.0, 31)
    y = np.linspace(-5.0, 5.0, 41)

    X, Y = np.meshgrid(x, y, indexing="xy")

    psi = np.exp(-(X**2 + Y**2) / 4.0).astype(complex)
    psi[[0, -1], :] = 0.0
    psi[:, [0, -1]] = 0.0

    initial = WaveFunction(
        coordinates=(x, y),
        psi=psi,
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.01,
        hbar=1.0,
    )

    intervals = solver.evolve_with_midpoints(
        initial,
        steps=3,
    )

    assert len(intervals) == 3

    for interval in intervals:
        assert interval.initial_state.ndim == 2
        assert interval.midpoint_state.ndim == 2
        assert interval.final_state.ndim == 2

        assert interval.initial_state.psi.shape == psi.shape
        assert interval.midpoint_state.psi.shape == psi.shape
        assert interval.final_state.psi.shape == psi.shape


def test_schrodinger_solver_evolve_states_in_2d():
    x = np.linspace(-5.0, 5.0, 31)
    y = np.linspace(-5.0, 5.0, 41)

    X, Y = np.meshgrid(x, y, indexing="xy")

    psi = np.exp(-(X**2 + Y**2) / 4.0).astype(complex)
    psi[[0, -1], :] = 0.0
    psi[:, [0, -1]] = 0.0

    initial = WaveFunction(
        coordinates=(x, y),
        psi=psi,
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=Potential(values=0.0),
        dt=0.01,
        hbar=1.0,
    )

    states = solver.evolve_states(initial, steps=3)

    assert len(states) == 4

    for state in states:
        assert state.ndim == 2
        assert state.psi.shape == psi.shape

    assert states[0] is initial
    assert not np.allclose(states[-1].psi, initial.psi)


def test_schrodinger_solver_preserves_probability_with_2d_potential():
    x = np.linspace(-5.0, 5.0, 41)
    y = np.linspace(-5.0, 5.0, 51)

    X, Y = np.meshgrid(x, y, indexing="xy")

    psi = np.exp(-(X**2 + Y**2) / 4.0).astype(complex)
    psi[[0, -1], :] = 0.0
    psi[:, [0, -1]] = 0.0

    initial = WaveFunction(
        coordinates=(x, y),
        psi=psi,
    )

    potential = Potential(
        values=0.5 * (X**2 + Y**2),
        coordinates=(x, y),
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=0.01,
        hbar=1.0,
    )

    numerical = solver.evolve(initial, steps=20)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    initial_probability = np.sum(np.abs(initial.psi) ** 2) * dx * dy
    final_probability = np.sum(np.abs(numerical.psi) ** 2) * dx * dy

    np.testing.assert_allclose(
        final_probability,
        initial_probability,
        rtol=1e-12,
        atol=1e-12,
    )


def test_crank_nicolson_has_second_order_temporal_convergence_in_2d():
    from scipy.linalg import eigh

    from quantumforge.numerics.operators import hamiltonian_nd

    x = np.linspace(-4.0, 4.0, 21)
    y = np.linspace(-4.0, 4.0, 21)

    mass = 1.0
    hbar = 1.0
    total_time = 0.1

    X, Y = np.meshgrid(x, y, indexing="xy")

    potential = Potential(
        values=0.5 * (X**2 + Y**2),
        coordinates=(x, y),
    )

    hamiltonian = hamiltonian_nd(
        coordinates=(x, y),
        mass=mass,
        potential=potential,
        hbar=hbar,
    ).toarray()

    energies, eigenvectors = eigh(hamiltonian)

    energy = energies[0]
    eigenstate = eigenvectors[:, 0].astype(complex)

    psi = np.zeros((y.size, x.size), dtype=complex)
    psi[1:-1, 1:-1] = eigenstate.reshape(
        (y.size - 2, x.size - 2)
    )

    wavefunction = WaveFunction(
        coordinates=(x, y),
        psi=psi,
    )

    def evolution_error(dt):
        steps = round(total_time / dt)

        solver = SchrodingerSolver(
            mass=mass,
            potential=potential,
            dt=dt,
            hbar=hbar,
        )

        evolved = solver.evolve(wavefunction, steps=steps)

        exact = np.exp(-1j * energy * total_time / hbar) * eigenstate

        numerical_interior = evolved.psi[1:-1, 1:-1].ravel()

        return np.linalg.norm(numerical_interior - exact)

    coarse_error = evolution_error(0.01)
    fine_error = evolution_error(0.005)

    assert coarse_error > 0.0
    assert fine_error < coarse_error
    assert np.isclose(
        coarse_error / fine_error,
        4.0,
        rtol=0.02,
    )


def test_harmonic_oscillator_2d_ground_state_has_second_order_spatial_convergence():
    from scipy.linalg import eigh

    from quantumforge.numerics.operators import hamiltonian_nd

    errors = []

    for points in (17, 33, 65):
        x = np.linspace(-5.0, 5.0, points)
        y = np.linspace(-5.0, 5.0, points)

        X, Y = np.meshgrid(x, y, indexing="xy")

        potential = Potential(
            values=0.5 * (X**2 + Y**2),
            coordinates=(x, y),
        )

        hamiltonian = hamiltonian_nd(
            coordinates=(x, y),
            mass=1.0,
            potential=potential,
            hbar=1.0,
        ).toarray()

        energies, _ = eigh(hamiltonian)

        errors.append(abs(energies[0] - 1.0))

    coarse_ratio = errors[0] / errors[1]
    fine_ratio = errors[1] / errors[2]

    assert np.isclose(coarse_ratio, 4.0, rtol=0.15)
    assert np.isclose(fine_ratio, 4.0, rtol=0.15)

def test_harmonic_oscillator_2d_ground_state_matches_analytical_solution():
    from scipy.sparse.linalg import eigsh

    from quantumforge.numerics.operators import hamiltonian_nd

    x = np.linspace(-5.0, 5.0, 65)
    y = np.linspace(-5.0, 5.0, 65)

    X, Y = np.meshgrid(x, y, indexing="xy")

    potential = Potential(
        values=0.5 * (X**2 + Y**2),
        coordinates=(x, y),
    )

    hamiltonian = hamiltonian_nd(
        coordinates=(x, y),
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    energies, eigenvectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        tol=1e-10,
    )

    numerical = eigenvectors[:, 0]

    analytical = (
        np.pi ** (-0.5)
        * np.exp(-(X[1:-1, 1:-1] ** 2 + Y[1:-1, 1:-1] ** 2) / 2.0)
    ).ravel()

    numerical /= np.linalg.norm(numerical)
    analytical /= np.linalg.norm(analytical)

    overlap = np.abs(np.vdot(analytical, numerical))

    assert np.isclose(energies[0], 1.0, atol=0.002)
    assert overlap > 0.999


def test_harmonic_oscillator_2d_ground_state_density_is_stationary():
    from scipy.sparse.linalg import eigsh

    from quantumforge.numerics.operators import hamiltonian_nd

    x = np.linspace(-5.0, 5.0, 65)
    y = np.linspace(-5.0, 5.0, 65)

    X, Y = np.meshgrid(x, y, indexing="xy")

    potential = Potential(
        values=0.5 * (X**2 + Y**2),
        coordinates=(x, y),
    )

    hamiltonian = hamiltonian_nd(
        coordinates=(x, y),
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    _, eigenvectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        tol=1e-10,
    )

    psi = np.zeros((y.size, x.size), dtype=complex)
    psi[1:-1, 1:-1] = eigenvectors[:, 0].reshape(
        (y.size - 2, x.size - 2)
    )

    initial = WaveFunction(
        coordinates=(x, y),
        psi=psi,
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=0.01,
        hbar=1.0,
    )

    final = solver.evolve(initial, steps=100)

    np.testing.assert_allclose(
        np.abs(final.psi) ** 2,
        np.abs(initial.psi) ** 2,
        rtol=1e-8,
        atol=1e-10,
    )

#3D
def test_schrodinger_solver_preserves_probability_in_3d():
    x = np.linspace(-5.0, 5.0, 17)
    y = np.linspace(-5.0, 5.0, 17)
    z = np.linspace(-5.0, 5.0, 17)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    psi = np.exp(-(X**2 + Y**2 + Z**2) / 2).astype(complex)

    wavefunction = WaveFunction(
        coordinates=(x, y, z),
        psi=psi,
    )
    potential = Potential(
        values=np.zeros_like(psi.real),
        coordinates=(x, y, z),
    )
    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=0.001,
        hbar=1.0,
    )

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    initial_probability = np.sum(np.abs(wavefunction.psi) ** 2) * dx * dy * dz
    final = solver.evolve(wavefunction, steps=10)

    assert final.ndim == 3
    assert final.psi.shape == wavefunction.psi.shape

    final_probability = np.sum(np.abs(final.psi) ** 2) * dx * dy * dz

    np.testing.assert_allclose(
        final_probability,
        initial_probability,
        rtol=1e-9,
        atol=1e-11,
    )


def test_schrodinger_solver_evolves_wavefunction_in_3d():
    x = np.linspace(-5.0, 5.0, 17)
    y = np.linspace(-5.0, 5.0, 17)
    z = np.linspace(-5.0, 5.0, 17)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    psi = np.exp(-(X**2 + Y**2 + Z**2) / 2).astype(complex)

    wavefunction = WaveFunction(
        coordinates=(x, y, z),
        psi=psi,
    )
    potential = Potential(
        values=np.zeros_like(psi.real),
        coordinates=(x, y, z),
    )
    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=0.01,
        hbar=1.0,
    )

    evolved = solver.evolve(wavefunction, steps=10)

    assert evolved.ndim == 3
    assert evolved.psi.shape == wavefunction.psi.shape
    assert not np.allclose(evolved.psi, wavefunction.psi)


def test_schrodinger_solver_preserves_probability_with_3d_potential():
    x = np.linspace(-5.0, 5.0, 17)
    y = np.linspace(-5.0, 5.0, 17)
    z = np.linspace(-5.0, 5.0, 17)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    psi = np.exp(-(X**2 + Y**2 + Z**2) / 2).astype(complex)
    potential_values = 0.5 * (X**2 + Y**2 + Z**2)

    wavefunction = WaveFunction(
        coordinates=(x, y, z),
        psi=psi,
    )
    potential = Potential(
        values=potential_values,
        coordinates=(x, y, z),
    )
    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=0.001,
        hbar=1.0,
    )

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    initial_probability = (
        np.sum(np.abs(wavefunction.psi) ** 2) * dx * dy * dz
    )
    final = solver.evolve(wavefunction, steps=10)
    final_probability = np.sum(np.abs(final.psi) ** 2) * dx * dy * dz

    np.testing.assert_allclose(
        final_probability,
        initial_probability,
        rtol=1e-9,
        atol=1e-11,
    )


def test_crank_nicolson_has_second_order_temporal_convergence_in_3d():
    from scipy.linalg import eigh

    from quantumforge.numerics.operators import hamiltonian_nd

    x = np.linspace(-4.0, 4.0, 11)
    y = np.linspace(-4.0, 4.0, 11)
    z = np.linspace(-4.0, 4.0, 11)

    mass = 1.0
    hbar = 1.0
    total_time = 0.1

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    potential = Potential(
        values=0.5 * (X**2 + Y**2 + Z**2),
        coordinates=(x, y, z),
    )

    hamiltonian = hamiltonian_nd(
        coordinates=(x, y, z),
        mass=mass,
        potential=potential,
        hbar=hbar,
    ).toarray()

    energies, eigenvectors = eigh(hamiltonian)

    energy = energies[0]
    eigenstate = eigenvectors[:, 0].astype(complex)

    psi = np.zeros((z.size, y.size, x.size), dtype=complex)
    psi[1:-1, 1:-1, 1:-1] = eigenstate.reshape(
        (z.size - 2, y.size - 2, x.size - 2)
    )

    wavefunction = WaveFunction(
        coordinates=(x, y, z),
        psi=psi,
    )

    def evolution_error(dt):
        steps = round(total_time / dt)

        solver = SchrodingerSolver(
            mass=mass,
            potential=potential,
            dt=dt,
            hbar=hbar,
        )

        evolved = solver.evolve(wavefunction, steps=steps)

        exact = np.exp(-1j * energy * total_time / hbar) * eigenstate

        numerical_interior = evolved.psi[1:-1, 1:-1, 1:-1].ravel()

        return np.linalg.norm(numerical_interior - exact)

    coarse_error = evolution_error(0.01)
    fine_error = evolution_error(0.005)

    assert coarse_error > 0.0
    assert fine_error < coarse_error
    assert np.isclose(
        coarse_error / fine_error,
        4.0,
        rtol=0.02,
    )


def test_harmonic_oscillator_3d_ground_state_has_second_order_spatial_convergence():

    from quantumforge.numerics.operators import hamiltonian_nd

    mass = 1.0
    hbar = 1.0
    exact_energy = 1.5

    errors = []

    for points in (17, 33, 65):
        x = np.linspace(-5.0, 5.0, points)
        y = np.linspace(-5.0, 5.0, points)
        z = np.linspace(-5.0, 5.0, points)

        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

        potential = Potential(
            values=0.5 * (X**2 + Y**2 + Z**2),
            coordinates=(x, y, z),
        )

        hamiltonian = hamiltonian_nd(
            coordinates=(x, y, z),
            mass=mass,
            potential=potential,
            hbar=hbar,
        )

        energies, _ = eigsh(
            hamiltonian,
            k=1,
            which="SA",
            tol=1e-10,
        )

        errors.append(abs(energies[0] - exact_energy))

    coarse_ratio = errors[0] / errors[1]
    fine_ratio = errors[1] / errors[2]
    assert np.isclose(coarse_ratio, 4.0, rtol=0.25)
    assert np.isclose(fine_ratio, 4.0, rtol=0.25)



def test_harmonic_oscillator_3d_ground_state_matches_analytical_solution():
    from scipy.sparse.linalg import eigsh

    from quantumforge.numerics.operators import hamiltonian_nd

    x = np.linspace(-5.0, 5.0, 33)
    y = np.linspace(-5.0, 5.0, 33)
    z = np.linspace(-5.0, 5.0, 33)

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    potential = Potential(
        values=0.5 * (X**2 + Y**2 + Z**2),
        coordinates=(x, y, z),
    )

    hamiltonian = hamiltonian_nd(
        coordinates=(x, y, z),
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    energies, eigenvectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        tol=1e-10,
    )

    numerical = eigenvectors[:, 0]

    X_interior = X[1:-1, 1:-1, 1:-1]
    Y_interior = Y[1:-1, 1:-1, 1:-1]
    Z_interior = Z[1:-1, 1:-1, 1:-1]

    analytical = np.pi**(-0.75) * np.exp(
        -(X_interior**2 + Y_interior**2 + Z_interior**2) / 2
    ).ravel()

    numerical /= np.linalg.norm(numerical)
    analytical /= np.linalg.norm(analytical)

    overlap = abs(np.vdot(analytical, numerical))

    assert np.isclose(energies[0], 1.5, atol=0.01)
    assert overlap > 0.99


def test_harmonic_oscillator_3d_ground_state_density_is_stationary():
    from scipy.sparse.linalg import eigsh

    from quantumforge.numerics.operators import hamiltonian_nd

    x = np.linspace(-5.0, 5.0, 17)
    y = np.linspace(-5.0, 5.0, 17)
    z = np.linspace(-5.0, 5.0, 17)

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    potential = Potential(
        values=0.5 * (X**2 + Y**2 + Z**2),
        coordinates=(x, y, z),
    )

    hamiltonian = hamiltonian_nd(
        coordinates=(x, y, z),
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    _, eigenvectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        tol=1e-10,
    )

    psi = np.zeros((z.size, y.size, x.size), dtype=complex)
    psi[1:-1, 1:-1, 1:-1] = eigenvectors[:, 0].reshape(
        (z.size - 2, y.size - 2, x.size - 2)
    )

    initial = WaveFunction(
        coordinates=(x, y, z),
        psi=psi,
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=0.01,
        hbar=1.0,
    )

    final = solver.evolve(initial, steps=100)

    np.testing.assert_allclose(
        np.abs(final.psi) ** 2,
        np.abs(initial.psi) ** 2,
        rtol=1e-8,
        atol=1e-10,
    )
