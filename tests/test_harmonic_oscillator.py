
import numpy as np
from scipy.integrate import trapezoid
from scipy.sparse.linalg import eigsh

from quantumforge.dynamics import BohmianTrajectoryIntegrator
from quantumforge.numerics.operators import hamiltonian_1d
from quantumforge.numerics.schrodinger import SchrodingerSolver
from quantumforge.physics.potential import Potential
from quantumforge.physics.wavefunction import WaveFunction


def harmonic_oscillator_ground_state(x: np.ndarray) -> np.ndarray:
    """Analytical ground state for m = hbar = omega = 1."""
    return np.pi ** (-0.25) * np.exp(-(x**2) / 2.0)


def test_harmonic_oscillator_ground_state_is_stationary():
    x = np.linspace(-8.0, 8.0, 2001)
    dt = 0.01
    steps = 100

    psi_0 = harmonic_oscillator_ground_state(x)
    wavefunction = WaveFunction(x=x, psi=psi_0)
    potential = Potential(values=0.5 * x**2, x=x)

    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=dt,
        hbar=1.0,
    )

    final_state = solver.evolve(wavefunction, steps)

    initial_density = np.abs(psi_0) ** 2
    final_density = np.abs(final_state.psi) ** 2

    np.testing.assert_allclose(
        final_density,
        initial_density,
        rtol=2e-4,
        atol=1e-8,
    )


def test_harmonic_oscillator_ground_state_preserves_probability():
    x = np.linspace(-8.0, 8.0, 2001)
    dt = 0.01
    steps = 100

    psi_0 = harmonic_oscillator_ground_state(x)
    wavefunction = WaveFunction(x=x, psi=psi_0)
    potential = Potential(values=0.5 * x**2, x=x)

    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=dt,
        hbar=1.0,
    )

    final_state = solver.evolve(wavefunction, steps)

    initial_probability = trapezoid(np.abs(psi_0) ** 2, x)
    final_probability = trapezoid(np.abs(final_state.psi) ** 2, x)

    np.testing.assert_allclose(
        final_probability,
        initial_probability,
        rtol=1e-10,
        atol=1e-12,
    )


def test_harmonic_oscillator_ground_state_has_stationary_trajectories():
    x = np.linspace(-8.0, 8.0, 2001)
    dt = 0.01
    steps = 100

    psi_0 = harmonic_oscillator_ground_state(x)
    wavefunction = WaveFunction(x=x, psi=psi_0)

    initial_positions = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])

    integrator = BohmianTrajectoryIntegrator(
        mass=1.0,
        dt=dt,
        hbar=1.0,
    )

    intervals = [(wavefunction, wavefunction, wavefunction) for _ in range(steps)]
    for position in initial_positions[:, 0]:
        trajectory = integrator.integrate(
            np.array([position]),
            intervals,
        )

        np.testing.assert_allclose(
            trajectory,
            position,
            rtol=0.0,
            atol=1e-10,
        )


def test_harmonic_oscillator_first_excited_state_is_stationary():
    x = np.linspace(-8.0, 8.0, 2001)
    dt = 0.01
    steps = 100

    psi_1 = np.sqrt(2.0) * x * harmonic_oscillator_ground_state(x)
    wavefunction = WaveFunction(x=x, psi=psi_1)
    potential = Potential(values=0.5 * x**2, x=x)

    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=dt,
        hbar=1.0,
    )

    final_state = solver.evolve(wavefunction, steps)

    np.testing.assert_allclose(
        np.abs(final_state.psi) ** 2,
        np.abs(psi_1) ** 2,
        rtol=2e-4,
        atol=1e-8,
    )


def test_harmonic_oscillator_first_excited_state_preserves_probability():
    x = np.linspace(-8.0, 8.0, 2001)
    dt = 0.01
    steps = 100

    psi_1 = np.sqrt(2.0) * x * harmonic_oscillator_ground_state(x)
    wavefunction = WaveFunction(x=x, psi=psi_1)
    potential = Potential(values=0.5 * x**2, x=x)

    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=dt,
        hbar=1.0,
    )

    final_state = solver.evolve(wavefunction, steps)

    initial_probability = trapezoid(np.abs(psi_1) ** 2, x)
    final_probability = trapezoid(np.abs(final_state.psi) ** 2, x)

    np.testing.assert_allclose(
        final_probability,
        initial_probability,
        rtol=1e-10,
        atol=1e-12,
    )


def test_harmonic_oscillator_ground_state_energy():
    x = np.linspace(-8.0, 8.0, 2001)
    psi_0 = harmonic_oscillator_ground_state(x)
    potential = Potential(values=0.5 * x**2, x=x)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    interior_psi = psi_0[1:-1]
    energy = np.vdot(interior_psi, hamiltonian @ interior_psi)
    energy /= np.vdot(interior_psi, interior_psi)

    np.testing.assert_allclose(
        energy.real,
        0.5,
        rtol=1e-5,
        atol=1e-8,
    )


def test_harmonic_oscillator_first_excited_state_energy():
    x = np.linspace(-8.0, 8.0, 2001)
    psi_1 = np.sqrt(2.0) * x * harmonic_oscillator_ground_state(x)
    potential = Potential(values=0.5 * x**2, x=x)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    interior_psi = psi_1[1:-1]
    energy = np.vdot(interior_psi, hamiltonian @ interior_psi)
    energy /= np.vdot(interior_psi, interior_psi)

    np.testing.assert_allclose(
        energy.real,
        1.5,
        rtol=1e-5,
        atol=1e-8,
    )

#tests harmonic
def test_harmonic_oscillator_ground_state_is_hamiltonian_eigenstate():
    x = np.linspace(-8.0, 8.0, 2001)
    psi_0 = harmonic_oscillator_ground_state(x)
    potential = Potential(values=0.5 * x**2, x=x)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    interior_psi = psi_0[1:-1]
    residual = hamiltonian @ interior_psi - 0.5 * interior_psi

    relative_residual = np.linalg.norm(residual) / np.linalg.norm(interior_psi)

    assert relative_residual < 3e-5


def test_harmonic_oscillator_first_excited_state_is_hamiltonian_eigenstate():
    x = np.linspace(-8.0, 8.0, 2001)
    psi_1 = np.sqrt(2.0) * x * harmonic_oscillator_ground_state(x)
    potential = Potential(values=0.5 * x**2, x=x)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    interior_psi = psi_1[1:-1]
    residual = hamiltonian @ interior_psi - 1.5 * interior_psi

    relative_residual = np.linalg.norm(residual) / np.linalg.norm(interior_psi)

    assert relative_residual < 3e-5


def test_harmonic_oscillator_ground_state_energy_converges_quadratically():
    def calculate_energy(number_of_points):
        x = np.linspace(-8.0, 8.0, number_of_points)
        psi_0 = harmonic_oscillator_ground_state(x)
        potential = Potential(values=0.5 * x**2, x=x)

        hamiltonian = hamiltonian_1d(
            x=x,
            mass=1.0,
            potential=potential,
            hbar=1.0,
        )

        interior_psi = psi_0[1:-1]
        energy = np.vdot(interior_psi, hamiltonian @ interior_psi)
        energy /= np.vdot(interior_psi, interior_psi)

        return energy.real

    energy_coarse = calculate_energy(101)
    energy_medium = calculate_energy(201)
    energy_fine = calculate_energy(401)

    error_coarse = abs(energy_coarse - 0.5)
    error_medium = abs(energy_medium - 0.5)
    error_fine = abs(energy_fine - 0.5)

    assert error_coarse > error_medium > error_fine

    assert error_coarse / error_medium > 3.5
    assert error_medium / error_fine > 3.5


def test_harmonic_oscillator_ground_state_temporal_convergence():
    x = np.linspace(-8.0, 8.0, 2001)
    psi_0 = harmonic_oscillator_ground_state(x)
    potential = Potential(values=0.5 * x**2, x=x)

    def evolve(dt, steps):
        wavefunction = WaveFunction(x=x, psi=psi_0)

        solver = SchrodingerSolver(
            mass=1.0,
            potential=potential,
            dt=dt,
            hbar=1.0,
        )

        return solver.evolve(wavefunction, steps)

    final_coarse = evolve(dt=0.02, steps=50)
    final_fine = evolve(dt=0.01, steps=100)

    initial_density = np.abs(psi_0) ** 2

    error_coarse = np.linalg.norm(
        np.abs(final_coarse.psi) ** 2 - initial_density
    )
    error_fine = np.linalg.norm(
        np.abs(final_fine.psi) ** 2 - initial_density
    )

    assert error_fine < error_coarse


def test_harmonic_oscillator_first_excited_state_temporal_convergence():
    x = np.linspace(-8.0, 8.0, 2001)
    psi_1 = np.sqrt(2.0) * x * harmonic_oscillator_ground_state(x)
    potential = Potential(values=0.5 * x**2, x=x)

    def evolve(dt, steps):
        wavefunction = WaveFunction(x=x, psi=psi_1)

        solver = SchrodingerSolver(
            mass=1.0,
            potential=potential,
            dt=dt,
            hbar=1.0,
        )

        return solver.evolve(wavefunction, steps)

    final_coarse = evolve(dt=0.02, steps=50)
    final_medium = evolve(dt=0.01, steps=100)
    final_fine = evolve(dt=0.005, steps=200)

    error_coarse = np.linalg.norm(
        final_coarse.psi - final_fine.psi
    )
    error_medium = np.linalg.norm(
        final_medium.psi - final_fine.psi
    )

    assert error_medium < error_coarse


def test_harmonic_oscillator_ground_state_preserves_energy():
    x = np.linspace(-8.0, 8.0, 2001)
    psi_0 = harmonic_oscillator_ground_state(x)
    potential = Potential(values=0.5 * x**2, x=x)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=0.01,
        hbar=1.0,
    )

    initial_state = WaveFunction(x=x, psi=psi_0)
    final_state = solver.evolve(initial_state, 100)

    initial_psi = psi_0[1:-1]
    final_psi = final_state.psi[1:-1]

    initial_energy = np.vdot(
        initial_psi,
        hamiltonian @ initial_psi,
    ) / np.vdot(initial_psi, initial_psi)

    final_energy = np.vdot(
        final_psi,
        hamiltonian @ final_psi,
    ) / np.vdot(final_psi, final_psi)

    np.testing.assert_allclose(
        final_energy.real,
        initial_energy.real,
        rtol=1e-10,
        atol=1e-12,
    )


def test_harmonic_oscillator_first_excited_state_preserves_energy():
    x = np.linspace(-8.0, 8.0, 2001)
    psi_1 = np.sqrt(2.0) * x * harmonic_oscillator_ground_state(x)
    potential = Potential(values=0.5 * x**2, x=x)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    solver = SchrodingerSolver(
        mass=1.0,
        potential=potential,
        dt=0.01,
        hbar=1.0,
    )

    initial_state = WaveFunction(x=x, psi=psi_1)
    final_state = solver.evolve(initial_state, 100)

    initial_psi = psi_1[1:-1]
    final_psi = final_state.psi[1:-1]

    initial_energy = np.vdot(
        initial_psi,
        hamiltonian @ initial_psi,
    ) / np.vdot(initial_psi, initial_psi)

    final_energy = np.vdot(
        final_psi,
        hamiltonian @ final_psi,
    ) / np.vdot(final_psi, final_psi)

    np.testing.assert_allclose(
        final_energy.real,
        initial_energy.real,
        rtol=1e-10,
        atol=1e-12,
    )


def test_harmonic_oscillator_ground_and_first_excited_states_are_orthogonal():
    x = np.linspace(-8.0, 8.0, 2001)

    psi_0 = harmonic_oscillator_ground_state(x)
    psi_1 = np.sqrt(2.0) * x * psi_0

    overlap = trapezoid(np.conjugate(psi_0) * psi_1, x)

    np.testing.assert_allclose(
        overlap,
        0.0,
        atol=1e-12,
    )


def test_harmonic_oscillator_analytical_states_are_normalized():
    x = np.linspace(-8.0, 8.0, 2001)

    psi_0 = harmonic_oscillator_ground_state(x)
    psi_1 = np.sqrt(2.0) * x * psi_0

    probability_0 = trapezoid(np.abs(psi_0) ** 2, x)
    probability_1 = trapezoid(np.abs(psi_1) ** 2, x)

    np.testing.assert_allclose(probability_0, 1.0, atol=1e-12)
    np.testing.assert_allclose(probability_1, 1.0, atol=1e-12)


def test_harmonic_oscillator_ground_state_temporal_convergence_order():
    x = np.linspace(-8.0, 8.0, 2001)
    potential = Potential(values=0.5 * x**2, x=x)

    hamiltonian = hamiltonian_1d(
        x=x,
        mass=1.0,
        potential=potential,
        hbar=1.0,
    )

    energies, eigenvectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        tol=1e-12,
    )

    energy = energies[0]

    psi = np.zeros(x.size, dtype=complex)
    psi[1:-1] = eigenvectors[:, 0]

    final_time = 1.0
    errors = []

    for dt in (0.04, 0.02, 0.01, 0.005):
        solver = SchrodingerSolver(
            mass=1.0,
            potential=potential,
            dt=dt,
            hbar=1.0,
        )

        final_state = solver.evolve(
            WaveFunction(x=x, psi=psi),
            round(final_time / dt),
        )

        exact_state = psi * np.exp(-1j * energy * final_time)

        error = np.linalg.norm(
            final_state.psi - exact_state
        ) / np.linalg.norm(exact_state)

        errors.append(error)

    convergence_ratios = np.array(errors[:-1]) / np.array(errors[1:])

    np.testing.assert_allclose(
        convergence_ratios,
        4.0,
        rtol=1e-3,
    )
