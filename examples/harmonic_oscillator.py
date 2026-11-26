"""Quantum harmonic oscillator with Bohmian trajectories.

This experiment evolves the ground and first excited states of a
one-dimensional harmonic oscillator using the Crank-Nicolson method.

The numerical evolution is compared with analytical stationary states,
and Bohmian trajectories are integrated using the numerical wavefunctions.

Natural units are used: mass = hbar = angular frequency = 1.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from quantumforge.dynamics.trajectory import BohmianTrajectoryIntegrator
from quantumforge.numerics.schrodinger import SchrodingerSolver
from quantumforge.physics.potential import Potential
from quantumforge.physics.wavefunction import WaveFunction


def ground_state(x: np.ndarray) -> np.ndarray:
    """Return the analytical harmonic oscillator ground state."""
    return np.pi ** (-0.25) * np.exp(-0.5 * x**2)


def first_excited_state(x: np.ndarray) -> np.ndarray:
    """Return the analytical first excited state."""
    return np.sqrt(2.0) * x * ground_state(x)


def expectation_energy(
    wavefunction: WaveFunction,
    potential: Potential,
    mass: float,
    hbar: float,
) -> float:
    """Compute the expectation value of the Hamiltonian."""
    x = wavefunction.x
    psi = wavefunction.psi
    dx = x[1] - x[0]

    second_derivative = (
        psi[2:] - 2.0 * psi[1:-1] + psi[:-2]
    ) / dx**2

    kinetic_energy = (
        -hbar**2 / (2.0 * mass) * second_derivative
    )

    potential_values = potential.values[1:-1]

    hamiltonian_psi = kinetic_energy + potential_values * psi[1:-1]

    numerator = trapezoid(
        np.conjugate(psi[1:-1]) * hamiltonian_psi,
        x[1:-1],
    )

    denominator = trapezoid(
        np.abs(psi[1:-1]) ** 2,
        x[1:-1],
    )

    if denominator <= 0.0:
        raise ValueError("Wavefunction must have nonzero probability.")

    return float(np.real(numerator / denominator))

def main() -> None:
    """Run the harmonic oscillator experiment."""
    # Natural units.
    mass = 1.0
    hbar = 1.0
    angular_frequency = 1.0

    # Numerical parameters.
    domain_extent = 8.0
    spatial_points = 2001
    final_time = 10.0
    steps = 1000

    x = np.linspace(-domain_extent, domain_extent, spatial_points)
    dx = x[1] - x[0]
    dt = final_time / steps

    # Harmonic oscillator potential.
    potential = Potential(
        values=0.5 * mass * angular_frequency**2 * x**2,
        x=x,
    )


    # Analytical initial states.
    psi_0 = ground_state(x)
    psi_1 = first_excited_state(x)

    ground_wavefunction = WaveFunction(x=x, psi=psi_0)
    excited_wavefunction = WaveFunction(x=x, psi=psi_1)

    # Schrödinger solver.
    solver = SchrodingerSolver(
        mass=mass,
        potential=potential,
        dt=dt,
        hbar=hbar,
    )


    # Quantum evolution.
    ground_intervals = solver.evolve_with_midpoints(
        ground_wavefunction,
        steps=steps,
    )

    excited_intervals = solver.evolve_with_midpoints(
        excited_wavefunction,
        steps=steps,
    )

    ground_states = [ground_wavefunction] + [
        interval.final_state for interval in ground_intervals
    ]

    excited_states = [excited_wavefunction] + [
        interval.final_state for interval in excited_intervals
    ]

    times = np.linspace(0.0, final_time, steps + 1)

    # Energy expectation values.
    ground_energies = np.array(
        [
            expectation_energy(state, potential, mass, hbar)
            for state in ground_states
        ]
    )

    excited_energies = np.array(
        [
            expectation_energy(state, potential, mass, hbar)
            for state in excited_states
        ]
    )

    ground_energy_reference = 0.5 * hbar * angular_frequency
    excited_energy_reference = 1.5 * hbar * angular_frequency

    ground_energy_deviation = np.max(
        np.abs(ground_energies - ground_energies[0])
    )

    excited_energy_deviation = np.max(
        np.abs(excited_energies - excited_energies[0])
    )

    # Bohmian trajectory integration.
    integrator = BohmianTrajectoryIntegrator(
        mass=mass,
        dt=dt,
        hbar=hbar,
    )

    ground_initial_positions = np.linspace(-3.0, 3.0, 9)

    excited_initial_positions = np.array(
        [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]
    )

    ground_trajectories = np.asarray(
        [
            integrator.integrate(position, ground_intervals)
            for position in ground_initial_positions
        ]
    )

    excited_trajectories = np.asarray(
        [
            integrator.integrate(position, excited_intervals)
            for position in excited_initial_positions
        ]
    )


    # Analytical reference densities.
    ground_density = np.abs(psi_0) ** 2
    excited_density = np.abs(psi_1) ** 2

    numerical_ground_density = np.abs(ground_states[-1].psi) ** 2
    numerical_excited_density = np.abs(excited_states[-1].psi) ** 2

    # Probability normalization.
    initial_ground_probability = trapezoid(ground_density, x)
    final_ground_probability = trapezoid(numerical_ground_density, x)

    initial_excited_probability = trapezoid(excited_density, x)
    final_excited_probability = trapezoid(numerical_excited_density, x)

    # Density errors.
    ground_density_error = np.max(
        np.abs(numerical_ground_density - ground_density)
    )

    excited_density_error = np.max(
        np.abs(numerical_excited_density - excited_density)
    )

    # Bohmian trajectory deviations.
    ground_trajectory_deviation = np.max(
        np.abs(ground_trajectories - ground_initial_positions[:, None])
    )

    excited_trajectory_deviation = np.max(
        np.abs(excited_trajectories - excited_initial_positions[:, None])
    )


    # Visualization.
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))

    ground_density_axis, excited_density_axis = axes[0]
    ground_trajectory_axis, excited_trajectory_axis = axes[1]
    ground_energy_axis, excited_energy_axis = axes[2]


    selected_indices = np.linspace(
        0,
        steps,
        5,
        dtype=int,
    )

    # Ground state density.
    for index in selected_indices:
        ground_density_axis.plot(
            x,
            np.abs(ground_states[index].psi) ** 2,
            label=f"t = {times[index]:.1f}",
        )

    ground_density_axis.plot(
        x,
        ground_density,
        "--",
        color="black",
        label="Analytical",
    )

    ground_density_axis.set_title("Ground state density")
    ground_density_axis.set_xlabel("x")
    ground_density_axis.set_ylabel(r"$|\psi_0(x,t)|^2$")
    ground_density_axis.legend()
    ground_density_axis.grid(True)

    # First excited state density.
    for index in selected_indices:
        excited_density_axis.plot(
            x,
            np.abs(excited_states[index].psi) ** 2,
            label=f"t = {times[index]:.1f}",
        )

    excited_density_axis.plot(
        x,
        excited_density,
        "--",
        color="black",
        label="Analytical",
    )

    excited_density_axis.set_title("First excited state density")
    excited_density_axis.set_xlabel("x")
    excited_density_axis.set_ylabel(r"$|\psi_1(x,t)|^2$")
    excited_density_axis.legend()
    excited_density_axis.grid(True)

    # Ground state Bohmian trajectories.
    for trajectory in ground_trajectories:
        ground_trajectory_axis.plot(times, trajectory)

    ground_trajectory_axis.set_title("Ground state Bohmian trajectories")
    ground_trajectory_axis.set_xlabel("Time")
    ground_trajectory_axis.set_ylabel("Position")
    ground_trajectory_axis.grid(True)

    # First excited state Bohmian trajectories.
    for trajectory in excited_trajectories:
        excited_trajectory_axis.plot(times, trajectory)

    excited_trajectory_axis.set_title("First excited state Bohmian trajectories")
    excited_trajectory_axis.set_xlabel("Time")
    excited_trajectory_axis.set_ylabel("Position")
    excited_trajectory_axis.grid(True)

    # Ground state energy deviation.
    ground_energy_axis.plot(
        times,
        ground_energies - ground_energy_reference,
    )

    ground_energy_axis.axhline(
        0.0,
        linestyle="--",
        color="black",
        label="Analytical reference",
    )

    ground_energy_axis.set_title("Ground state energy deviation")
    ground_energy_axis.set_xlabel("Time")
    ground_energy_axis.set_ylabel(r"$\Delta E$")
    ground_energy_axis.legend()
    ground_energy_axis.grid(True)

    # First excited state energy deviation.
    excited_energy_axis.plot(
        times,
        excited_energies - excited_energy_reference,
    )

    excited_energy_axis.axhline(
        0.0,
        linestyle="--",
        color="black",
        label="Analytical reference",
    )

    excited_energy_axis.set_title("First excited state energy deviation")
    excited_energy_axis.set_xlabel("Time")
    excited_energy_axis.set_ylabel(r"$\Delta E$")
    excited_energy_axis.legend()
    excited_energy_axis.grid(True)

    fig.tight_layout()


    # Save figure.
    output_directory = Path(__file__).resolve().parent / "output"
    output_directory.mkdir(exist_ok=True)

    figure_path = output_directory / "harmonic_oscillator.png"
    fig.savefig(figure_path, dpi=300)

    plt.show()

    # Numerical report.
    print("Quantum harmonic oscillator experiment")
    print("--------------------------------------")
    print(f"Mass:                              {mass:.6f}")
    print(f"Reduced Planck constant:           {hbar:.6f}")
    print(f"Angular frequency:                 {angular_frequency:.6f}")
    print(f"Spatial step:                      {dx:.6e}")
    print(f"Time step:                         {dt:.6e}")
    print()
    print(f"Ground initial probability:        {initial_ground_probability:.12f}")
    print(f"Ground final probability:          {final_ground_probability:.12f}")
    print(f"Excited initial probability:       {initial_excited_probability:.12f}")
    print(f"Excited final probability:         {final_excited_probability:.12f}")
    print()
    print(f"Ground maximum density error:      {ground_density_error:.6e}")
    print(f"Excited maximum density error:     {excited_density_error:.6e}")
    print(f"Ground maximum trajectory drift:   {ground_trajectory_deviation:.6e}")
    print(f"Excited maximum trajectory drift:  {excited_trajectory_deviation:.6e}")
    print()
    print(f"Ground analytical energy:          {ground_energy_reference:.6f}")
    print(f"Ground numerical energy:          {ground_energies[0]:.6f}")
    print(f"Ground maximum energy deviation:  {ground_energy_deviation:.6e}")
    print()
    print(f"Excited analytical energy:        {excited_energy_reference:.6f}")
    print(f"Excited numerical energy:         {excited_energies[0]:.6f}")
    print(f"Excited maximum energy deviation: {excited_energy_deviation:.6e}")
    print(f"Figure saved to: {figure_path}")

if __name__ == "__main__":
    main()
