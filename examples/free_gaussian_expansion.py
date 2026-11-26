"""Free expansion of a Gaussian wave packet with Bohmian trajectories.

This experiment evolves a free Gaussian wave packet in one dimension using
the Crank-Nicolson method and integrates Bohmian trajectories guided by the
numerically evolved wavefunction.

The numerical results are compared with analytical solutions for the free
Gaussian wave packet.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantumforge.dynamics.trajectory import BohmianTrajectoryIntegrator
from quantumforge.numerics.schrodinger import SchrodingerSolver
from quantumforge.physics.constants import SI_CONSTANTS
from quantumforge.physics.potential import Potential
from quantumforge.physics.wavefunction import WaveFunction


def analytical_width(time: float, sigma_0: float, dispersion_time: float) -> float:
    """Return the analytical Gaussian width at a given time."""
    return sigma_0 * np.sqrt(1.0 + (time / dispersion_time) ** 2)


def analytical_trajectory(
    initial_position: np.ndarray,
    time: float,
    sigma_0: float,
    dispersion_time: float,
) -> np.ndarray:
    """Return analytical Bohmian trajectories for a free Gaussian packet."""
    return initial_position * np.sqrt(1.0 + (time / dispersion_time) ** 2)


def main() -> None:
    # Physical parameters (SI units).
    mass = SI_CONSTANTS.electron_mass
    hbar = SI_CONSTANTS.hbar
    sigma_0 = 1e-9
    x_0 = 0.0
    p_0 = 0.0

    dispersion_time = 2.0 * mass * sigma_0**2 / hbar
    final_time = dispersion_time

    # Numerical parameters.
    domain_extent = 12.0 * sigma_0
    spatial_points = 4801
    steps = 200

    x = np.linspace(-domain_extent, domain_extent, spatial_points)
    dx = x[1] - x[0]
    dt = final_time / steps

    # Initial Gaussian wave packet.
    psi_0 = (
        (2.0 * np.pi * sigma_0**2) ** (-0.25)
        * np.exp(-(x - x_0) ** 2 / (4.0 * sigma_0**2))
        * np.exp(1j * p_0 * (x - x_0) / hbar)
    )

    wavefunction = WaveFunction(x=x, psi=psi_0)

    # Free-particle evolution.
    potential = Potential(values=np.zeros_like(x), x=x)
    solver = SchrodingerSolver(
        mass=mass,
        potential=potential,
        dt=dt,
        hbar=hbar,
    )

    intervals = solver.evolve_with_midpoints(
        wavefunction,
        steps=steps,
    )

    states = [wavefunction] + [interval.final_state for interval in intervals]
    times = np.linspace(0.0, final_time, steps + 1)

    # Bohmian trajectories.
    initial_positions = np.linspace(-3.0 * sigma_0, 3.0 * sigma_0, num=9)

    integrator = BohmianTrajectoryIntegrator(
        mass=mass,
        dt=dt,
        hbar=hbar,
    )

    trajectories = [
        integrator.integrate(
            initial_position=position,
            intervals=intervals,
        )
        for position in initial_positions
    ]

    trajectories = np.asarray(trajectories)

    # Analytical references.
    analytical_final_width = analytical_width(
        final_time,
        sigma_0,
        dispersion_time,
    )

    analytical_positions = np.asarray(
        [
            analytical_trajectory(
                initial_positions,
                time,
                sigma_0,
                dispersion_time,
            )
            for time in times
        ]
    ).T

    numerical_final_density = np.abs(states[-1].psi) ** 2
    analytical_final_density = (
        1.0
        / np.sqrt(2.0 * np.pi * analytical_final_width**2)
        * np.exp(
            -(x - x_0) ** 2
            / (2.0 * analytical_final_width**2)
        )
    )

    numerical_width = np.sqrt(
        np.trapezoid(
            (x - x_0) ** 2 * numerical_final_density,
            x,
        )
    )

    # Visualization.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    density_axis, trajectory_axis = axes

    selected_indices = np.linspace(
        0,
        steps,
        5,
        dtype=int,
    )

    for index in selected_indices:
        density_axis.plot(
            x / sigma_0,
            np.abs(states[index].psi) ** 2 * sigma_0,
            label=f"t/t_d = {times[index] / dispersion_time:.2f}",
        )

    density_axis.plot(
        x / sigma_0,
        analytical_final_density * sigma_0,
        "--",
        color="black",
        label="Analytical final density",
    )

    density_axis.set_xlabel(r"$x/\sigma_0$")
    density_axis.set_ylabel(r"$|\psi|^2\sigma_0$")
    density_axis.set_title("Free Gaussian wave-packet expansion")
    density_axis.legend()
    density_axis.grid(True)

    for numerical, analytical in zip(
        trajectories,
        analytical_positions,
        strict=True,
    ):
        trajectory_axis.plot(
            times / dispersion_time,
            numerical / sigma_0,
        )

        trajectory_axis.plot(
            times / dispersion_time,
            analytical / sigma_0,
            "--",
            alpha=0.5,
        )

    trajectory_axis.set_xlabel(r"$t/t_d$")
    trajectory_axis.set_ylabel(r"$X(t)/\sigma_0$")
    trajectory_axis.set_title("Bohmian trajectories")
    trajectory_axis.grid(True)

    fig.tight_layout()

    output_directory = Path(__file__).resolve().parent / "output"
    output_directory.mkdir(exist_ok=True)

    figure_path = output_directory / "free_gaussian_expansion.png"
    fig.savefig(figure_path, dpi=300)

    plt.show()

    # Numerical report.
    initial_probability = np.trapezoid(
        np.abs(wavefunction.psi) ** 2,
        x,
    )

    final_probability = np.trapezoid(
        numerical_final_density,
        x,
    )

    trajectory_error = np.max(
        np.abs(trajectories - analytical_positions)
    )

    print("Free Gaussian expansion experiment")
    print("--------------------------------")
    print(f"Electron mass:           {mass:.6e} kg")
    print(f"Initial width:           {sigma_0:.6e} m")
    print(f"Dispersion time:         {dispersion_time:.6e} s")
    print(f"Time step:               {dt:.6e} s")
    print(f"Spatial step:            {dx:.6e} m")
    print(f"Initial probability:      {initial_probability:.12f}")
    print(f"Final probability:        {final_probability:.12f}")
    print(f"Numerical final width:   {numerical_width:.6e} m")
    print(f"Analytical final width:   {analytical_final_width:.6e} m")
    width_relative_error = abs(numerical_width - analytical_final_width) / analytical_final_width
    print(f"Width relative error:     {width_relative_error:.6e}")
    print(f"Maximum trajectory error: {trajectory_error:.6e} m")
    print(f"Figure saved to:          {figure_path}")


if __name__ == "__main__":
    main()
