# QuantumForge

**QuantumForge** is a computational laboratory for exploring quantum systems, with de Broglie–Bohm (Bohmian) mechanics as its foundational framework.

The project aims to provide modular, testable, and reproducible numerical tools for investigating quantum dynamics, wavefunctions, velocity fields, and particle trajectories.

QuantumForge is a computational laboratory, not a theoretical or pedagogical platform. Mathematical derivations, theoretical discussions, and educational material belong in the separate Fundamental Physics website.

## Project status

QuantumForge is under active development. The initial computational foundations are implemented, including wavefunction representations, physical quantities, numerical operators, Schrödinger evolution, and Bohmian trajectory integration.

The current implementation is evolving toward a modular framework for computational experiments in quantum mechanics.

## Design principles

- Physical models, numerical algorithms, and visualization remain modular and independent.
- Numerical behavior must be testable and reproducible.
- Spatial dimensionality is treated independently of the underlying numerical implementation.
- Assumptions, units, and model limitations must be documented.
- New physics modules require validation tests before being treated as reliable.
- Computational flexibility is preferred, with explicit validation and warnings when appropriate.
- No claim is made that the Bohmian formulation resolves every foundational problem in quantum mechanics.

## Implemented foundations

- Wavefunction representation and spatial grids.
- Probability density and quantum physical quantities.
- Physical constants and potential models.
- Numerical differential operators.
- Bohmian velocity calculations.
- Velocity interpolation on spatial grids.
- One-dimensional Schrödinger time evolution using numerical integration methods.
- Evolution intervals containing initial, midpoint, and final wavefunction states.
- Bohmian trajectory integration using the midpoint method.
- Trajectory integration in arbitrary spatial dimensions, subject to the supported numerical components.
- Validation of physical parameters, spatial grids, and numerical inputs.

## Repository layout

```text
src/quantumforge/
├── physics/        # Wavefunctions, potentials, constants and physical quantities
├── dynamics/       # Evolution intervals and Bohmian trajectory integration
├── numerics/       # Operators and numerical Schrödinger evolution
└── visualization/  # Visualization tools

tests/              # Automated tests
examples/           # Computational experiments
docs/               # Technical design notes
```

## Development setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

## Development roadmap

The development roadmap includes:

1. Expand and validate the numerical physics foundations.
2. Extend Schrödinger evolution beyond the current one-dimensional implementation.
3. Develop and validate multidimensional quantum dynamics.
4. Improve trajectory integration, numerical stability, and convergence analysis.
5. Develop reproducible computational experiments.
6. Implement visualization tools for wavefunctions, probability distributions, velocity fields, and trajectories.
7. Expand numerical tests and physical validation benchmarks.

## Scope

QuantumForge is intended for computational exploration of quantum systems. Its foundational approach is de Broglie–Bohm mechanics, while the software architecture is designed to remain modular and extensible.

The implementation is not intended to establish the universal validity or superiority of any interpretation of quantum mechanics.

## License

License to be decided before public release.

