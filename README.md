# QuantumForge

**QuantumForge** is a computational laboratory for exploring quantum systems, initially using the de Broglie–Bohm (Bohmian) formulation of quantum mechanics.

The project focuses on reliable numerical methods, reproducible simulations, and testable scientific software. It is a computational project: theoretical explanations, mathematical derivations, and pedagogical material belong in the separate physics-education website.

## Project status

Early repository scaffold. The numerical physics engine is **not implemented yet**.

## Design principles

- Physical models and numerical algorithms remain separate from visualization.
- Numerical behavior must be testable and reproducible.
- Assumptions, units, and model limitations must be documented.
- New physics modules require validation tests before being treated as reliable.
- No claim is made that the Bohmian formulation resolves every foundational problem in quantum mechanics.

## Repository layout

```text
src/quantumforge/
├── physics/        # Physical quantities and model definitions
├── dynamics/      # Time evolution and trajectory engines
├── numerics/      # Numerical algorithms and integration methods
└── visualization/ # Plotting and interactive visualization

tests/              # Automated tests
examples/           # Reproducible computational experiments
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

## Current scope

The initial development sequence is expected to be:

1. Define conventions for units, coordinates, arrays, and numerical tolerances.
2. Implement and test wavefunction representations.
3. Implement probability density and probability current.
4. Implement the Bohmian velocity field under clearly stated assumptions.
5. Implement trajectory integration and convergence checks.
6. Add validated examples such as free Gaussian wave packets and interference.

This is a roadmap, not a list of implemented features.

## License

License to be decided before public release.
