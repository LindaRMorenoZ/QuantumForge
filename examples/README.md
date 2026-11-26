# Examples

This directory contains self-contained, reproducible computational experiments built on QuantumForge.

Each example documents a physical model, its assumptions, numerical parameters, methods, and validation criteria. Experiments are intended to exercise and verify the computational tools provided by the library.

## Available experiments

### Free Gaussian wavepacket expansion

**Script:** `free_gaussian_expansion.py`

**Physical model:**
- Free particle in one dimension.
- Gaussian initial wavepacket.
- No external potential.

**Numerical method:**
- Crank–Nicolson time evolution.
- Second-order finite-difference spatial discretization.
- Bohmian trajectory integration using the numerical wavefunction.

**Validation:**
- Probability conservation.
- Comparison of numerical and analytical wavepacket widths.
- Comparison of numerical Bohmian trajectories against the analytical reference.

**Execution:**

```bash
python examples/free_gaussian_expansion.py
```

### Quantum harmonic oscillator

**Script:** `harmonic_oscillator.py`

**Physical model:**
- One-dimensional quantum harmonic oscillator.
- Ground state and first excited state.
- Harmonic potential: \(V(x)=\frac{1}{2}m\omega^2x^2\).
- Natural units: \(m=\hbar=\omega=1\).

**Numerical method:**
- Crank–Nicolson time evolution.
- Second-order finite-difference spatial discretization.
- Bohmian trajectory integration using numerical wavefunctions and the explicit midpoint method (RK2).

**Validation performed by the experiment:**
- Probability conservation for both states.
- Comparison of numerical densities with analytical stationary-state densities.
- Stationary Bohmian trajectories.
- Energy conservation throughout numerical evolution.
- Comparison with analytical energy eigenvalues.

**Additional automated validation:**
- Second-order spatial convergence of the ground-state energy.
- Second-order temporal convergence against the discrete Hamiltonian eigenstate reference.
- Verification of analytical energy eigenvalues and eigenstate properties.
- Orthogonality and normalization of analytical states.

**Execution:**

```bash
python examples/harmonic_oscillator.py
```
Figures are saved in `examples/output/`.

## Scope

These experiments are computational demonstrations and numerical validation cases. They do not imply that all physical models, dimensions, or regimes are supported by QuantumForge.

Additional experiments will be included as the corresponding computational capabilities are developed and validated.
