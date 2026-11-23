# Architecture

## Separation of responsibilities

- `physics`: definitions of physical models and derived fields.
- `dynamics`: evolution equations and trajectory orchestration.
- `numerics`: discretization, integration, interpolation, and numerical diagnostics.
- `visualization`: rendering only; it must not own or silently modify physical calculations.

## Initial conventions to decide

Before implementing the first physical model, document:
- SI units or a clearly declared dimensionless convention;
- coordinate and array ordering;
- representation of complex-valued wavefunctions;
- boundary conditions and spatial domains;
- numerical tolerances and error reporting;
- deterministic handling of initial conditions.

## Scientific validation

Every numerical feature should have tests for:
- input validation and edge cases;
- analytic limiting cases where available;
- conservation or equivariance properties where applicable;
- convergence under refinement of spatial and temporal resolution.
