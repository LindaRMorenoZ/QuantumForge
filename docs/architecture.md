# Architecture

## Separation of responsibilities

- `physics`: definitions of physical models and derived fields.
- `dynamics`: evolution equations and trajectory orchestration.
- `numerics`: discretization, integration, interpolation, and numerical diagnostics.
- `visualization`: rendering only; it must not own or silently modify physical calculations.

## Numerical and physical conventions

- Physical parameters use SI units unless an experiment explicitly declares a dimensionless convention.
- Wavefunctions are represented as complex-valued NumPy arrays.
- Spatial coordinates are represented by one-dimensional coordinate arrays per spatial dimension.
- Multidimensional arrays follow NumPy axis ordering, with one axis per spatial dimension.
- Spatial domains and boundary conditions are specified explicitly by each numerical model.
- Numerical methods must validate their input parameters, spatial grids, and temporal intervals.
- Numerical errors and convergence behavior should be assessed through analytical references and systematic refinement studies whenever possible.
- Initial conditions and experiment parameters must be explicitly specified to ensure reproducibility.

## Scientific validation

Every numerical feature should be tested for:

- input validation, edge cases, and numerical stability;
- analytical limiting cases and reference solutions where available;
- conservation or equivariance properties where applicable;
- convergence under spatial and temporal refinement;
- reproducibility across controlled numerical configurations.

Convergence results must be interpreted within the scope of the tested model and numerical configuration. Observed convergence orders in particular cases must not be presented as general properties of a numerical method without additional evidence.
