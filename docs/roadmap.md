# Development roadmap

This roadmap describes the current development status and intended future work of QuantumForge.

## 1. Physical and numerical foundations

- [x] Establish physical and numerical conventions.
- [x] Add wavefunction representation and validation.
- [x] Add probability density and probability current.
- [x] Add the Bohmian velocity field with explicit assumptions and singularity handling.
- [x] Add one-dimensional external potential representation.
- [x] Implement the one-dimensional finite-difference Laplacian.
- [x] Implement the one-dimensional Hamiltonian operator.
- [x] Implement a one-dimensional Crank–Nicolson Schrödinger solver.
- [x] Validate norm preservation, boundary conditions, eigenstate evolution, and temporal convergence.
- [x] Validate free Gaussian wave-packet evolution against its analytical solution.
- [x] Validate probability conservation and analytical wave-packet spreading in SI units.
- [x] Verify second-order temporal and spatial convergence for free Gaussian evolution.

## 2. Bohmian dynamics

- [x] Implement Bohmian trajectory integration using the midpoint method.
- [x] Generalize trajectory integration to arbitrary spatial dimensions.
- [x] Implement velocity interpolation for trajectory evaluation.
- [x] Add validation for initial positions, spatial grids, evolution intervals, and numerical inputs.
- [x] Add automated tests for trajectory integration and numerical behavior.
- [x] Validate trajectory evolution against the analytical free 1D Gaussian wavepacket.
- [x] Develop systematic trajectory convergence tests.
- [x] Validate SI-scale temporal intervals and continuity handling.
- [x] Validate the connection between Schrödinger evolution intervals and Bohmian trajectory integration for a free Gaussian wavepacket.

## 3. Quantum experiments

- [x] Establish an analytical reference for free-particle Gaussian wave-packet evolution.
- [x] Validate the free Gaussian wave-packet as a computational benchmark for the Schrödinger solver and Bohmian trajectory integrator.
- [ ] Develop a reproducible free-particle Gaussian wave-packet experiment.
- [ ] Add coherent superpositions and interference experiments.
- [ ] Add a double-slit model after the preceding components are validated.

## 4. Higher-dimensional extensions

- [ ] Generalize spatial operators and Schrödinger evolution to two dimensions.
- [ ] Validate two-dimensional probability density and current evolution.
- [x] Implement multidimensional Bohmian trajectory integration.
- [ ] Validate multidimensional trajectories against analytical or independently established reference cases.
- [ ] Develop the double-slit computational benchmark.

## 5. Future extensions

- [ ] Consider time-dependent potentials.
- [ ] Consider many-particle extensions.
- [ ] Explore additional numerical methods after the foundations are stable.

This roadmap is subject to revision as the computational laboratory develops. Theoretical explanations, mathematical derivations, and educational material belong to the separate Fundamental Physics platform.
