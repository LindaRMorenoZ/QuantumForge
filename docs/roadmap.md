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

## 2. Bohmian dynamics

- [x] Implement Bohmian trajectory integration using the midpoint method.
- [x] Generalize trajectory integration to arbitrary spatial dimensions.
- [x] Implement velocity interpolation for trajectory evaluation.
- [x] Add validation for initial positions, spatial grids, evolution intervals, and numerical inputs.
- [x] Add automated tests for trajectory integration and numerical behavior.
- [x] Validate trajectory evolution against the analytical free 1D Gaussian wavepacket.
- [x] Develop systematic trajectory convergence tests.

## 3. Quantum experiments

- [ ] Add free-particle Gaussian wave-packet experiments.
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

