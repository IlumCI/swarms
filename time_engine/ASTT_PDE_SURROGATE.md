ASTT PDE / Surrogate Adapter Guide
==================================

This file describes the minimal adapter API used by the Time Engine to plug PDE solvers or surrogate models.

Adapter interfaces
- PDEAdapter: implement `step(local_grid, dt, params) -> local_grid` where `local_grid` is a 1D array/list representing a field tile.
- SurrogateAdapter: implement `predict(local_state, dt) -> local_state` where `local_state` is a small state vector.

Factory
- `get_physics_adapter(phidesc)` returns an adapter instance by `phidesc['field_model']` (e.g., 'pde_cpu', 'surrogate_dummy').

Integration notes
- `evolve_state` will call the adapter when `phidesc` contains a `field_model`. The adapter must accept and return arrays compatible with `resolution_config.field_variables`.

CPU fallback
- `PDEAdapterCpu` provides an explicit diffusion stencil example.
- `SurrogateAdapterDummy` provides a linear-scaling placeholder.

How to add a GPU adapter
- Create a class `PDEAdapterGpu` implementing `step(...)`, wrap your GPU kernels, and register in `get_physics_adapter`.

License: internal research usage.


