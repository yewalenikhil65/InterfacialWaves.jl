```@meta
CurrentModule = InterfacialWaves
```

# [Travelling Waves](@id travelling_waves)

A travelling-wave solution is a steady profile in a frame moving with a
constant phase speed. In the package, a base state is returned as a
[`WaveSolution`](@ref WaveSolution), which stores the continuation states, return codes,
parameter schedule, grid, and originating problem.

## Current formulations

- [Pure Gravity Waves](@ref pure_gravity_waves): first the Longuet–Higgins
  Fourier-coefficient formulation, followed by conformal-mapping collocation;
- [Gravity–Capillary Waves](@ref gravity_capillary_waves): exact fixed-B target
  states with full translational invariance;
- [Viscous Gravity–Capillary Waves](@ref viscous_gravity_capillary_waves):
  finite-Reynolds-number, finite-pressure continuation paths;
- [Interfacial Waves](@ref interfacial_waves): two-fluid formulation and current
  implementation status.

## Common numerical pattern

For continuation-based methods, an initial small-amplitude state is followed by
nonlinear solves at a sequence of continuation parameters. The converged state
at one parameter value is used as the initial guess for the next. The selected
continuation parameter depends on the formulation:

- `ak` for the Longuet–Higgins pure-gravity formulation;
- energy `ε` for collocation and viscous formulations;
- a fixed target energy for the current gravity–capillary solve.
