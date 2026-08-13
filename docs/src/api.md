```@meta
CurrentModule = InterfacialWaves
```

# [API Reference](@id api_reference)

## Problems and methods

```@docs
GravityProblem
GCProblem
ViscousGCProblem
ViscousGCFixedBProblem
LH
Collocation
WaveSolution
LinearStabProblem
StabilityResult
```

## Longuet–Higgins formulation

```@docs
LHResidual!
LHJacobian!
lh_phase_speed
lh_phase_speed_half
```

## Stability utilities

```@docs
max_growth_rate
unstable_modes
is_unstable
```

## Spectral and continuation infrastructure

```@docs
WaveGrid
SpectralWorkspace
continuation
initial_energy
```

## Index

```@index
```
