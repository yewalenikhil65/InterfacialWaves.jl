```@meta
CurrentModule = InterfacialWaves
```

# [InterfacialWaves.jl](@id home)

![Stokes drift](assets/logo_anim.gif)

`InterfacialWaves.jl` is a Julia package for computing exact steady travelling-wave
solutions and their linear stability. The package is organized around the wave
class being solved:

- **Travelling Waves**: pure-gravity, gravity–capillary, viscous gravity–capillary/
  wind-forced, and two-fluid interfacial waves;
- **Standing Waves**: axisymmetric standing waves in a cylindrical basin via
  HOSE boundary-integral methods (`AxiStandingWaves` submodule);
- **Stability**: linear stability problems built from a computed base state.

The documentation distinguishes between implemented methods and planned work.
Methods whose base-state or stability implementation is not yet available are
listed under [Progress](@ref progress).

## Citing this package

If you use `InterfacialWaves.jl` in your research, please cite:

```bibtex
@misc{InterfacialWaves,
  author  = {Yewale, Nikhil and Dasgupta, Ratul},
  title   = {{InterfacialWaves.jl}: A {Julia} Package for Nonlinear Interfacial Waves},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/yewalenikhil65/InterfacialWaves.jl},
  note    = {Julia package, v0.1.0},
}
```

## Quick start: pure-gravity waves

```@example home
using InterfacialWaves

prob = GravityProblem(128; ak=0.20)
sol = solve(prob, LH())

sol.c       # phase speed
sol.ak      # final steepness
sol.a       # Fourier coefficients in the a₀/2 convention
```

The pure-gravity tutorial develops the Longuet–Higgins calculation first,
then the collocation calculation, and finally compares their phase-speed
curves:

- [Pure Gravity Waves](@ref pure_gravity_waves)

## Documentation map

- [Travelling Waves](@ref travelling_waves)
- [Standing Waves](@ref standing_waves)
- [Stability](@ref stability)
- [Progress](@ref progress)
- [API Reference](@ref api_reference)
- [References](@ref references)
