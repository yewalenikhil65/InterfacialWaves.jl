# InterfacialWaves.jl

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://yewalenikhil65.github.io/InterfacialWaves.jl/)

![Stokes drift animation](docs/src/assets/logo_anim.gif)

A Julia package for computing exact steady travelling-wave solutions and their linear stability.

## Wave types

- **Pure gravity** — Longuet–Higgins Fourier-coefficient and conformal-mapping collocation formulations
- **Gravity–capillary** — fixed-Bond-number target states
- **Viscous gravity–capillary** — finite-Reynolds-number derived-B and fixed-B formulations
- **Interfacial** — two-fluid Stokes waves with density ratio ρ₁/ρ₂ < 1
- **Axisymmetric standing waves** — HOSE boundary-integral method in a cylindrical basin (`AxiStandingWaves`)

## Quick start

```julia
using InterfacialWaves

# Pure gravity wave at ak = 0.40
sol = solve(GravityProblem(128; ak=0.40), LH())
sol.c    # phase speed
sol.ak   # steepness
sol.a    # Fourier coefficients

# Linear stability
stab   = LinearStabProblem(base_state=sol, m=1, n_choose=100)
result = solve(stab)
max_growth_rate(result)
is_unstable(result)
```

## Documentation

Full documentation including formulation derivations, solver details, and API reference:

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://yewalenikhil65.github.io/InterfacialWaves.jl/)

## Citation

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
